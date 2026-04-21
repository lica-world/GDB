"""Claude Code CLI agent, exposed as a first-class GDB model.

This wrapper lets the standard GDB harness (``scripts/run_benchmarks.py``,
``BenchmarkRunner``, CSV/JSON reporting, etc.) drive Anthropic's Claude Code
agent instead of a single-shot API model. It exists to enable apples-to-apples
parity with the Harbor adapter: the same agent, same model, same per-task
instruction format, run on both sides.

The agent works by, for each sample:
  1. Creating an isolated temp workdir.
  2. Staging any input images under ``workspace/inputs/``.
  3. Writing an instruction that mirrors the Harbor task's ``instruction.md``:
     "Write your answer to ``/workspace/<output_file>``" where ``<output_file>``
     is determined by the benchmark id (matching the Harbor adapter's
     ``OUTPUT_FILES`` mapping).
  4. Invoking ``claude --print`` non-interactively in that workdir.
  5. Reading the resulting file back and returning it as a ``ModelOutput``.

Text outputs (``.txt``/``.svg``/``.json``) are returned via ``ModelOutput.text``.
Media outputs (``.png``/``.mp4``) are persisted under ``output_dir`` (or the
tempdir, if none was provided) and returned via ``ModelOutput.images``.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from .base import BaseModel, Modality, ModelInput, ModelOutput
from .registry import register_model

logger = logging.getLogger(__name__)


# Mirrors adapters/gdb/src/gdb_adapter/adapter.py::OUTPUT_FILES in the Harbor
# fork. Keep this in sync so agent-vs-agent parity is exact.
_OUTPUT_FILES: Dict[str, str] = {
    "svg-1": "answer.txt",
    "svg-2": "answer.txt",
    "category-1": "answer.txt",
    "category-2": "answer.txt",
    "layout-4": "answer.txt",
    "layout-5": "answer.txt",
    "layout-6": "answer.txt",
    "template-1": "answer.txt",
    "temporal-2": "answer.txt",
    "typography-1": "answer.txt",
    "typography-2": "answer.txt",
    "template-2": "answer.txt",
    "template-3": "answer.txt",
    "svg-3": "answer.svg",
    "svg-4": "answer.svg",
    "svg-5": "answer.svg",
    "svg-6": "answer.svg",
    "svg-7": "answer.svg",
    "svg-8": "answer.svg",
    "typography-3": "answer.json",
    "typography-4": "answer.json",
    "typography-5": "answer.json",
    "typography-6": "answer.json",
    "temporal-1": "answer.json",
    "temporal-3": "answer.json",
    "layout-7": "answer.json",
    "layout-2": "answer.json",
    "layout-3": "answer.json",
    "template-4": "answer.json",
    "template-5": "answer.json",
    "lottie-1": "answer.json",
    "lottie-2": "answer.json",
    "layout-1": "output.png",
    "layout-8": "output.png",
    "typography-7": "output.png",
    "typography-8": "output.png",
    "temporal-4": "output.mp4",
    "temporal-5": "output.mp4",
    "temporal-6": "output.mp4",
}

_DEFAULT_ALLOWED_TOOLS = "Bash,Read,Write,Edit,LS,Glob,Grep"
_MEDIA_EXTS = {".png", ".jpg", ".jpeg", ".mp4", ".webm"}

# Benchmarks whose output_file is an image: if the agent emits an SVG or a
# text/JSON fallback, the harness rasterizes it to the expected PNG so the
# downstream evaluator (OCR, NIMA) sees a real image.
_IMAGE_BENCHMARKS = {"layout-1", "layout-8", "typography-7", "typography-8"}


@register_model("claude_code")
class ClaudeCodeAgent(BaseModel):
    """Run the Claude Code CLI as a GDB model.

    The underlying API model is selected via ``model_id`` (e.g.
    ``claude-sonnet-4-20250514``). Requires ``ANTHROPIC_API_KEY`` and a working
    ``claude`` CLI on ``PATH`` (see ``parity/claude-code-setup.sh``).
    """

    modality = Modality.ANY
    supports_image_input = True
    supports_image_output = True
    supports_video_output = True
    supports_mask_editing = False

    def __init__(
        self,
        model_id: str = "claude-sonnet-4-20250514",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        timeout_sec: int = 1800,
        allowed_tools: str = _DEFAULT_ALLOWED_TOOLS,
        output_dir: Optional[str] = None,
        cli_cmd: str = "claude",
        default_output_file: str = "answer.txt",
        **kwargs: Any,
    ) -> None:
        self.model_id = model_id
        self.name = f"claude-code@{model_id}"
        # Accepted for harness uniformity; Claude Code does not expose them.
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_sec = timeout_sec
        self.allowed_tools = allowed_tools
        self.cli_cmd = cli_cmd
        self.default_output_file = default_output_file
        self.output_dir = Path(output_dir).resolve() if output_dir else None
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def predict(self, inp: ModelInput) -> ModelOutput:
        bid = (inp.metadata or {}).get("benchmark_id") or ""
        sample_id = (inp.metadata or {}).get("sample_id") or ""
        output_file = _OUTPUT_FILES.get(bid, self.default_output_file)

        work_parent = Path(tempfile.mkdtemp(prefix="gdb-claude-"))
        workdir = work_parent / "workspace"
        workdir.mkdir(parents=True)

        try:
            staged = self._stage_images(workdir, inp.images or [])
            instruction = self._build_instruction(
                bid=bid,
                user_text=inp.text or "",
                staged_rel_paths=staged,
                output_file=output_file,
            )

            proc_info = self._run_claude(workdir, instruction)

            out_path = workdir / output_file
            if not out_path.is_file() and bid in _IMAGE_BENCHMARKS:
                rendered = self._rasterize_image_output(workdir, out_path)
                if rendered is not None:
                    out_path = rendered
            if not out_path.is_file():
                out_path = self._find_any_output(workdir)

            if out_path is None or not out_path.is_file():
                logger.warning(
                    "claude-code produced no output for %s/%s (rc=%s)",
                    bid or "?",
                    sample_id or "?",
                    proc_info.get("returncode"),
                )
                return ModelOutput(
                    text="",
                    usage={
                        "error": "no_output_file",
                        "returncode": proc_info.get("returncode"),
                        "stderr": (proc_info.get("stderr") or "")[:500],
                    },
                )

            return self._materialize(out_path, bid=bid, sample_id=sample_id)

        finally:
            shutil.rmtree(work_parent, ignore_errors=True)

    @staticmethod
    def _stage_images(workdir: Path, images: list) -> list:
        inputs_dir = workdir / "inputs"
        staged: list = []
        for i, img in enumerate(images):
            if not isinstance(img, (str, Path)):
                # In-memory bytes / PIL images are rare in GDB's pipeline;
                # skipping keeps the agent side a pure path-based contract.
                continue
            src = Path(img)
            if not src.is_file():
                continue
            inputs_dir.mkdir(exist_ok=True)
            suffix = src.suffix or ".bin"
            dst = inputs_dir / f"input_{i}{suffix}"
            shutil.copy2(src, dst)
            staged.append(f"inputs/input_{i}{suffix}")
        return staged

    def _build_instruction(
        self,
        *,
        bid: str,
        user_text: str,
        staged_rel_paths: list,
        output_file: str,
    ) -> str:
        lines: list = []
        if bid:
            lines += [f"# GDB: {bid}", ""]

        if staged_rel_paths:
            lines += ["## Input Files", ""]
            lines += [f"- `{p}`" for p in staged_rel_paths]
            lines += [""]

        lines += ["## Task", "", user_text.strip(), ""]
        lines += [
            "## Output",
            "",
            f"Write your answer to `{output_file}` in the current directory.",
            "Write ONLY the answer — no explanation, no markdown fences, no extra text.",
        ]
        if bid in _IMAGE_BENCHMARKS:
            lines += [
                "",
                "### Image output requirements",
                "",
                "- You MUST produce a real rasterized image at `output.png`.",
                "- `python3` is available. `Pillow`, `cairosvg`, and `numpy` are "
                "installed in the active environment and importable.",
                "- Preferred approach: build an SVG describing the design, then "
                "rasterize it to `output.png` with "
                "`python3 -c \"import cairosvg; "
                "cairosvg.svg2png(url='design.svg', "
                "write_to='output.png', output_width=1024)\"`.",
                "- Alternatively, use Pillow's `ImageDraw`/`ImageFont` to draw "
                "directly and save a PNG.",
                "- Do NOT write natural-language descriptions or SVG-as-text into "
                "`output.png` — the file must be a valid image.",
            ]
        lines += [""]
        return "\n".join(lines)

    def _run_claude(self, workdir: Path, instruction: str) -> Dict[str, Any]:
        env = dict(os.environ)
        env.setdefault("ANTHROPIC_MODEL", self.model_id)
        # Disable keychain/oauth probing; parity runs are headless.
        env.setdefault("FORCE_AUTO_BACKGROUND_TASKS", "1")
        env.setdefault("ENABLE_BACKGROUND_TASKS", "1")
        # On macOS, cairosvg's ctypes loader can't find Homebrew's libcairo
        # unless DYLD_FALLBACK_LIBRARY_PATH includes /opt/homebrew/lib.
        # Appending is idempotent and harmless on Linux.
        if "DYLD_FALLBACK_LIBRARY_PATH" in env:
            if "/opt/homebrew/lib" not in env["DYLD_FALLBACK_LIBRARY_PATH"]:
                env["DYLD_FALLBACK_LIBRARY_PATH"] = (
                    "/opt/homebrew/lib:" + env["DYLD_FALLBACK_LIBRARY_PATH"]
                )
        else:
            env["DYLD_FALLBACK_LIBRARY_PATH"] = "/opt/homebrew/lib"

        cmd = [
            self.cli_cmd,
            "--print",
            "--model",
            self.model_id,
            "--allowedTools",
            self.allowed_tools,
        ]

        try:
            proc = subprocess.run(
                cmd,
                input=instruction,
                cwd=str(workdir),
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
            )
            return {
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"{self.cli_cmd!r} CLI not found on PATH. "
                "Install via parity/claude-code-setup.sh or "
                "`npm install -g @anthropic-ai/claude-code`."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            return {
                "returncode": -1,
                "stdout": (exc.stdout or "") if hasattr(exc, "stdout") else "",
                "stderr": f"timeout after {self.timeout_sec}s",
            }

    @staticmethod
    def _rasterize_image_output(workdir: Path, target: Path) -> Optional[Path]:
        """Rasterize an SVG (or SVG-bearing text file) to `target`.

        Called when the agent was asked for `output.png` but only wrote an SVG
        or a text/json file containing an SVG. Uses cairosvg + Pillow.
        """
        import re

        svg_re = re.compile(r"<svg\b.*?</svg>", re.DOTALL | re.IGNORECASE)

        candidates: list[Path] = []
        for name in ("output.svg", "answer.svg", "design.svg"):
            p = workdir / name
            if p.is_file():
                candidates.append(p)
        for ext in (".txt", ".json"):
            for p in sorted(workdir.glob(f"answer{ext}")):
                if p.is_file():
                    candidates.append(p)
            for p in sorted(workdir.glob(f"output{ext}")):
                if p.is_file():
                    candidates.append(p)

        svg_text: Optional[str] = None
        for p in candidates:
            try:
                content = p.read_text(errors="replace")
            except OSError:
                continue
            if content.lstrip().startswith("<?xml") or "<svg" in content.lower():
                m = svg_re.search(content)
                svg_text = m.group(0) if m else content
                break

        if not svg_text:
            return None

        try:
            import cairosvg
        except Exception as exc:
            logger.warning("cairosvg unavailable for post-render: %s", exc)
            return None

        try:
            cairosvg.svg2png(
                bytestring=svg_text.encode("utf-8"),
                write_to=str(target),
                output_width=1024,
            )
        except Exception as exc:
            logger.warning("cairosvg rasterization failed: %s", exc)
            return None

        return target if target.is_file() else None

    @staticmethod
    def _find_any_output(workdir: Path) -> Optional[Path]:
        for name in (
            "output.mp4",
            "output.webm",
            "output.png",
            "output.jpg",
            "output.jpeg",
        ):
            p = workdir / name
            if p.is_file():
                return p
        for ext in (".txt", ".svg", ".json"):
            p = workdir / f"answer{ext}"
            if p.is_file():
                return p
        for pattern in ("answer.*", "output.*"):
            for p in sorted(workdir.glob(pattern)):
                if p.is_file():
                    return p
        return None

    def _materialize(
        self,
        out_path: Path,
        *,
        bid: str,
        sample_id: str,
    ) -> ModelOutput:
        suffix = out_path.suffix.lower()

        if suffix in _MEDIA_EXTS:
            # Must live OUTSIDE the per-call work_parent (which the caller will
            # rmtree). Fall back to a stable path under the system tempdir.
            dst_dir = self.output_dir or (
                Path(tempfile.gettempdir()) / "gdb-claude-code-outputs"
            )
            dst_dir.mkdir(parents=True, exist_ok=True)
            stem = f"{bid or 'gdb'}_{sample_id or uuid.uuid4().hex[:8]}"
            dst = dst_dir / f"{stem}_{uuid.uuid4().hex[:6]}{suffix}"
            try:
                shutil.copy2(out_path, dst)
            except OSError:
                # If copy fails we still cannot point at the tempdir file, so
                # bail out gracefully.
                return ModelOutput(
                    text="",
                    usage={"error": "media_copy_failed", "source": str(out_path)},
                )
            return ModelOutput(images=[str(dst)], text=str(dst))

        return ModelOutput(text=out_path.read_text(errors="replace"))
