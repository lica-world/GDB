"""SVG benchmarks: svg-1 through svg-8.

Data contract: each task reads ``{task-id}.json`` from the ``--data`` directory.
The JSON schema varies by task — see ``load_data`` in each class.
"""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Union

from design_benchmarks.base import BaseBenchmark, BenchmarkMeta, TaskType, benchmark
from design_benchmarks.utils.data_helpers import load_task_json
from design_benchmarks.utils.text_helpers import (
    normalized_edit_distance,
    strip_thinking,
)

logger = logging.getLogger(__name__)

_svg_validity_warned = False


def _strip_svg_wrapper(text: str) -> str:
    text = strip_thinking(text)
    text = re.sub(r"^```(?:xml|svg|html)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    text = re.sub(r"<\?xml[^>]*\?>\s*", "", text)
    return text.strip()


_ANSWER_RE = [
    re.compile(r"Answer:\s*([A-D])", re.IGNORECASE),
    re.compile(r"(?:The\s+)?answer(?:\s+is)?(?:\s*[:=]?\s*)([A-D])", re.IGNORECASE),
    re.compile(
        r"(?:I|my)?\s*(?:choose|select|pick)(?:\s+option)?\s*(?:is|:)?\s*([A-D])",
        re.IGNORECASE,
    ),
    re.compile(
        r"option\s*([A-D])(?:\s+is\s+(?:correct|right|the\s+answer))", re.IGNORECASE,
    ),
    re.compile(r"^([A-D])$"),
]


def _parse_answer_letter(raw: str) -> str:
    text = strip_thinking(raw.strip())
    for pat in _ANSWER_RE:
        matches = pat.findall(text)
        if matches:
            return matches[-1].upper()
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line and re.match(r"^[A-D]$", line):
            return line.upper()
    return ""


# ---------------------------------------------------------------------------
# Shared metric helpers
# ---------------------------------------------------------------------------


def _repair_accuracy(pred: str, gt: str) -> float:
    return 1.0 if normalized_edit_distance(pred, gt) < 0.05 else 0.0


def _repair_similarity(pred: str, gt: str) -> float:
    return 1.0 - normalized_edit_distance(pred, gt)


def _compression_ratio(original: str, optimized: str) -> float:
    if not original:
        return 1.0
    return len(optimized.encode("utf-8")) / len(original.encode("utf-8"))


# -- SVG rendering (optional deps) -----------------------------------------


def _render_svg(svg_code: str, size: int = 256) -> Any:
    try:
        import cairosvg
        from PIL import Image

        png = cairosvg.svg2png(
            bytestring=svg_code.encode("utf-8"), output_width=size, output_height=size,
        )
        img = Image.open(io.BytesIO(png)).convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.paste(img, mask=img)
        return bg.convert("RGB")
    except Exception:
        return None


def _pixel_mse(pred_svg: str, gt_svg: str, size: int = 256) -> float:
    try:
        import numpy as np

        pred_img = _render_svg(pred_svg, size)
        gt_img = _render_svg(gt_svg, size)
        if pred_img is None or gt_img is None:
            return 1.0
        return float(np.mean((np.array(pred_img, dtype=np.float64) - np.array(gt_img, dtype=np.float64)) ** 2)) / 65025.0
    except ImportError:
        return 0.0


def _pixel_ssim(pred_svg: str, gt_svg: str, size: int = 256) -> float:
    try:
        import numpy as np
        from skimage.metrics import structural_similarity

        pred_img = _render_svg(pred_svg, size)
        gt_img = _render_svg(gt_svg, size)
        if pred_img is None or gt_img is None:
            return 0.0
        return float(structural_similarity(
            np.array(pred_img), np.array(gt_img), channel_axis=2, data_range=255,
        ))
    except ImportError:
        return 0.0


def _pixel_lpips(pred_svg: str, gt_svg: str, size: int = 256) -> float:
    try:
        import lpips as lpips_mod
        import numpy as np
        import torch

        pred_img = _render_svg(pred_svg, size)
        gt_img = _render_svg(gt_svg, size)
        if pred_img is None or gt_img is None:
            return 0.0
        fn = lpips_mod.LPIPS(net="alex", verbose=False)

        def _to_tensor(img: Any) -> Any:
            arr = np.array(img, dtype=np.float32) / 255.0
            return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) * 2 - 1

        return float(fn(_to_tensor(pred_img), _to_tensor(gt_img)).item())
    except ImportError:
        return 0.0


def _clip_text_image_score(text: str, svg_code: str, size: int = 256) -> float:
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        img = _render_svg(svg_code, size)
        if img is None or not text:
            return 0.0
        clip_id = "openai/clip-vit-base-patch32"
        proc = CLIPProcessor.from_pretrained(clip_id)
        model = CLIPModel.from_pretrained(clip_id)
        inputs = proc(text=[text], images=[img], return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        score = outputs.logits_per_text.item() / 100.0
        return max(0.0, min(1.0, score))
    except ImportError:
        return 0.0


def _svg_validity(svg_code: str) -> float:
    """Check whether *svg_code* can be rasterised by ``cairosvg``.

    Returns ``1.0`` (valid), ``0.0`` (invalid or missing ``<svg>`` tag).
    Logs a one-time warning when ``cairosvg`` or the system ``libcairo``
    library is unavailable so the caller knows the metric is skipped.
    """
    global _svg_validity_warned  # noqa: PLW0603
    if "<svg" not in svg_code:
        return 0.0
    try:
        import cairosvg

        cairosvg.svg2png(bytestring=svg_code.encode("utf-8"), output_width=64, output_height=64)
        return 1.0
    except ImportError:
        if not _svg_validity_warned:
            logger.warning(
                "svg_validity: cairosvg is not installed — install "
                "lica-bench[metrics] and the libcairo system library."
            )
            _svg_validity_warned = True
        return 0.0
    except OSError:
        if not _svg_validity_warned:
            logger.warning(
                "svg_validity: cairosvg found but the libcairo system "
                "library is missing (e.g. apt-get install libcairo2-dev)."
            )
            _svg_validity_warned = True
        return 0.0
    except Exception:
        return 0.0


def _svg_weighted_complexity(svg_code: str) -> float:
    path_count = len(re.findall(r"<path[\s>]", svg_code, re.IGNORECASE))
    d_attrs = re.findall(r'\bd="([^"]*)"', svg_code)
    d_length = sum(len(d) for d in d_attrs)
    colors = set(re.findall(r"#[0-9a-fA-F]{3,8}", svg_code))
    colors.update(re.findall(r"rgba?\([^)]+\)", svg_code))
    tags = set(re.findall(r"<(\w+)[\s/>]", svg_code))
    has_transform = 1.0 if "transform" in svg_code else 0.0
    has_gradient = 1.0 if re.search(r"<(linearGradient|radialGradient)", svg_code) else 0.0
    has_clippath = 1.0 if "clipPath" in svg_code else 0.0
    byte_size = len(svg_code.encode("utf-8"))
    return (
        path_count * 2.0
        + d_length * 0.01
        + len(colors) * 1.5
        + len(tags) * 1.0
        + has_transform * 5.0
        + has_gradient * 8.0
        + has_clippath * 8.0
        + byte_size * 0.001
    )


# ===========================================================================
# Understanding tasks — svg-1, svg-2
# ===========================================================================


class _SVGQABase(BaseBenchmark):
    """Shared logic for SVG multiple-choice Q/A tasks."""

    QA_FILTER: str = ""

    def load_data(self, data_dir, *, n=None, dataset_root: Union[str, Path]):
        data, data_root = load_task_json(data_dir, self.meta.id)
        samples: list = []
        idx = 0
        for entry in data:
            svg_path = entry.get("svg_path", "")
            image_path = entry.get("image_path", "")
            if svg_path and not Path(svg_path).is_absolute():
                svg_path = str(data_root / svg_path)
            if image_path and not Path(image_path).is_absolute():
                image_path = str(data_root / image_path)
            svg_code = ""
            if svg_path and Path(svg_path).is_file():
                svg_code = Path(svg_path).read_text(encoding="utf-8")
            for q_key, q_data in entry.get("questions", {}).items():
                answer = q_data.get("answer")
                if answer is None:
                    continue
                if self.QA_FILTER and self.QA_FILTER not in q_key:
                    continue
                samples.append({
                    "sample_id": f"svg_qa_{idx:04d}",
                    "ground_truth": str(answer).strip().upper(),
                    "svg_code": svg_code,
                    "image_path": image_path,
                    "question": q_data.get("question", ""),
                    "options": q_data.get("option", {}),
                })
                idx += 1
                if n is not None and len(samples) >= n:
                    return samples
        return samples

    def build_model_input(self, sample, *, modality=None):
        from design_benchmarks.models.base import ModelInput

        opts = "; ".join(f"{k}) {v}" for k, v in sample["options"].items())
        qblock = f"Question: {sample['question']}\nOptions: {opts}"
        if sample.get("svg_code"):
            text = f"{self.PROMPT}\n\nSVG Code:\n{sample['svg_code']}\n\n{qblock}"
        else:
            text = f"{self.PROMPT}\n\n{qblock}"
        images: list = []
        ip = sample.get("image_path", "")
        if ip and Path(ip).is_file():
            images.append(ip)
        return ModelInput(text=text, images=images)

    def parse_model_output(self, output):
        return _parse_answer_letter(output.text)

    def evaluate(self, predictions, ground_truth):
        if not ground_truth:
            return {"accuracy": 0.0}
        correct = sum(1 for p, g in zip(predictions, ground_truth) if str(p) == str(g))
        return {"accuracy": correct / len(ground_truth)}


@benchmark
class SVGPerceptualQA(_SVGQABase):
    """svg-1 — Perceptual Q/A about visual properties of SVG graphics."""

    pipeline_implemented = True
    QA_FILTER = "perceptual"

    PROMPT = (
        "You are an SVG analysis expert.  Analyze the given SVG code and "
        "answer the multiple-choice question about its visual properties.\n"
        "Output your answer in the exact format 'Answer: X' where X is "
        "one of A, B, C, or D.  Do not include any other text."
    )

    meta = BenchmarkMeta(
        id="svg-1",
        name="SVG Perceptual Q/A",
        task_type=TaskType.UNDERSTANDING,
        domain="svg",
        description="Perceptual Q/A about visual properties of SVG graphics",
        input_spec="SVG code + question with multiple-choice options (+ optional rendered image)",
        output_spec="Answer letter (A, B, C, or D)",
        metrics=["accuracy"],
    )


@benchmark
class SVGSemanticQA(_SVGQABase):
    """svg-2 — Semantic Q/A about meaning and purpose of SVG graphics."""

    pipeline_implemented = True
    QA_FILTER = "semantic"

    PROMPT = (
        "You are an SVG analysis expert.  Analyze the given SVG code and "
        "answer the multiple-choice question about what it depicts or represents.\n"
        "Output your answer in the exact format 'Answer: X' where X is "
        "one of A, B, C, or D.  Do not include any other text."
    )

    meta = BenchmarkMeta(
        id="svg-2",
        name="SVG Semantic Q/A",
        task_type=TaskType.UNDERSTANDING,
        domain="svg",
        description="Semantic Q/A about meaning and purpose of SVG graphics",
        input_spec="SVG code + question with multiple-choice options (+ optional rendered image)",
        output_spec="Answer letter (A, B, C, or D)",
        metrics=["accuracy"],
    )


# ===========================================================================
# Editing tasks — svg-3, svg-4, svg-5
# ===========================================================================


@benchmark
class SVGBugFixing(BaseBenchmark):
    """svg-3 — Fix bugs in SVG code so it renders correctly."""

    pipeline_implemented = True

    PROMPT = (
        "You are an SVG code repair assistant. "
        "Given a buggy SVG, output ONLY the corrected SVG code. "
        "Do not include any explanation, markdown fences, or extra text."
    )

    meta = BenchmarkMeta(
        id="svg-3",
        name="SVG Bug Fixing",
        task_type=TaskType.UNDERSTANDING,
        domain="svg",
        description="Fix bugs in SVG code so it renders correctly",
        input_spec="Buggy SVG code",
        output_spec="Fixed SVG code",
        metrics=["repair_accuracy", "repair_similarity", "edit_distance"],
    )

    def load_data(self, data_dir, *, n=None, dataset_root: Union[str, Path]):
        data, _root = load_task_json(data_dir, self.meta.id)
        samples = [
            {
                "sample_id": f"bugfix_{i:04d}",
                "ground_truth": item.get("ground_truth", ""),
                "bug_svg": item.get("bug_svg", ""),
                "error_type": item.get("error_type", ""),
                "difficulty": item.get("difficulty", ""),
            }
            for i, item in enumerate(data)
        ]
        return samples[:n] if n is not None else samples

    def build_model_input(self, sample, *, modality=None):
        from design_benchmarks.models.base import ModelInput

        return ModelInput(
            text=f"{self.PROMPT}\n\nFix this SVG:\n\n{sample['bug_svg']}",
            images=[],
        )

    def parse_model_output(self, output):
        return _strip_svg_wrapper(output.text)

    def evaluate(self, predictions, ground_truth):
        n = max(len(predictions), 1)
        return {
            "repair_accuracy": sum(_repair_accuracy(p, g) for p, g in zip(predictions, ground_truth)) / n,
            "repair_similarity": sum(_repair_similarity(p, g) for p, g in zip(predictions, ground_truth)) / n,
            "edit_distance": sum(normalized_edit_distance(p, g) for p, g in zip(predictions, ground_truth)) / n,
        }


@benchmark
class SVGCodeOptimization(BaseBenchmark):
    """svg-4 — Optimize SVG code to reduce file size."""

    pipeline_implemented = True

    PROMPT = (
        "You are an SVG code optimizer. "
        "Given an SVG, output ONLY the optimized SVG code that is smaller "
        "but renders identically. "
        "Do not include any explanation, markdown fences, or extra text."
    )

    meta = BenchmarkMeta(
        id="svg-4",
        name="SVG Code Optimization",
        task_type=TaskType.UNDERSTANDING,
        domain="svg",
        description="Optimize SVG code to reduce file size",
        input_spec="Original SVG code (+ optional target ratio)",
        output_spec="Optimized SVG code",
        metrics=["compression_ratio", "reference_compression_ratio", "mse"],
    )

    def load_data(self, data_dir, *, n=None, dataset_root: Union[str, Path]):
        data, _root = load_task_json(data_dir, self.meta.id)
        samples = [
            {
                "sample_id": f"optim_{i:04d}",
                "ground_truth": {
                    "optimized_svg": item.get("opti_svg", item.get("origin_svg", "")),
                    "origin_svg": item.get("origin_svg", ""),
                },
                "origin_svg": item.get("origin_svg", ""),
                "opti_ratio": item.get("opti_ratio", 1.0),
            }
            for i, item in enumerate(data)
        ]
        return samples[:n] if n is not None else samples

    def build_model_input(self, sample, *, modality=None):
        from design_benchmarks.models.base import ModelInput

        return ModelInput(
            text=(
                f"{self.PROMPT}\n\nTarget compression ratio: {sample['opti_ratio']}"
                f"\n\nOptimize this SVG:\n\n{sample['origin_svg']}"
            ),
            images=[],
        )

    def parse_model_output(self, output):
        return _strip_svg_wrapper(output.text)

    def evaluate(self, predictions, ground_truth):
        n = max(len(predictions), 1)
        cr_sum = rcr_sum = mse_sum = 0.0
        for pred, gt_dict in zip(predictions, ground_truth):
            origin = gt_dict["origin_svg"]
            ref = gt_dict["optimized_svg"]
            cr_sum += _compression_ratio(origin, pred)
            rcr_sum += _compression_ratio(origin, ref)
            mse_sum += _pixel_mse(pred, ref)
        return {
            "compression_ratio": cr_sum / n,
            "reference_compression_ratio": rcr_sum / n,
            "mse": mse_sum / n,
        }


@benchmark
class SVGStyleEditing(BaseBenchmark):
    """svg-5 — Apply style edits to SVG based on a natural-language command."""

    pipeline_implemented = True

    PROMPT = (
        "You are an SVG style editor. "
        "Given an SVG and an edit command, output ONLY the modified SVG code. "
        "Do not include any explanation, markdown fences, or extra text."
    )

    meta = BenchmarkMeta(
        id="svg-5",
        name="SVG Style Editing",
        task_type=TaskType.UNDERSTANDING,
        domain="svg",
        description="Apply style edits to SVG based on a natural-language command",
        input_spec="Original SVG + edit command",
        output_spec="Modified SVG code",
        metrics=["edit_distance", "mse"],
    )

    def load_data(self, data_dir, *, n=None, dataset_root: Union[str, Path]):
        data, _root = load_task_json(data_dir, self.meta.id)
        samples = [
            {
                "sample_id": f"style_{i:04d}",
                "ground_truth": item.get("modified", ""),
                "original_svg": item.get("original", ""),
                "command": item.get("command", ""),
            }
            for i, item in enumerate(data)
        ]
        return samples[:n] if n is not None else samples

    def build_model_input(self, sample, *, modality=None):
        from design_benchmarks.models.base import ModelInput

        return ModelInput(
            text=(
                f"{self.PROMPT}\n\nEdit command: {sample['command']}"
                f"\n\nOriginal SVG:\n\n{sample['original_svg']}"
            ),
            images=[],
        )

    def parse_model_output(self, output):
        return _strip_svg_wrapper(output.text)

    def evaluate(self, predictions, ground_truth):
        n = max(len(predictions), 1)
        return {
            "edit_distance": sum(normalized_edit_distance(p, g) for p, g in zip(predictions, ground_truth)) / n,
            "mse": sum(_pixel_mse(p, g) for p, g in zip(predictions, ground_truth)) / n,
        }


# ===========================================================================
# Generation tasks — svg-6, svg-7, svg-8
# ===========================================================================

_GEN_METRICS = ["mse", "ssim", "lpips", "clip_score", "code_length", "weighted_complexity", "svg_validity"]


def _evaluate_svg_generation(
    predictions: List[str], ground_truth: List[Dict[str, str]],
) -> Dict[str, float]:
    n = max(len(predictions), 1)
    mse_s = ssim_s = lpips_s = clip_s = cl_s = wc_s = val_s = 0.0
    for pred, gt_dict in zip(predictions, ground_truth):
        target = gt_dict.get("target_svg", "")
        desc = gt_dict.get("description", "")
        mse_s += _pixel_mse(pred, target)
        ssim_s += _pixel_ssim(pred, target)
        lpips_s += _pixel_lpips(pred, target)
        clip_s += _clip_text_image_score(desc, pred)
        cl_s += len(pred.encode("utf-8"))
        wc_s += _svg_weighted_complexity(pred)
        val_s += _svg_validity(pred)
    return {
        "mse": mse_s / n,
        "ssim": ssim_s / n,
        "lpips": lpips_s / n,
        "clip_score": clip_s / n,
        "code_length": cl_s / n,
        "weighted_complexity": wc_s / n,
        "svg_validity": val_s / n,
    }


@benchmark
class TextToSVGGeneration(BaseBenchmark):
    """svg-6 — Generate SVG code from a natural-language description."""

    pipeline_implemented = True

    PROMPT = (
        "You are an SVG code generator. "
        "Given a description of a graphic, output ONLY valid SVG code. "
        "Do not include any explanation, markdown fences, or extra text."
    )

    meta = BenchmarkMeta(
        id="svg-6",
        name="Text-to-SVG Generation",
        task_type=TaskType.GENERATION,
        domain="svg",
        description="Generate SVG code from a natural-language description",
        input_spec="Natural-language description of the target graphic",
        output_spec="SVG code",
        metrics=_GEN_METRICS,
    )

    def load_data(self, data_dir, *, n=None, dataset_root: Union[str, Path]):
        data, data_root = load_task_json(data_dir, self.meta.id)
        samples = [
            {
                "sample_id": f"svg_gen_{i:03d}",
                "ground_truth": {
                    "target_svg": item["answer"],
                    "description": item["question"][0] if item.get("question") else "",
                },
                "description": item["question"][0] if item.get("question") else "",
                "image_path": (
                    str(data_root / item["image"])
                    if item.get("image") and not Path(item["image"]).is_absolute()
                    else item.get("image", "")
                ),
            }
            for i, item in enumerate(data)
        ]
        return samples[:n] if n is not None else samples

    def build_model_input(self, sample, *, modality=None):
        from design_benchmarks.models.base import ModelInput

        return ModelInput(
            text=f"{self.PROMPT}\n\nDescription: {sample['description']}",
            images=[],
        )

    def parse_model_output(self, output):
        return _strip_svg_wrapper(output.text)

    def evaluate(self, predictions, ground_truth):
        return _evaluate_svg_generation(predictions, ground_truth)


@benchmark
class ImageToSVGGeneration(BaseBenchmark):
    """svg-7 — Generate SVG code that reproduces a given image."""

    pipeline_implemented = True

    PROMPT = (
        "You are an SVG code generator. "
        "Given an image, output ONLY valid SVG code that reproduces this graphic. "
        "Do not include any explanation, markdown fences, or extra text."
    )

    meta = BenchmarkMeta(
        id="svg-7",
        name="Image-to-SVG Generation",
        task_type=TaskType.GENERATION,
        domain="svg",
        description="Generate SVG code that reproduces a given image",
        input_spec="Rendered image of target graphic",
        output_spec="SVG code",
        metrics=_GEN_METRICS,
    )

    def load_data(self, data_dir, *, n=None, dataset_root: Union[str, Path]):
        data, data_root = load_task_json(data_dir, self.meta.id)
        samples = [
            {
                "sample_id": f"svg_gen_{i:03d}",
                "ground_truth": {
                    "target_svg": item["answer"],
                    "description": "",
                },
                "image_path": (
                    str(data_root / item["image"])
                    if item.get("image") and not Path(item["image"]).is_absolute()
                    else item.get("image", "")
                ),
            }
            for i, item in enumerate(data)
        ]
        return samples[:n] if n is not None else samples

    def build_model_input(self, sample, *, modality=None):
        from design_benchmarks.models.base import ModelInput

        images: list = []
        ip = sample.get("image_path", "")
        if ip and Path(ip).exists():
            images.append(ip)
        return ModelInput(text=self.PROMPT, images=images)

    def parse_model_output(self, output):
        return _strip_svg_wrapper(output.text)

    def evaluate(self, predictions, ground_truth):
        return _evaluate_svg_generation(predictions, ground_truth)


@benchmark
class ImageTextToSVGGeneration(BaseBenchmark):
    """svg-8 — Generate SVG code from an image and its description."""

    pipeline_implemented = True

    PROMPT = (
        "You are an SVG code generator. "
        "Given an image and its description, output ONLY valid SVG code "
        "that reproduces this graphic. "
        "Do not include any explanation, markdown fences, or extra text."
    )

    meta = BenchmarkMeta(
        id="svg-8",
        name="Image-Text-to-SVG Generation",
        task_type=TaskType.GENERATION,
        domain="svg",
        description="Generate SVG code from an image and its description",
        input_spec="Rendered image + natural-language description of target graphic",
        output_spec="SVG code",
        metrics=_GEN_METRICS,
    )

    def load_data(self, data_dir, *, n=None, dataset_root: Union[str, Path]):
        data, data_root = load_task_json(data_dir, self.meta.id)
        samples = [
            {
                "sample_id": f"svg_gen_{i:03d}",
                "ground_truth": {
                    "target_svg": item["answer"],
                    "description": item["question"][0] if item.get("question") else "",
                },
                "description": item["question"][0] if item.get("question") else "",
                "image_path": (
                    str(data_root / item["image"])
                    if item.get("image") and not Path(item["image"]).is_absolute()
                    else item.get("image", "")
                ),
            }
            for i, item in enumerate(data)
        ]
        return samples[:n] if n is not None else samples

    def build_model_input(self, sample, *, modality=None):
        from design_benchmarks.models.base import ModelInput

        images: list = []
        ip = sample.get("image_path", "")
        if ip and Path(ip).exists():
            images.append(ip)
        return ModelInput(
            text=f"{self.PROMPT}\n\nDescription: {sample['description']}",
            images=images,
        )

    def parse_model_output(self, output):
        return _strip_svg_wrapper(output.text)

    def evaluate(self, predictions, ground_truth):
        return _evaluate_svg_generation(predictions, ground_truth)
