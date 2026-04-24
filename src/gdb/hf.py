"""Load benchmark samples from the HuggingFace Hub dataset (lica-world/GDB).

Used when the runner has no ``--dataset-root``. Images are cached to disk
so ``build_model_input()`` receives file paths, matching the local contract.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

HF_REPO_ID = "lica-world/GDB"
_DEFAULT_CACHE = Path.home() / ".cache" / "gdb" / "images"


def _parse_ground_truth(raw: str) -> Any:
    """Reverse the JSON serialization done during upload."""
    if not raw:
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    try:
        return int(raw)
    except (ValueError, TypeError):
        pass
    try:
        return float(raw)
    except (ValueError, TypeError):
        pass
    return raw


def _save_image(pil_img: Any, dest: Path) -> str:
    """Save a PIL image to *dest* and return the absolute path string."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fmt = "PNG"
    if dest.suffix.lower() in (".jpg", ".jpeg"):
        fmt = "JPEG"
        pil_img = pil_img.convert("RGB")
    pil_img.save(dest, format=fmt)
    return str(dest)


_UNSAFE_CHARS = str.maketrans({c: "_" for c in '/\\:*?"<>|'})


def _image_cache_path(cache_dir: Path, benchmark_id: str, sample_id: str) -> Path:
    safe_sid = sample_id.translate(_UNSAFE_CHARS)
    if len(safe_sid) > 120:
        safe_sid = hashlib.md5(sample_id.encode()).hexdigest()
    return cache_dir / benchmark_id / f"{safe_sid}.png"


def load_from_hub(
    benchmark_id: str,
    *,
    n: Optional[int] = None,
    repo_id: str = HF_REPO_ID,
    cache_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Load samples for *benchmark_id* from the HuggingFace Hub.

    Matches ``BaseBenchmark.load_data()``: returns dicts with ``sample_id``,
    ``ground_truth``, plus any fields unpacked from the ``metadata`` column.
    Images are cached under *cache_dir* (default ``~/.cache/gdb/images/``)
    and surfaced as ``image_path`` strings.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required to load from HuggingFace. "
            'Install it with: pip install "lica-gdb[hub]"'
        )

    if cache_dir is None:
        cache_dir = _DEFAULT_CACHE

    logger.info("Loading %s from HuggingFace (%s)...", benchmark_id, repo_id)
    ds = load_dataset(repo_id, benchmark_id, split="train", streaming=True)

    samples: List[Dict[str, Any]] = []
    for row in ds:
        sample: Dict[str, Any] = {
            "sample_id": row["sample_id"],
            "ground_truth": _parse_ground_truth(row["ground_truth"]),
            "prompt": row.get("prompt", ""),
        }

        sample["question"] = sample["prompt"]
        sample["description"] = sample["prompt"]

        meta_raw = row.get("metadata", "{}")
        try:
            extra = json.loads(meta_raw) if meta_raw else {}
        except (json.JSONDecodeError, TypeError):
            extra = {}

        for k, v in extra.items():
            if k not in sample:
                sample[k] = v

        # Generation-only configs type ``image`` as Value("string") and store "",
        # so we check for a PIL-like object rather than truthiness.
        pil_img = row.get("image")
        has_pil = pil_img is not None and hasattr(pil_img, "save")
        if has_pil:
            dest = _image_cache_path(cache_dir, benchmark_id, sample["sample_id"])
            if dest.exists():
                img_path = str(dest)
            else:
                img_path = _save_image(pil_img, dest)
            sample["image_path"] = img_path
        else:
            sample["image_path"] = ""

        samples.append(sample)
        if n is not None and len(samples) >= n:
            break

    logger.info("Loaded %d samples for %s from HuggingFace", len(samples), benchmark_id)

    bad = sum(1 for s in samples if not s.get("sample_id") or not s.get("prompt"))
    if bad > len(samples) * 0.5:
        logger.warning(
            "%s: %d/%d samples have empty sample_id or prompt. "
            "This benchmark may need local data (--dataset-root) for full fidelity.",
            benchmark_id, bad, len(samples),
        )

    return samples
