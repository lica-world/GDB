"""Shared data loading helpers used across benchmark task modules."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


def load_task_json(
    data_dir: Union[str, Path],
    task_id: str,
    *,
    alt_filenames: Sequence[str] = (),
) -> Tuple[Any, Path]:
    """Load ``{task_id}.json`` from *data_dir* and return ``(data, resolved_root)``.

    If the primary filename is not found, each name in *alt_filenames* is
    tried in order before raising ``FileNotFoundError``.
    """
    root = Path(data_dir).resolve()
    candidates = [f"{task_id}.json", *alt_filenames]
    for filename in candidates:
        json_path = root / filename
        if json_path.is_file():
            with open(json_path, encoding="utf-8") as f:
                return json.load(f), root
    tried = ", ".join(candidates)
    raise FileNotFoundError(f"None of [{tried}] found in {root}")


def parse_expected_value(raw: str) -> Any:
    """Parse a ground-truth string: try JSON → int → fallback to raw string."""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        return int(raw)
    except ValueError:
        pass
    return raw


def load_csv_samples(
    data_dir: Union[str, Path],
    *,
    n: Optional[int] = None,
    dataset_root: Union[str, Path],
) -> List[Dict[str, Any]]:
    """Read ``samples.csv`` → list of dicts with ``sample_id``, ``ground_truth``, ``image_path``, ``prompt``.

    Expected CSV columns: ``sample_id``, ``prompt``, ``image_path``, ``expected_output``.

    ``image_path`` values in the CSV are resolved against *dataset_root*.
    """
    root = Path(data_dir).resolve()
    csv_path = root / "samples.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"samples.csv not found in {root}")

    base = Path(dataset_root).resolve()

    samples: List[Dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gt = parse_expected_value(row["expected_output"])
            samples.append({
                "sample_id": row["sample_id"],
                "ground_truth": gt,
                "image_path": str((base / row["image_path"]).resolve()),
                "prompt": row["prompt"],
            })
            if n is not None and len(samples) >= n:
                break
    return samples


def build_vision_input(sample: Dict[str, Any], *, modality: Any = None) -> Any:
    """Build a ``ModelInput`` with text prompt and a single image from a CSV sample.

    ``modality`` is accepted for the same signature as ``BaseBenchmark.build_model_input``
    but is currently ignored; callers should branch in the benchmark if they need
    text-only or other modes.
    """
    from design_benchmarks.models.base import ModelInput

    return ModelInput(text=sample["prompt"], images=[sample["image_path"]])
