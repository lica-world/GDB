#!/usr/bin/env python3
"""Upload the GDB benchmark dataset to HuggingFace Hub.

Usage::

    python scripts/upload_to_hf.py                              # push all benchmarks
    python scripts/upload_to_hf.py --dry-run                    # load data but don't push
    python scripts/upload_to_hf.py --benchmarks svg-1 svg-2     # push specific benchmarks
    python scripts/upload_to_hf.py --dataset-root /path/to/data # custom data path

Requires: pip install datasets huggingface_hub Pillow
Login first: huggingface-cli login
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "gdb-dataset"
HF_REPO_ID = "lica-world/GDB"

SKIP_BENCHMARKS = set()

# Benchmarks whose load_data() is too slow (image compositing, alpha checks)
# and should be loaded directly from their manifest CSVs instead.
MANIFEST_BENCHMARKS = {
    "layout-1": {
        "csv": "layout2_manifest.csv",
        "prompt_key": "prompt",
        "gt_key": "source_layout",
        "image_key": "reference_image",
    },
    "layout-2": {
        "csv": "layout_single_manifest.csv",
        "prompt_key": "prompt",
        "gt_key": "ground_truth_image",
        "image_key": "input_composite",
    },
    "layout-3": {
        "csv": "g4_firestore_image_gen_pairs_manifest.filtered_component_renders.csv",
        "prompt_key": None,
        "gt_key": None,
        "image_key": "a_image_path",
    },
    "layout-8": {
        "csv": "g15_object_insertion_manifest.csv",
        "prompt_key": "prompt",
        "gt_key": "ground_truth_image",
        "image_key": "masked_layout",
    },
    "typography-7": {
        "csv": "g10_text_element_manifest.csv",
        "prompt_key": "prompt",
        "gt_key": "ground_truth_image",
        "image_key": "input_image",
    },
    "typography-8": {
        "csv": "g10_text_inpaint_manifest.csv",
        "prompt_key": "prompt",
        "gt_key": "ground_truth_image",
        "image_key": "input_image",
    },
}


def _serialize(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def _is_image_file(path_str: str) -> bool:
    suffix = Path(path_str).suffix.lower()
    return suffix in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")


def _find_image(sample: Dict[str, Any]) -> Optional[str]:
    for key in ("image_path", "input_image", "input_composite",
                "source_image", "reference_image"):
        val = sample.get(key)
        if val and isinstance(val, str) and Path(val).exists():
            return val
    return None


def _is_video(path: str) -> bool:
    return path.lower().endswith(".mp4")


def _read_csv(csv_path: Path) -> List[Dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_json(json_path: Path) -> Any:
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def load_csv_benchmark(
    benchmark_id: str,
    meta: Any,
    data_dir: Path,
    dataset_root: Path,
) -> List[Dict[str, Any]]:
    csv_path = data_dir / "samples.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"samples.csv not found in {data_dir}")

    rows_out = []
    base = dataset_root.resolve()

    for row in _read_csv(csv_path):
        img_rel = row.get("image_path", "")
        img_abs = str((base / img_rel).resolve()) if img_rel else ""
        is_vid = _is_video(img_abs) if img_abs else False

        has_image = bool(img_abs and not is_vid
                         and Path(img_abs).exists() and _is_image_file(img_abs))

        extra = {k: v for k, v in row.items()
                 if k not in ("sample_id", "prompt", "image_path", "expected_output")}

        rows_out.append({
            "sample_id": row.get("sample_id", ""),
            "benchmark_id": benchmark_id,
            "domain": meta.domain,
            "task_type": meta.task_type.value,
            "benchmark_name": meta.name,
            "prompt": row.get("prompt", ""),
            "ground_truth": row.get("expected_output", ""),
            "image": img_abs if has_image else None,
            "media_path": img_rel,
            "media_type": "video" if is_vid else ("image" if has_image else "none"),
            "metadata": json.dumps(extra, ensure_ascii=False) if extra else "{}",
        })

    return rows_out


JSON_FIELD_MAP = {
    "svg-1": {"gt_key": "answer", "extra": ["svg_code", "question", "options"]},
    "svg-2": {"gt_key": "answer", "extra": ["svg_code", "question", "options"]},
    "svg-3": {"gt_key": "fixed_svg", "extra": ["bug_svg", "error_type", "difficulty"]},
    "svg-4": {"gt_key": None, "extra": ["origin_svg", "opti_ratio"]},
    "svg-5": {"gt_key": "answer", "extra": ["original_svg", "command"]},
    "svg-6": {"gt_key": None, "extra": ["description"]},
    "svg-7": {"gt_key": None, "extra": ["description"]},
    "svg-8": {"gt_key": None, "extra": ["description"]},
    "lottie-1": {"gt_key": None, "extra": ["description"]},
    "lottie-2": {"gt_key": None, "extra": ["description"]},
    "template-1": {"gt_key": "label", "extra": []},
    "template-2": {"gt_key": None, "extra": []},
    "template-3": {"gt_key": None, "extra": ["n_clusters"]},
    "template-4": {"gt_key": None, "extra": []},
    "template-5": {"gt_key": None, "extra": ["difficulty"]},
}


def load_json_benchmark(
    benchmark_id: str,
    meta: Any,
    data_dir: Path,
    dataset_root: Path,
) -> List[Dict[str, Any]]:
    json_path = data_dir / f"{benchmark_id}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"{benchmark_id}.json not found in {data_dir}")

    data = _read_json(json_path)
    base = dataset_root.resolve()
    rows_out = []

    if isinstance(data, list):
        items = data
    else:
        for key in ("samples", "items", "pairs", "queries", "problems"):
            if key in data:
                items = data[key]
                break
        else:
            items = [data]

    for item in items:
        sid = str(item.get("id", item.get("sample_id", "")))

        gt = item.get("answer", item.get("ground_truth", item.get("label", "")))
        if isinstance(gt, (dict, list)):
            gt = json.dumps(gt, ensure_ascii=False)
        else:
            gt = str(gt) if gt is not None else ""

        prompt = item.get("question", item.get("prompt", item.get("description", "")))
        if isinstance(prompt, (dict, list)):
            prompt = json.dumps(prompt, ensure_ascii=False)

        img_rel = item.get("image_path", item.get("image", ""))
        img_abs = ""
        if img_rel and isinstance(img_rel, str):
            candidate = base / img_rel
            if candidate.exists():
                img_abs = str(candidate.resolve())
            else:
                candidate2 = data_dir / img_rel
                if candidate2.exists():
                    img_abs = str(candidate2.resolve())

        is_vid = _is_video(img_abs) if img_abs else False
        has_image = bool(img_abs and not is_vid
                         and Path(img_abs).exists() and _is_image_file(img_abs))

        skip_keys = {"id", "sample_id", "answer", "ground_truth", "label",
                     "question", "prompt", "description", "image_path", "image"}
        extra = {k: v for k, v in item.items() if k not in skip_keys}

        rows_out.append({
            "sample_id": sid,
            "benchmark_id": benchmark_id,
            "domain": meta.domain,
            "task_type": meta.task_type.value,
            "benchmark_name": meta.name,
            "prompt": str(prompt) if prompt else "",
            "ground_truth": gt,
            "image": img_abs if has_image else None,
            "media_path": str(img_rel) if img_rel else "",
            "media_type": "video" if is_vid else ("image" if has_image else "none"),
            "metadata": json.dumps(extra, ensure_ascii=False, default=str) if extra else "{}",
        })

    return rows_out


def load_manifest_benchmark(
    benchmark_id: str,
    meta: Any,
    data_dir: Path,
    dataset_root: Path,
) -> List[Dict[str, Any]]:
    spec = MANIFEST_BENCHMARKS[benchmark_id]
    csv_path = data_dir / spec["csv"]
    if not csv_path.exists():
        raise FileNotFoundError(f"{spec['csv']} not found in {data_dir}")

    base = dataset_root.resolve()
    prompt_key = spec["prompt_key"]
    gt_key = spec["gt_key"]
    image_key = spec["image_key"]
    skip_keys = {"sample_id", prompt_key, gt_key, image_key} - {None}

    rows_out = []
    for row in _read_csv(csv_path):
        sid = row.get("sample_id", row.get("pair_id", ""))
        prompt = row.get(prompt_key, "") if prompt_key else ""
        gt_raw = row.get(gt_key, "") if gt_key else ""

        img_rel = row.get(image_key, "") if image_key else ""
        img_abs = ""
        if img_rel:
            for candidate_base in [data_dir, base]:
                candidate = candidate_base / img_rel
                if candidate.exists():
                    img_abs = str(candidate.resolve())
                    break

        is_vid = _is_video(img_abs) if img_abs else False
        has_image = bool(img_abs and not is_vid
                         and Path(img_abs).exists() and _is_image_file(img_abs))

        extra = {k: v for k, v in row.items() if k not in skip_keys}

        rows_out.append({
            "sample_id": sid,
            "benchmark_id": benchmark_id,
            "domain": meta.domain,
            "task_type": meta.task_type.value,
            "benchmark_name": meta.name,
            "prompt": prompt,
            "ground_truth": gt_raw,
            "image": img_abs if has_image else None,
            "media_path": img_rel,
            "media_type": "video" if is_vid else ("image" if has_image else "none"),
            "metadata": json.dumps(extra, ensure_ascii=False, default=str) if extra else "{}",
        })

    return rows_out


def load_via_registry(
    registry,
    benchmark_id: str,
    meta: Any,
    data_dir: Path,
    dataset_root: Path,
) -> List[Dict[str, Any]]:
    bench = registry.get(benchmark_id)
    samples = bench.load_data(data_dir, dataset_root=str(dataset_root))

    rows_out = []
    for sample in samples:
        img_path = _find_image(sample)
        is_vid = _is_video(img_path) if img_path else False
        has_image = bool(img_path and not is_vid
                         and Path(img_path).exists() and _is_image_file(img_path))

        skip = {"sample_id", "ground_truth", "prompt", "image_path",
                "input_image", "input_composite", "source_image", "video_path"}
        extra = {k: v for k, v in sample.items() if k not in skip}

        rows_out.append({
            "sample_id": str(sample.get("sample_id", "")),
            "benchmark_id": benchmark_id,
            "domain": meta.domain,
            "task_type": meta.task_type.value,
            "benchmark_name": meta.name,
            "prompt": sample.get("prompt", ""),
            "ground_truth": _serialize(sample.get("ground_truth", "")),
            "image": img_path if has_image else None,
            "media_path": img_path or "",
            "media_type": "video" if is_vid else ("image" if has_image else "none"),
            "metadata": json.dumps(extra, ensure_ascii=False, default=str) if extra else "{}",
        })

    return rows_out


def load_benchmark(
    registry,
    benchmark_id: str,
    dataset_root: Path,
) -> List[Dict[str, Any]]:
    if benchmark_id in SKIP_BENCHMARKS:
        logger.info("Skipping %s (excluded)", benchmark_id)
        return []

    bench = registry.get(benchmark_id)
    meta = bench.meta

    try:
        data_dir = bench.resolve_data_dir(dataset_root)
    except FileNotFoundError as exc:
        logger.warning("Skipping %s: %s", benchmark_id, exc)
        return []

    csv_path = data_dir / "samples.csv"
    json_path = data_dir / f"{benchmark_id}.json"

    t0 = time.time()
    try:
        if benchmark_id in MANIFEST_BENCHMARKS:
            rows = load_manifest_benchmark(benchmark_id, meta, data_dir, dataset_root)
        elif csv_path.exists():
            rows = load_csv_benchmark(benchmark_id, meta, data_dir, dataset_root)
        elif json_path.exists():
            rows = load_json_benchmark(benchmark_id, meta, data_dir, dataset_root)
        else:
            rows = load_via_registry(registry, benchmark_id, meta, data_dir, dataset_root)
    except Exception as exc:
        logger.warning("Failed to load %s: %s: %s", benchmark_id, type(exc).__name__, exc)
        return []

    dt = time.time() - t0
    logger.info("Loaded %s: %d samples (%.1fs)", benchmark_id, len(rows), dt)
    return rows


def build_dataset(all_rows: List[Dict[str, Any]]):
    import datasets

    has_images = any(r["image"] is not None for r in all_rows)

    features = datasets.Features({
        "sample_id": datasets.Value("string"),
        "benchmark_id": datasets.Value("string"),
        "domain": datasets.Value("string"),
        "task_type": datasets.Value("string"),
        "benchmark_name": datasets.Value("string"),
        "prompt": datasets.Value("large_string"),
        "ground_truth": datasets.Value("large_string"),
        "image": datasets.Image() if has_images else datasets.Value("string"),
        "media_path": datasets.Value("string"),
        "media_type": datasets.Value("string"),
        "metadata": datasets.Value("large_string"),
    })

    for r in all_rows:
        if has_images:
            if not r["image"]:
                r["image"] = None
        else:
            r["image"] = ""

    return datasets.Dataset.from_list(all_rows, features=features)


def generate_dataset_card(config_names: Optional[List[str]] = None) -> str:
    if config_names is None:
        config_names = ["all"]

    configs_yaml = "\n".join(
        f'  - config_name: {c}\n    data_files:\n      - split: train\n        path: "{c}/train-*"'
        for c in config_names
    )

    header = f"""\
---
license: apache-2.0
task_categories:
  - visual-question-answering
  - image-to-text
  - text-to-image
language:
  - en
tags:
  - benchmark
  - design
  - multimodal
  - graphic-design
  - svg
  - typography
  - layout
  - animation
  - lottie
pretty_name: "GDB: GraphicDesignBench"
size_categories:
  - 1K<n<10K
configs:
{configs_yaml}
---"""

    return header + """

# GDB: GraphicDesignBench

40 benchmarks for evaluating vision-language models on graphic design tasks — layout, typography, SVG, template matching, animation. Built on 1,148 real design layouts from the [Lica dataset](https://lica.world).

**Paper:** [arXiv:2604.04192](https://arxiv.org/abs/2604.04192) &nbsp;|&nbsp; **Code:** [github.com/lica-world/GDB](https://github.com/lica-world/GDB) &nbsp;|&nbsp; **Blog:** [lica.world](https://lica.world/blog/gdb-real-world-benchmark-for-graphic-design)

## Usage

```python
from datasets import load_dataset

ds = load_dataset("lica-world/GDB", "svg-1")
```

## Schema

| Field | Type | Description |
|-------|------|-------------|
| `sample_id` | string | Sample identifier |
| `benchmark_id` | string | e.g. `svg-1`, `typography-3` |
| `domain` | string | layout, typography, svg, template, temporal, category, lottie |
| `task_type` | string | `understanding` or `generation` |
| `prompt` | string | Evaluation prompt |
| `ground_truth` | string | Expected answer (JSON for complex types) |
| `image` | Image | Input image (when applicable) |
| `metadata` | string | Task-specific fields as JSON |

## Evaluation

```bash
pip install git+https://github.com/lica-world/GDB.git
```

```python
from gdb.registry import BenchmarkRegistry

registry = BenchmarkRegistry()
registry.discover()
bench = registry.get("svg-1")
scores = bench.evaluate(predictions, ground_truth)
```

## Citation

```bibtex
@article{gdb2026,
  title={GDB: A Real-World Benchmark for Graphic Design},
  author={Deganutti, Adrienne and Hirsch, Elad and Zhu, Haonan and Seol, Jaejung and Mehta, Purvanshi},
  journal={arXiv preprint arXiv:2604.04192},
  year={2026}
}
```

Apache 2.0
"""


def main():
    parser = argparse.ArgumentParser(description="Upload GDB dataset to HuggingFace Hub")
    parser.add_argument(
        "--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT,
        help=f"Path to gdb-dataset/ (default: {DEFAULT_DATASET_ROOT})",
    )
    parser.add_argument(
        "--repo-id", default=HF_REPO_ID,
        help=f"HuggingFace repo ID (default: {HF_REPO_ID})",
    )
    parser.add_argument(
        "--benchmarks", nargs="*", default=None,
        help="Specific benchmark IDs to upload (default: all)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Load data but don't push to HuggingFace",
    )
    parser.add_argument(
        "--private", action="store_true",
        help="Create the dataset as private",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    if not (dataset_root / "lica-data" / "metadata.csv").exists():
        logger.error(
            "Dataset not found at %s. Run: python scripts/download_data.py", dataset_root,
        )
        sys.exit(1)

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from gdb.registry import BenchmarkRegistry

    logger.info("Discovering benchmarks...")
    registry = BenchmarkRegistry()
    registry.discover()

    benchmark_ids = args.benchmarks or sorted(registry.list_ids())
    logger.info("Will process %d benchmarks", len(benchmark_ids))

    all_rows: List[Dict[str, Any]] = []
    per_benchmark: Dict[str, List[Dict[str, Any]]] = {}

    for bid in benchmark_ids:
        rows = load_benchmark(registry, bid, dataset_root)
        if rows:
            all_rows.extend(rows)
            per_benchmark[bid] = rows

    logger.info(
        "Total: %d samples across %d benchmarks",
        len(all_rows), len(per_benchmark),
    )

    if not all_rows:
        logger.error("No samples loaded.")
        sys.exit(1)

    if args.dry_run:
        logger.info("[DRY RUN] Would push %d samples to %s", len(all_rows), args.repo_id)
        for bid, rows in sorted(per_benchmark.items()):
            m = registry.get(bid).meta
            logger.info("  %-15s  %-12s %-14s  %4d samples", bid, m.domain, m.task_type.value, len(rows))
        return

    from huggingface_hub import HfApi

    api = HfApi()
    try:
        api.repo_info(repo_id=args.repo_id, repo_type="dataset")
        logger.info("Repository %s exists", args.repo_id)
    except Exception:
        logger.info("Creating repository %s ...", args.repo_id)
        api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=args.private)

    for bid, rows in sorted(per_benchmark.items()):
        logger.info("Pushing config '%s' (%d samples)...", bid, len(rows))
        ds = build_dataset(rows)
        ds.push_to_hub(args.repo_id, config_name=bid,
                       commit_message=f"Upload GDB benchmark: {bid}")

    logger.info("Uploading dataset card...")
    api.upload_file(
        path_or_fileobj=generate_dataset_card(sorted(per_benchmark.keys())).encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Add dataset card",
    )

    logger.info("Done! https://huggingface.co/datasets/%s", args.repo_id)


if __name__ == "__main__":
    main()
