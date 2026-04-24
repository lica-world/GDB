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

SKIP_BENCHMARKS: set = set()


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


def _normalize_paths(value: Any, dataset_root_str: str) -> Any:
    """Replace absolute paths under ``dataset_root`` with the relative tail.

    Benchmarks load locally with absolute paths like
    ``/home/.../gdb-dataset/benchmarks/svg/assets/foo.png``; those strings
    are useless to an HF consumer who doesn't have that tree on disk. We
    strip the root prefix so the parquet is portable — downstream code can
    still try ``Path(x).is_file()`` and gracefully fall back when missing.
    """
    if isinstance(value, str):
        if value.startswith(dataset_root_str + "/"):
            return value[len(dataset_root_str) + 1:]
        return value
    if isinstance(value, list):
        return [_normalize_paths(v, dataset_root_str) for v in value]
    if isinstance(value, dict):
        return {k: _normalize_paths(v, dataset_root_str) for k, v in value.items()}
    return value


def load_via_registry(
    registry,
    benchmark_id: str,
    meta: Any,
    data_dir: Path,
    dataset_root: Path,
) -> List[Dict[str, Any]]:
    bench = registry.get(benchmark_id)
    logger.info("  %s: calling bench.load_data()…", benchmark_id)
    t0 = time.time()
    samples = bench.load_data(data_dir, dataset_root=str(dataset_root))
    logger.info("  %s: load_data produced %d samples in %.1fs",
                benchmark_id, len(samples), time.time() - t0)

    dataset_root_str = str(Path(dataset_root).resolve())
    rows_out = []
    # Only keys whose values are faithfully preserved elsewhere in the parquet
    # row are excluded from ``metadata``. The ``image`` column packs at most
    # ONE PIL blob, so path-valued keys (``video_path``, ``input_image``,
    # ``shuffled_keyframe_paths`` etc.) MUST survive in metadata — otherwise
    # ``build_model_input`` crashes with KeyError on the HF side.
    metadata_skip = {"sample_id", "ground_truth", "prompt"}
    for i, sample in enumerate(samples):
        if i and i % 500 == 0:
            logger.info("  %s: packed %d/%d rows", benchmark_id, i, len(samples))
        img_path = _find_image(sample)
        is_vid = _is_video(img_path) if img_path else False
        has_image = bool(img_path and not is_vid
                         and Path(img_path).exists() and _is_image_file(img_path))

        extra = {k: _normalize_paths(v, dataset_root_str)
                 for k, v in sample.items() if k not in metadata_skip}

        media_path_rel = _normalize_paths(img_path, dataset_root_str) if img_path else ""

        rows_out.append({
            "sample_id": str(sample.get("sample_id", "")),
            "benchmark_id": benchmark_id,
            "domain": meta.domain,
            "task_type": meta.task_type.value,
            "benchmark_name": meta.name,
            "prompt": sample.get("prompt", ""),
            "ground_truth": _serialize(sample.get("ground_truth", "")),
            "image": img_path if has_image else None,
            "media_path": media_path_rel,
            "media_type": "video" if is_vid else ("image" if has_image else "none"),
            "metadata": json.dumps(extra, ensure_ascii=False, default=str) if extra else "{}",
        })

    return rows_out


def load_benchmark(
    registry,
    benchmark_id: str,
    dataset_root: Path,
) -> List[Dict[str, Any]]:
    """Always load via the benchmark's own ``load_data()``.

    The historical ``load_csv_benchmark`` / ``load_json_benchmark`` /
    ``load_manifest_benchmark`` shortcuts were faster but drifted from the
    sample shape each benchmark's ``build_model_input()`` expects at runtime
    (e.g. svg-1 was uploaded with nested ``questions`` structures that the
    pipeline could not consume, and several benchmarks were missing the
    per-entry sample expansion their ``load_data()`` does). Going through
    ``load_via_registry`` guarantees HF parquet rows round-trip to the same
    shape local-file loading produces.
    """
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

    t0 = time.time()
    try:
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


_BENCHMARK_ID_PATTERN = r"^[a-z]+-\d+$"


def _merge_card_configs(api, repo_id: str, new_configs: List[str]) -> List[str]:
    """Union new configs with any already-declared configs on the Hub.

    Avoids the footgun where pushing only a subset of benchmarks and then
    regenerating the card would delete declarations for the rest.
    """
    import re
    from huggingface_hub import hf_hub_download

    try:
        existing_readme = hf_hub_download(
            repo_id=repo_id, repo_type="dataset", filename="README.md",
        )
    except Exception:
        return sorted(set(new_configs))

    content = Path(existing_readme).read_text(encoding="utf-8")
    existing = set()
    for match in re.finditer(r"- config_name:\s*([^\s]+)", content):
        name = match.group(1).strip()
        if re.match(_BENCHMARK_ID_PATTERN, name):
            existing.add(name)
    return sorted(existing | set(new_configs))


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

39 benchmarks for evaluating vision-language models on graphic design tasks — layout, typography, SVG, template matching, animation. Built on 1,148 real design layouts from the [Lica dataset](https://lica.world).

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

    # Merge configs into the existing card rather than replacing, so that a
    # partial re-upload (e.g. --benchmarks template-4 template-5) doesn't wipe
    # out declarations for the other 37 configs that are still on the Hub.
    logger.info("Uploading dataset card...")
    card_config_names = _merge_card_configs(api, args.repo_id, sorted(per_benchmark.keys()))
    api.upload_file(
        path_or_fileobj=generate_dataset_card(card_config_names).encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Update dataset card",
    )

    logger.info("Done! https://huggingface.co/datasets/%s", args.repo_id)


if __name__ == "__main__":
    main()
