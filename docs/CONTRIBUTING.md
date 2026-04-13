# Contributing benchmarks

Guide for anyone adding a new benchmark task or a new domain. Read this before writing code.

## How the repo fits together

```
scripts/download_data.py    →  data/lica-benchmarks-dataset/   (lica-data/ + benchmarks/<domain>/)
scripts/run_benchmarks.py   →  load_data → inference → evaluate
```

| File / directory | Role |
|------------------|------|
| `src/design_benchmarks/tasks/*.py` | One file per domain. Contains `@benchmark` classes. |
| `src/design_benchmarks/base.py` | `BaseBenchmark`, `BenchmarkMeta`, `TaskType`, `@benchmark` decorator. |
| `src/design_benchmarks/registry.py` | Auto-discovers all `@benchmark` classes via `pkgutil.walk_packages`. |
| `scripts/run_benchmarks.py` | Runs benchmarks (`--dataset-root` required; `--data` optional). `--list` shows **ready**. |
| `src/design_benchmarks/metrics/` | Reusable metric functions (IoU, FID, SSIM, LPIPS, edit distance, font-name normalisation). |
| `src/design_benchmarks/utils/` | Shared helpers (image array handling, text cleanup, layout path resolution). |

## Benchmark dataset layout (`lica-benchmarks-dataset/`)

The download / release zip unpacks to **`lica-benchmarks-dataset/`** with:

- **`lica-data/`** — core Lica tree (`metadata.csv`, `layouts/`, `images/`, `annotations/`).
- **`benchmarks/<domain>/`** — evaluation inputs for each domain.

Rules:

- **`metadata.csv`** (under `lica-data/`) is the master index where tasks need it.
- Paths under `lica-data/` are **nested by `template_id`** (not flat).
- **`meta.domain`** names the task module and often matches a top-level `benchmarks/<domain>/` folder. Set **`meta.data_subpath`** when inputs live in a subfolder (e.g. `layout/…`, or under `image/…`).
- CSV `image_path` values are **relative to `--dataset-root`** (e.g. `lica-data/images/…`). The framework resolves them automatically.

Full details in the root [README.md](../README.md#benchmark-dataset-layout).

## Adding a benchmark to an existing domain

1. Open `src/design_benchmarks/tasks/<domain>.py` (e.g. `layout.py`). Set **`data_subpath`** on `BenchmarkMeta` to the folder under `benchmarks/` that holds this task's files, or omit it when `benchmarks/<domain>/` is correct.

2. Add a class:

```python
from pathlib import Path
from typing import Union

from design_benchmarks.base import BaseBenchmark, BenchmarkMeta, TaskType, benchmark

@benchmark
class MyNewTask(BaseBenchmark):
    """Short description of what this task evaluates."""

    pipeline_implemented = True

    meta = BenchmarkMeta(
        id="layout-9",                         # <domain>-<n>, unique in registry
        name="My New Layout Task",
        task_type=TaskType.UNDERSTANDING,      # or GENERATION
        domain="layout",                       # usually matches tasks/<file>.py name
        data_subpath="layout/MyTaskFolder",    # under <dataset_root>/benchmarks/
        description="One-line description",
        metrics=["accuracy"],
    )

    def load_data(self, data_dir, *, n=None, dataset_root: Union[str, Path]):
        # Return list of dicts, each with sample_id + ground_truth + task fields
        ...

    def build_model_input(self, sample, *, modality=None):
        from design_benchmarks.models.base import ModelInput
        return ModelInput(text="...", images=[sample["image_path"]])

    def evaluate(self, predictions, ground_truth):
        correct = sum(1 for p, g in zip(predictions, ground_truth) if str(p) == str(g))
        return {"accuracy": correct / len(predictions) if predictions else 0.0}
```

3. Verify:

```bash
pip install -e .
python scripts/run_benchmarks.py --list | grep layout-9
```

### What each method does

| Method | Purpose | Must implement? |
|--------|---------|:-:|
| `load_data(data_dir, *, n=None, dataset_root=…)` | Discover samples on disk, extract ground truth. Return list of dicts with `sample_id`, `ground_truth`, plus any paths/metadata. | Yes |
| `build_model_input(sample, *, modality=None)` | Convert one sample dict → `ModelInput` (text, images, metadata). | Yes |
| `evaluate(predictions, ground_truth)` | Compute metric scores. Return `{"metric_name": float}`. | Yes |
| `parse_model_output(output)` | Extract prediction from raw `ModelOutput`. Default: `output.text.strip()`. | Only if default is wrong |

### Conventions

- **ID format:** `<domain>-<n>` (e.g. `layout-9`, `temporal-6`). **Ids must be unique** in the whole registry. When adding a task, use the next unused `<n>` in that domain (no gaps in the shipped set).
- **`domain` field** should match the `tasks/*.py` filename. **`data_subpath`** (if set) must match the folder layout under `benchmarks/` in the dataset release.
- **All tasks must be fully implemented** — `load_data`, `build_model_input`, and `evaluate` must be real implementations with `pipeline_implemented = True`.
- **Auto-discovery**: the registry uses `pkgutil.walk_packages` on the `design_benchmarks` package. You do **not** edit `registry.py` or import your class anywhere — just decorate with `@benchmark` and it is found.

## Creating a new domain

If your benchmarks don't fit an existing file:

1. Add `src/design_benchmarks/tasks/<new_domain>.py`.
2. Put `@benchmark` classes inside (same pattern as above).
3. Update the domain table in `README.md`.
4. Ship the corresponding `benchmarks/<new_domain>/` layout in the Lica dataset release, and document it in the task module docstrings.

Keep cross-imports between task modules minimal; use lazy imports inside methods to avoid import cycles.

## Metrics and evaluation

- Reuse metrics under `src/design_benchmarks/metrics/` when possible.
- If you add a new metric family, keep it in a small focused module with light imports (optional dependencies stay optional).
- Your `evaluate()` should return metric names that match `BenchmarkMeta.metrics`.

## Optional dependencies

Heavy packages (torch, Pillow, scipy, cairosvg, etc.) go in `pyproject.toml` extras, not core `dependencies`. Document in your class docstring which extra is needed:

```python
class MyTask(BaseBenchmark):
    """my-task -- requires ``pip install -e ".[metrics]"`` for scipy/cairosvg."""
```

Available extras:

| Extra | Contains |
|-------|----------|
| `metrics` | scipy, sklearn, scikit-image, Pillow, cairosvg |
| `svg-metrics` | metrics + torch, transformers, lpips |
| `lottie-metrics` | metrics + rlottie-python |
| `layout-metrics` | torch, transformers, pyiqa, hpsv2, hpsv3, dreamsim, image-reward |

See `pyproject.toml` `[project.optional-dependencies]` for the full list.

## End-to-end walkthrough: adding `layout-9`

`layout-9` is a **placeholder** for the task you are adding; it is not in the repo until you implement it. To exercise steps 5–6 against **shipping** data first, substitute an existing id (for example `layout-4` or `layout-5`) instead of `layout-9`.

```bash
# 1. Edit the task file
#    Add @benchmark class to src/design_benchmarks/tasks/layout.py

# 2. Check registration
pip install -e .
python scripts/run_benchmarks.py --list | grep layout-9

# 3. Implement load_data, build_model_input, evaluate; set pipeline_implemented = True

# 4. Download Lica data (includes benchmarks/layout/)
python scripts/download_data.py

# 5. Stub run (after setting data_subpath for layout-9)
python scripts/run_benchmarks.py --stub-model --benchmarks layout-9 \
    --dataset-root data/lica-benchmarks-dataset --n 5

# Or pass --data explicitly:
#   --data data/lica-benchmarks-dataset/benchmarks/layout/MyTaskFolder

# 6. Real run
python scripts/run_benchmarks.py --benchmarks layout-9 \
    --provider openai --model-id gpt-5.4 \
    --dataset-root data/lica-benchmarks-dataset --n 5
```

## PR checklist

- [ ] `python scripts/run_benchmarks.py --list` shows your new id(s) as **ready**
- [ ] `BenchmarkMeta.id` is unique (registry raises on duplicates)
- [ ] `meta.domain` matches `tasks/<file>.py` name; `data_subpath` matches the dataset layout under `benchmarks/` (or omit if `benchmarks/<domain>/` is enough)
- [ ] `pipeline_implemented = True` and all three pipeline methods are implemented
- [ ] `load_data` raises a clear error if the resolved data directory is wrong
- [ ] Shipped inputs under `benchmarks/<domain>/` are documented (dataset release + task docstrings)
- [ ] No new hard dependencies added to core `dependencies` (use extras)
- [ ] `ruff check` passes on your files (`pip install -e ".[dev]"`)
