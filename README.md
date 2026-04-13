# lica-bench

**lica-bench** is a structured evaluation suite for measuring how well vision-language models understand, edit, and generate graphic design artifacts. It covers layout reasoning, typography, visual hierarchy, SVG/vector understanding, template variants, animation, and more.

Benchmarks use the [Lica dataset](https://github.com/purvanshi/lica-dataset) (1,148 graphic design layouts). The release zip is unpacked as **`lica-benchmarks-dataset/`** with two parts: **`lica-data/`** holds the **core Lica files** (`metadata.csv`, `layouts/`, `images/`, `annotations/`). **`benchmarks/<domain>/`** holds **task-specific evaluation data** (manifests, JSON specs, prepared assets).

## Benchmarks

Each task is one of two types: **understanding** (answer a question or edit an artifact), or **generation** (produce a new artifact). 45 tasks span seven domains across 39 benchmarks:

| Domain | Tasks | Benchmarks | Description |
|--------|------:|----------:|-------------|
| category | 2 | 2 | Design category classification and user intent prediction |
| layout | 8 | 8 | Spatial reasoning over design canvases (aspect ratio, element counting, component type and detection), layout generation (intent-to-layout, partial completion, aspect-ratio adaptation), and layer-aware object insertion (`layout-8`, reference- or description-guided per sample) |
| lottie | 2 | 2 | Lottie animation generation from text and image |
| svg | 8 | 8 | SVG reasoning and editing (perceptual and semantic Q/A, bug fixing, optimization, style editing) and generation (text-to-SVG, image-to-SVG, combined input) |
| template | 5 | 5 | Template matching, retrieval, clustering, and generation (style completion, color transfer) |
| temporal | 8 | 6 | Keyframe ordering; motion type classification; **video duration**, **component duration**, and **start-time** estimation (`temporal-3`, with motion type / speed / direction in the same benchmark); generation (animation parameters, motion trajectory, short-form video) |
| typography | 12 | 8 | Font family, color, size / weight / alignment / letter spacing / line height* (single benchmark), style ranges, curvature, rotation, and generation (styled text element, styled text rendering to layout) |

> \* `typography-3` (Text Params Estimation) expects one JSON object with five fields: `font_size`, `font_weight`, `text_align`, `letter_spacing`, and `line_height`.

> † **Temporal (8 tasks, 6 benchmarks):** five understanding lines = `temporal-1`, `temporal-2`, and three timing lines from `temporal-3` (clip/video duration, per-component duration, start time—**start time is separate** from both duration lines). Three generation lines = `temporal-4`–`temporal-6`.

## Getting started

### 1. Install

```bash
git clone https://github.com/purvanshi/lica-bench.git
cd lica-bench
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# Add extras you need (pick any combination)
pip install -e ".[metrics]"          # scipy, sklearn, Pillow, cairosvg, etc.
pip install -e ".[openai]"           # OpenAI provider
pip install -e ".[gemini]"           # Gemini provider
pip install -e ".[anthropic]"        # Anthropic provider
pip install -e ".[svg-metrics]"      # Full SVG eval (metrics + LPIPS, CLIP)
pip install -e ".[lottie-metrics]"   # Lottie frame-level eval (rlottie-python)
pip install -e ".[layout-metrics]"   # Layout/image metrics (Linux + Python<3.12 recommended)
pip install -e ".[dev]"              # ruff linter
```

The PyPI/setuptools distribution is **lica-bench**; import the library as **`design_benchmarks`**.

### 2. Verify installation (no data, no API keys)

```bash
python scripts/run_benchmarks.py --list                     # enumerate tasks and readiness
```

### 3. Download data

```bash
python scripts/download_data.py                              # → data/lica-benchmarks-dataset/
```

**`--dataset-root`** is the bundle root (contains `lica-data/` and `benchmarks/`). Task data is read from `benchmarks/` using each benchmark's metadata. Use **`--data`** to point at a specific directory.

### 4. Run benchmarks

```bash
# Stub model (no API keys; validates load_data + build_model_input on real data)
python scripts/run_benchmarks.py --stub-model --benchmarks category-1 \
    --dataset-root data/lica-benchmarks-dataset --n 5

# Real model
python scripts/run_benchmarks.py --benchmarks svg-1 \
    --provider openai --model-id gpt-5.4 \
    --dataset-root data/lica-benchmarks-dataset

# Temporal benchmarks (video-based)
python scripts/run_benchmarks.py --benchmarks temporal-1 \
    --provider gemini \
    --dataset-root data/lica-benchmarks-dataset
```

Use the same **`--dataset-root`** (Lica bundle root) for stub runs, API runs, and **`--batch-submit`** so paths inside CSVs/JSON resolve correctly.

See [scripts/README.md](scripts/README.md) for batch submit/collect, vLLM, HuggingFace, multi-model comparison, config files, and all CLI flags.

### 5. API keys

Set whichever provider(s) you need:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_API_KEY=...            # Gemini (Google AI Studio / google-genai API key)
```

For **Gemini on Vertex AI** (service account), pass a JSON key file instead of relying on `GOOGLE_API_KEY`:

```bash
python scripts/run_benchmarks.py --benchmarks svg-1 --provider gemini \
    --credentials /path/to/service-account.json \
    --dataset-root data/lica-benchmarks-dataset
```

The file must be either a **service account** key (`type: service_account`) or JSON containing an `api_key` field.

**Batch submit** for Gemini also needs a GCS bucket (`--bucket` or `DESIGN_BENCHMARKS_GCS_BUCKET`); see [scripts/README.md](scripts/README.md).

## Benchmark dataset layout

Everything lives under one root directory **`lica-benchmarks-dataset/`** (e.g. `data/lica-benchmarks-dataset/` after `download_data.py`):

```
lica-benchmarks-dataset/
├── lica-data/                    # core Lica release (layouts, renders, metadata)
│   ├── metadata.csv              # one row per layout
│   ├── layouts/<template_id>/<layout_id>.json
│   ├── images/<template_id>/<layout_id>.{png,jpg,webp,mp4}
│   └── annotations/…             # optional
│
└── benchmarks/                   # evaluation inputs per domain
    ├── category/                 #   CategoryClassification/, UserIntentPrediction/
    ├── image/
    ├── layout/
    ├── lottie/
    ├── svg/
    ├── template/
    ├── temporal/                 #   KeyframeOrdering/, MotionTypeClassification/, etc.
    └── typography/
```

**Using this bundle:** Set **`--dataset-root`** to this directory. CSV `image_path` and template `data_root` entries resolve relative to **`--dataset-root`**.

**What the two trees are:** **`lica-data/`** is the shared Lica corpus (layout JSON, renders, `metadata.csv`). **`benchmarks/`** holds evaluation payloads per domain (CSVs, JSON, manifests, copied assets). Exact filenames differ by task; see the module under `src/design_benchmarks/tasks/<domain>.py` or [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) when adding or packaging data.

## Project structure

```
lica-bench/
├── src/design_benchmarks/
│   ├── tasks/              # @benchmark classes — one file per domain
│   │   ├── category.py     #   category-1, category-2
│   │   ├── layout.py       #   layout-1 … layout-8
│   │   ├── lottie.py       #   lottie-1, lottie-2
│   │   ├── svg.py          #   svg-1 … svg-8
│   │   ├── template.py     #   template-1 … template-5
│   │   ├── temporal.py     #   temporal-1 … temporal-6
│   │   └── typography.py   #   typography-1 … typography-8
│   ├── models/             # Provider wrappers (OpenAI, Anthropic, Gemini, HF, vLLM)
│   ├── metrics/            # Reusable metric functions (IoU, FID, SSIM, LPIPS, edit distance)
│   ├── evaluation/
│   │   ├── tracker.py      # Per-sample JSONL logger
│   │   └── reporting.py    # BenchmarkResult / RunReport (CSV + JSON)
│   ├── inference/          # Batch API runners, GCS helpers
│   ├── utils/              # Shared helpers (image, text, layout path resolution)
│   ├── base.py             # BaseBenchmark, BenchmarkMeta, TaskType, @benchmark
│   ├── registry.py         # Auto-discovery via pkgutil.walk_packages
│   └── runner.py           # BenchmarkRunner orchestration
├── scripts/
│   ├── download_data.py    # Fetch + unpack into lica-benchmarks-dataset/
│   └── run_benchmarks.py   # Unified CLI for list, stub, real, and batch runs
├── docs/
│   └── CONTRIBUTING.md     # How to add tasks and domains
└── pyproject.toml
```

## Quick start (Python API)

```python
from pathlib import Path
from design_benchmarks import BenchmarkRegistry, BenchmarkRunner
from design_benchmarks.models import load_model

root = Path("data/lica-benchmarks-dataset")
registry = BenchmarkRegistry()
registry.discover()

runner = BenchmarkRunner(registry)
models = {"openai": load_model("openai", model_id="gpt-5.4")}
report = runner.run(
    benchmark_ids=["svg-1"],
    models=models,
    dataset_root=root,
    n=5,
)
print(report.summary())
report.save("outputs/report.json")
runner.tracker.save("outputs/tracker.jsonl")
```

`RunReport` includes both metric scores and reliability counters per benchmark/model:
`count`, `success_count`, `failure_count`, and `failure_rate`. This makes partial-run
failures visible in terminal summaries and saved JSON/CSV reports.

## Contributing

See **[docs/CONTRIBUTING.md](docs/CONTRIBUTING.md)** for:

- How to add a benchmark task to an existing domain
- How to create a new domain module
- Where benchmark inputs live in the Lica release and the PR checklist

## Limitations

- Some metrics (LPIPS, CLIP score, SSIM, CIEDE2000) need heavier extras (`.[svg-metrics]`, `.[lottie-metrics]`, `.[layout-metrics]`). The full `.[layout-metrics]` stack is enabled on Linux with Python < 3.12. Metrics whose dependencies are unavailable are omitted from the output (with a logged warning).
- **`--provider`** picks which backend runs the model (OpenAI, Gemini, Anthropic, etc.); **`--model-id`** is only the catalog string for *that* backend (it does not select the provider). If you omit **`--model-id`**, the default for the chosen provider is used (see `DEFAULT_MODEL_IDS` in `scripts/run_benchmarks.py`). With **`--multi-models`**, each entry is **`provider:model_id`** so both are explicit. Use a **`--model-id`** your account actually exposes (README examples may name newer IDs such as `gpt-5.4`).

## Models

| Provider | Install extra | CLI flag |
|----------|--------------|----------|
| OpenAI | `.[openai]` | `--provider openai` |
| Anthropic | `.[anthropic]` | `--provider anthropic` |
| Gemini | `.[gemini]` | `--provider gemini` |
| HuggingFace | (torch) | `--provider hf --device auto` |
| vLLM | `.[vllm]` | `--provider vllm` |
| Diffusion | `.[vllm-omni]` | `--provider diffusion` |
| OpenAI Image | `.[openai]` | `--provider openai_image` |

### Evaluation extras

| Extra | Contents | Used by |
|-------|----------|---------|
| `.[metrics]` | scipy, sklearn, scikit-image, Pillow, cairosvg | All implemented tasks (clustering, color, SVG rendering) |
| `.[svg-metrics]` | metrics + torch, transformers, lpips | SVG generation (LPIPS, CLIP score) |
| `.[lottie-metrics]` | metrics + rlottie-python | Lottie generation (frame MSE, frame SSIM) |
| `.[layout-metrics]` | torch, transformers (+ Linux/Python<3.12: pyiqa, hpsv2, hpsv3, dreamsim, image-reward) | Layout / image generation (FID, HPSv2/v3, DreamSim) |

## Dataset

The [Lica dataset](https://github.com/purvanshi/lica-dataset) underpins the initial benchmark release:

- 1,148 graphic design layouts across 9 design categories
- Structured JSON annotations (components, positions, styles, descriptions)
- Rendered images (PNG) and animations (MP4)
- Download: `python scripts/download_data.py`

## Citation

If you use this benchmark, please cite the original LICA dataset:

```bibtex
@misc{lica-dataset,
  author = {Mehta, Purvanshi and others},
  title  = {LICA: Open-Source Graphic Design Layout Dataset},
  year   = {2025},
  url    = {https://github.com/purvanshi/lica-dataset}
}
```
