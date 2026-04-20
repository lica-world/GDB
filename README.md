# GDB: GraphicDesignBench

**GDB** evaluates vision-language models on professional graphic design tasks — layout reasoning, typography, SVG editing, template matching, animation. 40 benchmarks across 7 domains, built on the [Lica dataset](https://github.com/lica-world/lica-dataset) (1,148 real design layouts).

**Paper:** [arXiv:2604.04192](https://arxiv.org/abs/2604.04192) &nbsp;|&nbsp; **Dataset:** [HuggingFace](https://huggingface.co/datasets/lica-world/GDB) &nbsp;|&nbsp; **Blog:** [lica.world](https://lica.world/blog/gdb-real-world-benchmark-for-graphic-design)

## Benchmarks

Each task is either **understanding** or **generation**:

| Domain | Tasks | Benchmarks | Description |
|--------|------:|----------:|-------------|
| category | 2 | 2 | Design category classification and user intent prediction |
| layout | 8 | 8 | Spatial reasoning over design canvases (aspect ratio, element counting, component type and detection), layout generation (intent-to-layout, partial completion, aspect-ratio adaptation), and layer-aware object insertion (`layout-8`, reference- or description-guided per sample) |
| lottie | 2 | 2 | Lottie animation generation from text and image |
| svg | 8 | 8 | SVG reasoning and editing (perceptual and semantic Q/A, bug fixing, optimization, style editing) and generation (text-to-SVG, image-to-SVG, combined input) |
| template | 5 | 5 | Template matching, retrieval, clustering, and generation (style completion, color transfer) |
| temporal | 8 | 6 | Keyframe ordering; motion type classification; video/component duration and start-time estimation; generation (animation parameters, motion trajectory, short-form video) |
| typography | 13 | 9 | Font family, color, size/weight/alignment/letter spacing/line height, style ranges, curvature, rotation, and generation (styled text element, styled text rendering to layout, text removal/background inpainting as `image-6`) |

## Setup

### Install

```bash
pip install lica-gdb
```

Or install from source with extras:

```bash
git clone https://github.com/lica-world/GDB.git
cd GDB
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[hub]"              # Load data from HuggingFace (no download step)
pip install -e ".[metrics]"          # scipy, sklearn, Pillow, cairosvg, etc.
pip install -e ".[openai]"           # OpenAI provider
pip install -e ".[gemini]"           # Gemini provider
pip install -e ".[anthropic]"        # Anthropic provider
pip install -e ".[svg-metrics]"      # Full SVG eval (metrics + LPIPS, CLIP)
pip install -e ".[lottie-metrics]"   # Lottie frame-level eval (rlottie-python)
pip install -e ".[layout-metrics]"   # Layout/image metrics (Linux + Python<3.12 recommended)
pip install -e ".[dev]"              # ruff linter
```

### Verify

```bash
python scripts/run_benchmarks.py --list
```

### Data

Without `--dataset-root`, benchmarks are loaded directly from [HuggingFace](https://huggingface.co/datasets/lica-world/GDB) (requires the `.[hub]` extra). No download step needed.

For local data (offline use, full benchmark coverage):

```bash
python scripts/download_data.py                 # → data/gdb-dataset/
```

Then pass `--dataset-root data/gdb-dataset` to benchmark runs.

### Run benchmarks

```bash
# From HuggingFace (no local data needed)
python scripts/run_benchmarks.py --stub-model --benchmarks category-1 --n 5

# From local data
python scripts/run_benchmarks.py --stub-model --benchmarks category-1 \
    --dataset-root data/gdb-dataset --n 5

# Real model
python scripts/run_benchmarks.py --benchmarks svg-1 \
    --provider openai --model-id gpt-5.4 \
    --dataset-root data/gdb-dataset

# Temporal benchmarks (video-based)
python scripts/run_benchmarks.py --benchmarks temporal-1 \
    --provider gemini \
    --dataset-root data/gdb-dataset

# User custom python model entrypoint
python scripts/run_benchmarks.py --benchmarks svg-1 \
    --provider custom --custom-entry my_models.wrapper:build_model \
    --custom-init-kwargs '{"checkpoint":"/models/foo"}' \
    --dataset-root data/gdb-dataset

# Local default VLM/LLM (defaults to Qwen3-VL-4B-Instruct)
python scripts/run_benchmarks.py --benchmarks svg-1 \
    --provider hf --device auto \
    --dataset-root data/gdb-dataset

# Diffusion / image generation (defaults to FLUX.2 klein 9B)
python scripts/run_benchmarks.py --benchmarks layout-1 \
    --provider diffusion \
    --dataset-root data/gdb-dataset

# Image-generation / editing task with a custom wrapper
python scripts/run_benchmarks.py --benchmarks typography-7 \
    --provider custom --custom-entry my_models.image_wrapper:build_model \
    --custom-modality image_generation \
    --dataset-root data/gdb-dataset

# Official FLUX.2 wrapper via the existing custom provider
python -m pip install --no-deps --ignore-requires-python \
    "git+https://github.com/black-forest-labs/flux2.git"
python scripts/run_benchmarks.py --benchmarks layout-1 layout-3 layout-8 typography-7 typography-8 \
    --provider custom \
    --custom-entry gdb.models.local_models:Flux2Model \
    --custom-init-kwargs '{"model_name":"flux.2-klein-9b"}' \
    --custom-modality image_generation \
    --dataset-root data/gdb-dataset
```

`--custom-entry` must point to an importable module attribute (installed or reachable via `PYTHONPATH`). For image-output tasks, use `--custom-modality image_generation`.

See [scripts/README.md](scripts/README.md) for batch submit/collect, vLLM, HuggingFace, custom model entrypoints, multi-model configs, and all CLI flags.

### HELM integration

GDB benchmarks can also be run through Stanford CRFM's [HELM](https://github.com/stanford-crfm/helm) framework:

```bash
pip install lica-gdb-helm

helm-run --run-entries gdb:benchmark_id=category-1,model=openai/gpt-4o \
         --suite gdb-eval --max-eval-instances 50

helm-summarize --suite gdb-eval
helm-server --suite gdb-eval
```

All 40 benchmarks are available. See [integrations/helm/](integrations/helm/) for details.

### API keys

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_API_KEY=...            # Gemini (Google AI Studio / google-genai API key)
```

For **Gemini on Vertex AI** (service account), pass a JSON key file instead of relying on `GOOGLE_API_KEY`:

```bash
python scripts/run_benchmarks.py --benchmarks svg-1 --provider gemini \
    --credentials /path/to/service-account.json \
    --dataset-root data/gdb-dataset
```

The file must be either a **service account** key (`type: service_account`) or JSON containing an `api_key` field.

Batch submit for Gemini also needs a GCS bucket (`--bucket` or `GDB_GCS_BUCKET`); see [scripts/README.md](scripts/README.md).

## Dataset layout

The local data bundle (`python scripts/download_data.py`) unpacks as:

```
gdb-dataset/
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

`--dataset-root` points here. `lica-data/` is the shared Lica corpus; `benchmarks/` holds per-domain evaluation inputs. See `src/gdb/tasks/<domain>.py` or [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for details.

## Project structure

```
GDB/
├── src/gdb/
│   ├── tasks/              # @benchmark classes — one file per domain
│   │   ├── category.py     #   category-1, category-2
│   │   ├── image.py        #   compatibility shim (re-exports image-6)
│   │   ├── layout.py       #   layout-1 … layout-8
│   │   ├── lottie.py       #   lottie-1, lottie-2
│   │   ├── svg.py          #   svg-1 … svg-8
│   │   ├── template.py     #   template-1 … template-5
│   │   ├── temporal.py     #   temporal-1 … temporal-6
│   │   └── typography.py   #   typography-1 … typography-8 + image-6 implementation
│   ├── models/             # Provider wrappers (OpenAI, Anthropic, Gemini, HF, vLLM)
│   ├── metrics/            # Reusable metric functions (IoU, FID, SSIM, LPIPS, edit distance)
│   ├── evaluation/
│   │   ├── tracker.py      # Per-sample JSONL logger
│   │   └── reporting.py    # BenchmarkResult / RunReport (CSV + JSON)
│   ├── inference/          # Batch API runners, GCS helpers
│   ├── utils/              # Shared helpers (image, text, layout path resolution)
│   ├── base.py             # BaseBenchmark, BenchmarkMeta, TaskType, @benchmark
│   ├── hf.py               # Load samples from HuggingFace Hub
│   ├── registry.py         # Auto-discovery via pkgutil.walk_packages
│   └── runner.py           # BenchmarkRunner orchestration
├── scripts/
│   ├── download_data.py    # Fetch + unpack into gdb-dataset/
│   ├── run_benchmarks.py   # Unified CLI for list, stub, real, and batch runs
│   └── upload_to_hf.py     # Upload dataset to HuggingFace Hub
├── integrations/
│   └── helm/               # HELM plugin (lica-gdb-helm on PyPI)
├── docs/
│   └── CONTRIBUTING.md     # How to add tasks and domains
└── pyproject.toml
```

## Python API

```python
from gdb import BenchmarkRegistry, BenchmarkRunner, load_from_hub
from gdb.models import load_model

# Load samples from HuggingFace (no local data needed)
samples = load_from_hub("category-1", n=10)

# Or use the full pipeline with local data
registry = BenchmarkRegistry()
registry.discover()
runner = BenchmarkRunner(registry)
models = {"openai": load_model("openai", model_id="gpt-5.4")}

# Without dataset_root → loads from HuggingFace automatically
report = runner.run(benchmark_ids=["category-1"], models=models, n=5)

# With dataset_root → loads from local files
report = runner.run(
    benchmark_ids=["svg-1"],
    models=models,
    dataset_root="data/gdb-dataset",
    n=5,
)
print(report.summary())
```

## Contributing

See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md).

## Known issues

- Some metrics (LPIPS, CLIP score, SSIM, CIEDE2000) need heavier extras (`.[svg-metrics]`, `.[lottie-metrics]`, `.[layout-metrics]`). Full `.[layout-metrics]` requires Linux + Python < 3.12. Missing metric deps are skipped with a warning.
- `--provider` picks the backend; `--model-id` is the catalog string within that backend. With `--multi-models`, each entry is `provider:model_id`.
- For local models, `--model-id` can be a hub ID or local path. Pass `--model-modality text` or `--model-modality text_and_image` if ambiguous.

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
| Custom Entrypoint | (your code) | `--provider custom --custom-entry module:attr` |

### Eval extras

| Extra | What it adds |
|-------|-------------|
| `.[metrics]` | scipy, sklearn, scikit-image, Pillow, cairosvg |
| `.[svg-metrics]` | + torch, transformers, lpips |
| `.[lottie-metrics]` | + rlottie-python |
| `.[layout-metrics]` | + pyiqa, hpsv2, hpsv3, dreamsim, image-reward (Linux + Python < 3.12) |

## Citation

```bibtex
@article{gdb2026,
  title={GDB: A Real-World Benchmark for Graphic Design},
  author={Deganutti, Adrienne and Hirsch, Elad and Zhu, Haonan and Seol, Jaejung and Mehta, Purvanshi},
  journal={arXiv preprint arXiv:2604.04192},
  year={2026}
}
```

## License

Apache 2.0
