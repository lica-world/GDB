# Scripts

## Data pipeline

### `download_data.py` — fetch the benchmark bundle

```bash
python scripts/download_data.py                            # → data/lica-benchmarks-dataset/
python scripts/download_data.py --out-dir /tmp             # custom parent directory
python scripts/download_data.py --from-zip ~/release.zip   # local zip
```

The layout is described in the root [README](../README.md#benchmark-dataset-layout): **`lica-data/`** (core Lica tree) and **`benchmarks/<domain>/`** (task inputs).

## `run_benchmarks.py`

Single entry point for all benchmark operations.

### Downloaded data + stub model (no API keys)

Use **`--stub-model`** with **`--dataset-root`** pointing at the Lica bundle root. Task directories under **`benchmarks/`** come from each benchmark's metadata; use **`--data`** for a custom path.

```bash
python scripts/run_benchmarks.py --stub-model --benchmarks layout-3 \
    --dataset-root data/lica-benchmarks-dataset --n 5
```

### Standard run

```bash
# API model
python scripts/run_benchmarks.py --benchmarks svg-1 \
    --provider gemini --dataset-root data/lica-benchmarks-dataset

# Local HuggingFace model
python scripts/run_benchmarks.py --benchmarks svg-6 \
    --provider hf --model-id Qwen/Qwen3-VL-4B-Instruct --device auto \
    --dataset-root data/lica-benchmarks-dataset

# Local vLLM (GPU, with sampling params)
python scripts/run_benchmarks.py --benchmarks svg-6 \
    --provider vllm --model-id Qwen/Qwen3-VL-4B-Instruct --top-k 20 --top-p 0.8 \
    --dataset-root data/lica-benchmarks-dataset

# Diffusion / image generation (defaults to FLUX.2 klein 4B)
python scripts/run_benchmarks.py --benchmarks layout-1 \
    --provider diffusion \
    --dataset-root data/lica-benchmarks-dataset

# User custom python model entrypoint
python scripts/run_benchmarks.py --benchmarks svg-1 \
    --provider custom --custom-entry my_models.wrapper:build_model \
    --custom-init-kwargs '{"checkpoint":"/models/foo","temperature":0.0}' \
    --dataset-root data/lica-benchmarks-dataset

# Image-generation / editing task with a custom wrapper
python scripts/run_benchmarks.py --benchmarks layout-3 \
    --provider custom --custom-entry my_models.image_wrapper:build_model \
    --custom-modality image_generation \
    --dataset-root data/lica-benchmarks-dataset

# Official FLUX.2 wrapper via the existing custom provider
python -m pip install --no-deps --ignore-requires-python \
    "git+https://github.com/black-forest-labs/flux2.git"
python scripts/run_benchmarks.py --benchmarks layout-1 layout-3 layout-8 typography-7 typography-8 \
    --provider custom \
    --custom-entry design_benchmarks.models.local_models:Flux2Model \
    --custom-init-kwargs '{"model_name":"flux.2-klein-4b"}' \
    --custom-modality image_generation \
    --dataset-root data/lica-benchmarks-dataset

# JSON config file (CLI args override config values)
python scripts/run_benchmarks.py --benchmarks svg-6 --config my_run_config.json \
    --dataset-root data/lica-benchmarks-dataset

```

`--custom-entry` must reference an importable module attribute
(`module.path:attr`). If the wrapper is not installed as a package, add its
parent directory to `PYTHONPATH` before running the benchmark.

For `hf` / `vllm`, `--model-id` can be either a hub repo ID or a local
checkpoint directory. If a model path or repo name does not clearly indicate
whether it is text-only or VLM, pass `--model-modality text` or
`--model-modality text_and_image` explicitly.

For image-output tasks, prefer `--custom-modality image_generation`. If your
wrapper consumes source images or masks, expose capability attributes on the
returned object so preflight warnings stay accurate:
`supports_image_output`, `supports_image_input`, `supports_mask_editing`,
`supports_video_output`.

The built-in `design_benchmarks.models.local_models:Flux2Model` wrapper also
uses this `custom` path. FLUX.2 weights and the shared autoencoder are fetched
from Hugging Face and can use either environment tokens (`HF_TOKEN`,
`HF_HUB_TOKEN`) or an existing cached login/token file.

The default local text/VLM model ID is now `Qwen/Qwen3-VL-4B-Instruct` for both
`hf` and `vllm`, and the default `diffusion` model ID is `flux.2-klein-4b`.

### Batch submit/collect (~50% cheaper)

```bash
export DESIGN_BENCHMARKS_GCS_BUCKET=your-bucket
python scripts/run_benchmarks.py --benchmarks svg-1 \
    --provider gemini --dataset-root data/lica-benchmarks-dataset --batch-submit

python scripts/run_benchmarks.py --collect jobs/job_20260316_120000_gemini.json
```

### Output

Per-benchmark CSVs and `tracker.jsonl` are written under `outputs/`.
Run summaries and saved reports include:

- metric scores (`accuracy`, `macro_f1`, etc.)
- `n` (total samples)
- `success_count`
- `failure_count`
- `failure_rate`

```bash
-o results.json    # override report path (format inferred from extension)
-o results.csv
--no-log           # disable per-sample tracker JSONL
```

### Providers

| Provider | Install extra | CLI flag | Notes |
|----------|--------------|----------|-------|
| Gemini | `.[gemini]` | `--provider gemini` | API, supports batch |
| OpenAI | `.[openai]` | `--provider openai` | API, supports batch |
| Anthropic | `.[anthropic]` | `--provider anthropic` | API, supports batch |
| HuggingFace | (torch) | `--provider hf --device auto` | Local, CPU/GPU, text/VLM inference. Use `--model-modality` when needed. |
| vLLM | `.[vllm]` | `--provider vllm` | Local GPU, native batching, text/VLM inference. |
| Diffusion | `.[vllm-omni]` | `--provider diffusion` | Local GPU image generation. `flux.2-*` routes to the built-in FLUX.2 wrapper. |
| OpenAI Image | `.[openai]` | `--provider openai_image` | Image generation/editing |
| Custom Entrypoint | (your code) | `--provider custom --custom-entry module:attr` | Load user model/factory/callable |

### All flags

```bash
python scripts/run_benchmarks.py --help
```

### Evaluation extras

Some tasks need optional dependencies for full metric computation. Install as needed:

```bash
pip install -e ".[metrics]"          # sklearn, cairosvg, Pillow (most tasks)
pip install -e ".[svg-metrics]"      # metrics + torch/transformers/lpips (SVG generation)
pip install -e ".[lottie-metrics]"   # metrics + rlottie-python (Lottie frame eval)
pip install -e ".[layout-metrics]"   # full stack on Linux with Python < 3.12
```

Metrics whose dependencies are unavailable are omitted from the output (with a logged warning).

See the root [README.md](../README.md) for benchmark counts, status, and dataset layout. See [docs/CONTRIBUTING.md](../docs/CONTRIBUTING.md) for adding new tasks.
