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
    --provider hf --model-id Qwen/Qwen2.5-Coder-0.5B-Instruct --device auto \
    --dataset-root data/lica-benchmarks-dataset

# Local vLLM (GPU, with sampling params)
python scripts/run_benchmarks.py --benchmarks svg-6 \
    --provider vllm --model-id Qwen/Qwen3-8B --top-k 20 --top-p 0.8 \
    --dataset-root data/lica-benchmarks-dataset

# JSON config file (CLI args override config values)
python scripts/run_benchmarks.py --benchmarks svg-6 --config my_run_config.json \
    --dataset-root data/lica-benchmarks-dataset
```

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
| HuggingFace | (torch) | `--provider hf --device auto` | Local, CPU/GPU |
| vLLM | `.[vllm]` | `--provider vllm` | Local GPU, native batching |
| Diffusion | `.[vllm-omni]` | `--provider diffusion` | Local GPU |
| OpenAI Image | `.[openai]` | `--provider openai_image` | Image generation/editing |

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
