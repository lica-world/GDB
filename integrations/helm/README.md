# lica-gdb-helm

HELM integration for [GDB (GraphicDesignBench)](https://github.com/lica-world/GDB) — run all 39 GDB benchmarks through Stanford CRFM's [HELM](https://github.com/stanford-crfm/helm) framework.

## Install

```bash
pip install lica-gdb-helm
```

This installs `lica-gdb` and `crfm-helm` as dependencies. For benchmarks with heavier metrics:

```bash
pip install "lica-gdb-helm[svg]"      # SVG rendering metrics (LPIPS, SSIM, CLIP)
pip install "lica-gdb-helm[layout]"   # Layout generation metrics (NIMA, HPS, FID)
pip install "lica-gdb-helm[full]"     # Everything
```

## Usage

```bash
# Run a single benchmark
helm-run --run-entries gdb:benchmark_id=category-1,model=openai/gpt-4o \
         --suite gdb-eval --max-eval-instances 50

# Run multiple benchmarks
helm-run --run-entries gdb:benchmark_id=svg-1,model=openai/gpt-4o \
                       gdb:benchmark_id=svg-2,model=openai/gpt-4o \
         --suite gdb-eval

# Summarize and view results
helm-summarize --suite gdb-eval
helm-server --suite gdb-eval
```

## Available benchmarks

All 39 GDB benchmarks are available. Pass any benchmark ID:

| Domain | Benchmark IDs |
|--------|--------------|
| Category | `category-1`, `category-2` |
| Layout | `layout-1` through `layout-8` |
| SVG | `svg-1` through `svg-8` |
| Template | `template-1` through `template-5` |
| Temporal | `temporal-1` through `temporal-6` |
| Typography | `typography-1` through `typography-8` |
| Lottie | `lottie-1`, `lottie-2` |

## Options

| Parameter | Description |
|-----------|-------------|
| `benchmark_id` | Required. GDB benchmark ID (e.g. `svg-1`) |
| `dataset_root` | Optional. Path to local GDB dataset. Defaults to HuggingFace Hub |
| `max_samples` | Optional. Limit number of samples loaded from GDB |

## How it works

This package is a thin adapter (~300 lines) that translates between HELM and GDB types. All benchmark logic — data loading, prompt construction, output parsing, and metric computation — is delegated to the `lica-gdb` package. Metrics from HELM runs are identical to standalone GDB evaluation.

## Development

```bash
cd integrations/helm
pip install -e "../../[hub]"   # install lica-gdb in editable mode
pip install -e .               # install lica-gdb-helm in editable mode

# Test with HELM's built-in echo model
helm-run --run-entries gdb:benchmark_id=category-1,model=simple/model1 \
         --suite test --max-eval-instances 5
```
