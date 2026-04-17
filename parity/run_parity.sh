#!/usr/bin/env bash
# Reference parity run: claude-code on GDB, matching the Harbor adapter's setup.
#
# Prereqs:
#   - ANTHROPIC_API_KEY exported
#   - claude CLI installed (see ./claude-code-setup.sh)
#   - gdb-dataset available at data/gdb-dataset (see scripts/download_data.py)
#   - `pip install -e ".[metrics,svg-metrics]"` from the repo root
#
# Usage:
#   ./parity/run_parity.sh                       # all 39 benchmarks, 2 samples each
#   ./parity/run_parity.sh svg-1 svg-2           # subset
#   N=1 ./parity/run_parity.sh layout-1          # override samples per benchmark
set -euo pipefail

cd "$(dirname "$0")/.."

: "${ANTHROPIC_API_KEY:?set ANTHROPIC_API_KEY before running parity}"

N="${N:-2}"
MODEL="${MODEL:-claude-sonnet-4-20250514}"
DATASET_ROOT="${DATASET_ROOT:-data/gdb-dataset}"
OUTPUT="${OUTPUT:-outputs/parity_claude_code.json}"

if [ "$#" -gt 0 ]; then
  BENCHMARKS=("$@")
else
  mapfile -t BENCHMARKS < <(python scripts/run_benchmarks.py --list 2>/dev/null \
    | awk 'NR>2 && $NF=="ready" {print $1}')
fi

mkdir -p "$(dirname "$OUTPUT")"

echo "Running ${#BENCHMARKS[@]} benchmark(s) × N=${N} with claude-code/${MODEL}"
python scripts/run_benchmarks.py \
  --provider claude_code \
  --model-id "$MODEL" \
  --benchmarks "${BENCHMARKS[@]}" \
  --dataset-root "$DATASET_ROOT" \
  --n "$N" \
  --output "$OUTPUT" \
  --save-images --images-dir outputs/claude-code-media

echo "Saved report: $OUTPUT"
