#!/usr/bin/env bash
# Chain two upstream (lica-bench) parity runs, sharding each run across
# SHARDS parallel processes on disjoint benchmark subsets. Outputs are
# merged into a single parity_claude_code_runN.json after both shards finish.
#
# Waits for gdb-dataset to be available before starting.
set -eo pipefail

cd "$(dirname "$0")/.."
# Export every assignment in ~/.env so child processes (subshells for shards)
# inherit ANTHROPIC_API_KEY etc.
set -a
# shellcheck disable=SC1090
source ~/.env
set +a

SHARDS="${SHARDS:-2}"

# Ensure libcairo is reachable for _pixel_mse (svg-6/7/8) on macOS.
if [ -z "$DYLD_FALLBACK_LIBRARY_PATH" ]; then
  export DYLD_FALLBACK_LIBRARY_PATH="/opt/homebrew/lib:/usr/local/lib"
fi

wait_for_dataset() {
  local max_wait=$((60*60*2))
  local waited=0
  while [ $waited -lt $max_wait ]; do
    if [ -d data/gdb-dataset/benchmarks ] && [ -d data/gdb-dataset/lica-data ]; then
      # Dataset layout: benchmarks/<category>/<BenchmarkClass>/, plus
      # benchmarks/svg/svg-N.json manifests for the SVG family. Require the
      # svg manifests as a representative readiness signal since extraction
      # touches them last.
      if [ -f data/gdb-dataset/benchmarks/svg/svg-1.json ]; then
        echo "[dataset] ready at $(date -u +%FT%TZ)"
        return 0
      fi
    fi
    sleep 30
    waited=$((waited+30))
    echo "[dataset] waiting ($waited s elapsed, size=$(du -sh data/gdb-dataset 2>/dev/null | awk '{print $1}'))"
  done
  echo "[dataset] TIMEOUT after ${max_wait}s"
  return 1
}

source .venv/bin/activate

# Enumerate benchmarks (mirrors parity/run_parity.sh auto-discovery).
# Use a while-read loop to stay compatible with bash 3.2 (macOS default).
ALL_BENCHMARKS=()
while IFS= read -r b; do
  [ -n "$b" ] && ALL_BENCHMARKS+=("$b")
done < <(python scripts/run_benchmarks.py --list 2>/dev/null \
  | awk 'NR>2 && $NF=="ready" {print $1}')

if [ "${#ALL_BENCHMARKS[@]}" -eq 0 ]; then
  echo "[fatal] could not enumerate benchmarks"
  exit 2
fi

# Round-robin into shards so heavy benchmarks (nima, ocr) spread across shards.
declare -a SHARD_LIST
for i in $(seq 0 $((SHARDS-1))); do SHARD_LIST[$i]=""; done
idx=0
for b in "${ALL_BENCHMARKS[@]}"; do
  s=$((idx % SHARDS))
  SHARD_LIST[$s]="${SHARD_LIST[$s]} $b"
  idx=$((idx+1))
done

run_shard() {
  # run_shard <run_label> <shard_idx> <benchmarks...>
  #
  # Runs each benchmark as its own python invocation so per-benchmark JSONs
  # land incrementally under outputs/parity_claude_code_${label}_shard${sidx}/.
  # This makes the chain shutdown-safe: if the laptop is closed mid-shard,
  # any benchmark that already wrote its JSON is kept on re-launch.
  local label="$1"; shift
  local sidx="$1"; shift
  local out="outputs/parity_claude_code_${label}_shard${sidx}.json"
  local log="outputs/parity_claude_code_${label}_shard${sidx}.log"
  local bench_dir="outputs/parity_claude_code_${label}_shard${sidx}_benchmarks"
  if [ -f "$out" ]; then
    echo "[skip] shard $sidx of $label already complete ($out)"
    return 0
  fi
  mkdir -p "$bench_dir"
  echo "[start] $label shard $sidx ($# benchmarks) @ $(date -u +%FT%TZ)" | tee -a "$log"
  for bench in "$@"; do
    local bout="$bench_dir/${bench}.json"
    if [ -f "$bout" ]; then
      echo "[skip-bench] $label shard $sidx: $bench already done" | tee -a "$log"
      continue
    fi
    echo "[bench] $label shard $sidx: $bench @ $(date -u +%FT%TZ)" | tee -a "$log"
    if OUTPUT="$bout" bash parity/run_parity.sh "$bench" >> "$log" 2>&1; then
      echo "[bench-done] $label shard $sidx: $bench" | tee -a "$log"
    else
      echo "[bench-fail] $label shard $sidx: $bench (continuing)" | tee -a "$log"
    fi
  done
  # Merge the per-benchmark JSONs into the shard JSON for downstream aggregation.
  python - "$bench_dir" "$out" <<'PY'
import json, sys
from pathlib import Path
bench_dir = Path(sys.argv[1])
out = Path(sys.argv[2])
merged = {"metadata": None, "results": {}}
benches = []
for p in sorted(bench_dir.glob("*.json")):
    doc = json.loads(p.read_text())
    if merged["metadata"] is None:
        merged["metadata"] = doc.get("metadata", {})
    benches.extend(doc.get("metadata", {}).get("benchmarks", []))
    for k, v in doc.get("results", {}).items():
        merged["results"][k] = v
if merged["metadata"] is None:
    merged["metadata"] = {}
merged["metadata"]["benchmarks"] = sorted(set(benches))
merged["metadata"]["source"] = f"Per-benchmark shard ({len(benches)} benchmarks from {bench_dir.name})"
out.write_text(json.dumps(merged, indent=2))
print(f"[merge-shard] wrote {out} ({len(merged['results'])} benchmarks)")
PY
  echo "[done]  $label shard $sidx @ $(date -u +%FT%TZ)" | tee -a "$log"
}

merge_shards() {
  # merge_shards <run_label> <final_output>
  local label="$1"
  local final="$2"
  python - <<PY
import json, glob
from pathlib import Path
shards = sorted(Path("outputs").glob(f"parity_claude_code_${label}_shard*.json"))
if not shards:
    raise SystemExit(f"no shards found for ${label}")
merged = {"metadata": None, "results": {}}
all_benchmarks = []
for s in shards:
    doc = json.loads(s.read_text())
    if merged["metadata"] is None:
        merged["metadata"] = doc.get("metadata", {})
    all_benchmarks.extend(doc.get("metadata", {}).get("benchmarks", []))
    for k, v in doc.get("results", {}).items():
        merged["results"][k] = v
merged["metadata"]["benchmarks"] = sorted(set(all_benchmarks))
merged["metadata"]["source"] = f"Merged from {len(shards)} parallel shards: {[s.name for s in shards]}"
Path("$final").write_text(json.dumps(merged, indent=2))
print(f"Merged {len(shards)} shards -> $final ({len(merged['results'])} benchmarks)")
PY
}

run_one() {
  local label="$1"
  local final="outputs/parity_claude_code_${label}.json"
  if [ -f "$final" ]; then
    echo "[skip] $final already exists"
    return 0
  fi
  local pids=()
  for i in $(seq 0 $((SHARDS-1))); do
    # shellcheck disable=SC2086
    run_shard "$label" "$i" ${SHARD_LIST[$i]} &
    pids+=($!)
  done
  local failed=0
  for p in "${pids[@]}"; do
    wait "$p" || failed=$((failed+1))
  done
  if [ "$failed" -gt 0 ]; then
    echo "[warn] $failed shard(s) returned non-zero for $label"
  fi
  merge_shards "$label" "$final"
}

wait_for_dataset
run_one run2
run_one run3
echo "[chain done] $(date -u +%FT%TZ)"
