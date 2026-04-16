# Parity: claude-code agent for GDB

This directory holds the tooling needed to run an agentic evaluation of GDB
(`lica-bench`) using Anthropic's Claude Code CLI. It exists so we can compare
the GDB benchmark against the Harbor adapter using the **same agent on both
sides** (Scenario 2 parity in the Harbor adapter program).

## What it adds

- `src/gdb/models/claude_code_agent.py` — a `BaseModel` wrapper that drives the
  `claude` CLI per sample, mirroring how the Harbor adapter invokes claude-code
  inside its per-task Docker container.
- `scripts/run_benchmarks.py --provider claude_code` — first-class CLI support
  (default model: `claude-sonnet-4-20250514`).
- `parity/claude-code-setup.sh` — installs Node 22 + `@anthropic-ai/claude-code`.
- `parity/run_parity.sh` — reference runner matching the Harbor adapter's
  parity subset (N=2 samples per benchmark).

## How it works

For each GDB sample, `ClaudeCodeAgent.predict()`:

1. Creates an isolated temp workdir.
2. Stages any input images under `workspace/inputs/`.
3. Writes a prompt that mirrors the Harbor task's `instruction.md`, including
   the same "Write your answer to `<output_file>`" directive. The output file
   name is looked up per-benchmark from a table kept in sync with the Harbor
   adapter's `OUTPUT_FILES` mapping.
4. Invokes `claude --print --model $MODEL_ID --allowedTools Bash,Read,Write,Edit,LS,Glob,Grep`
   non-interactively in that workdir.
5. Reads the resulting file back and returns it as a `ModelOutput` that flows
   through GDB's existing `BenchmarkRunner` → `parse_model_output` → `evaluate`
   pipeline.

This means GDB's native scoring code — metrics, aggregation, CSV/JSON
reporting — is unchanged; only the "what model did we call" step is different.

## Requirements

- `ANTHROPIC_API_KEY` in the environment.
- `claude` CLI on `PATH` (v2.x; run `./parity/claude-code-setup.sh` to install).
- `pip install -e ".[metrics,svg-metrics]"` at the repo root.
- The `gdb-dataset` downloaded locally (`python scripts/download_data.py`).

## Running parity

```bash
export ANTHROPIC_API_KEY=...

# Install the CLI if needed
./parity/claude-code-setup.sh

# Full parity subset (39 benchmarks × 2 samples)
./parity/run_parity.sh

# A single benchmark
./parity/run_parity.sh svg-1

# Override sample count
N=5 ./parity/run_parity.sh layout-1
```

The reference runner saves a report JSON to `outputs/parity_claude_code.json`
that the Harbor adapter's `parity_experiment.json` consumes under
`original_runs`.

## Direct `run_benchmarks.py` usage

You can also use the normal harness CLI without the wrapper:

```bash
python scripts/run_benchmarks.py \
  --provider claude_code \
  --model-id claude-sonnet-4-20250514 \
  --benchmarks svg-1 layout-1 temporal-4 \
  --dataset-root data/gdb-dataset \
  --n 2 \
  --output outputs/claude_code_sample.json \
  --save-images --images-dir outputs/claude-code-media
```

Extra flags specific to this provider:

| Flag                     | Default                                    |
|--------------------------|--------------------------------------------|
| `--claude-timeout`       | `1800` seconds per sample                  |
| `--claude-allowed-tools` | `Bash,Read,Write,Edit,LS,Glob,Grep`        |

## Output files produced by the agent

`ClaudeCodeAgent` returns a `ModelOutput` typed to match the benchmark:

| Benchmark family               | Expected file in workdir | `ModelOutput`                    |
|--------------------------------|--------------------------|----------------------------------|
| SVG semantics (`svg-1/2`, etc.) | `answer.txt`/`answer.svg`/`answer.json` | `text=<file contents>` |
| Layout / typography planning   | `answer.json`            | `text=<json>`                    |
| Image generation (`layout-1`, …) | `output.png`           | `images=[<path>]`                |
| Video generation (`temporal-4/5/6`) | `output.mp4`        | `images=[<path>]`                |

Media outputs are copied to the `--images-dir` (default
`outputs/claude-code/` under the repo root) so the tempdir can be cleaned up.

## Why not just use `--provider anthropic`?

`--provider anthropic` makes a single API call with a single turn — fast and
cheap, but not comparable to an agent that can read/write files, execute
commands, and iterate. The Harbor adapter runs claude-code end-to-end in a
Docker container and lets the agent iterate; this provider reproduces that
same loop on the GDB side so the comparison is apples-to-apples.
