# Harbor adapter branch

This branch (`harbor-adapter`) extends `lica-bench` with a Claude Code agent so
the GDB benchmark can be evaluated under the same agentic conditions used by
the [Harbor](https://www.harborframework.com/) adapter at
`laude-institute/harbor#1433`.

It exists to satisfy the Harbor adapter program's **Scenario 2 parity
requirement**: when an original benchmark is LLM-driven, parity must compare
the same agent (same CLI, same model, same instruction format) run inside both
harnesses. A single-shot `claude.messages.create` call on the GDB side vs. an
agentic `claude-code` Docker container on the Harbor side is not fair.

## What this branch adds

| Area            | File                                                | Purpose                                                               |
|-----------------|-----------------------------------------------------|-----------------------------------------------------------------------|
| Agent           | `src/gdb/models/claude_code_agent.py`               | `ClaudeCodeAgent(BaseModel)` — drives `claude` CLI per sample          |
| Registry        | `src/gdb/models/registry.py`                        | Registers the new model so `load_model("claude_code")` works           |
| Runner          | `src/gdb/runner.py`                                 | Injects `benchmark_id`/`sample_id` into `ModelInput.metadata`          |
| CLI             | `scripts/run_benchmarks.py`                         | `--provider claude_code`, `--claude-timeout`, `--claude-allowed-tools` |
| Parity tooling  | `parity/claude-code-setup.sh`, `parity/run_parity.sh`, `parity/_run_upstream_parity.sh`, `parity/README.md` | Setup, reference parity runner, and sharded multi-run chain |
| Rescoring       | `scripts/_nima_rescore_runs.py`                     | Post-hoc NIMA rescoring for `layout-8` predictions when `pyiqa` wasn't available at eval time |

See `parity/README.md` for full usage details.

### Image-output benchmark handling

`layout-1`, `layout-8`, `typography-7`, and `typography-8` score PNG
predictions via NIMA, OCR, or layer-aware compositing. `ClaudeCodeAgent`
gives the agent an explicit image-output instruction block for these
benchmarks (Pillow / cairosvg are available) and, if the agent still
emits only an SVG or a text/JSON fallback, `_rasterize_image_output`
converts it to a real PNG with cairosvg before the evaluator runs. This
is scaffolding so the existing evaluators see a valid image; no scoring
logic, ground truth, or thresholds are modified.

### Sharded parity chain

`parity/_run_upstream_parity.sh` chains two full parity runs (run2,
run3), sharding each across `$SHARDS` parallel processes on disjoint
benchmark subsets. Per-benchmark JSONs are written incrementally and
re-merged into the shard output on completion, so a mid-run shutdown
never discards finished benchmarks on re-launch.

### NIMA rescoring

`nima_score` on `layout-8` is only emitted when `pyiqa` is importable at
eval time. When a run completes without it,
`scripts/_nima_rescore_runs.py` runs NIMA against the per-run prediction
PNGs preserved under `outputs/claude-code-media/` and patches the score
into the corresponding `outputs/parity_claude_code_run{N}.json`. The
Harbor verifier image ships `pyiqa` pinned, so the Harbor side never
needs this step; both sides end up with the same metric on all runs.

## Quick start

```bash
git checkout harbor-adapter
pip install -e ".[metrics,svg-metrics]"

export ANTHROPIC_API_KEY=...
./parity/claude-code-setup.sh

python scripts/download_data.py   # if not already present

# Run 2 samples of svg-1 under claude-sonnet-4
python scripts/run_benchmarks.py \
  --provider claude_code \
  --benchmarks svg-1 \
  --dataset-root data/gdb-dataset \
  --n 2 \
  --output outputs/smoke.json
```

## Matching the Harbor side

The Harbor adapter at `harbor/adapters/gdb/` builds per-task `instruction.md`
files that ask the agent to write its answer to a benchmark-specific filename
(e.g. `/workspace/answer.svg` or `/workspace/output.png`). This branch's
`ClaudeCodeAgent` constructs an identical instruction in-process and runs the
same `claude` CLI against it. The only differences between the two runs are:

- **Harbor side**: the agent runs inside the task's Docker container.
- **GDB side**: the agent runs in a local temp workdir on the host.

Both sides end up with the same `answer.<ext>` / `output.<ext>` files, which
are then scored by each framework's native evaluation code. The Harbor
adapter's `parity_experiment.json` consumes the GDB report JSON under
`original_runs`.

## Not merged to `main`

This branch is intentionally kept separate from `main`. The core GDB benchmark
continues to use its standard provider list (`openai`, `anthropic`, `gemini`,
etc.) for non-agentic evaluation. Agent-based evaluation lives here so it can
be adopted selectively without imposing Claude Code CLI as a runtime
dependency on regular users.
