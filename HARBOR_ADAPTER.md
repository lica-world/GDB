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
| Parity tooling  | `parity/claude-code-setup.sh`, `parity/run_parity.sh`, `parity/README.md` | Setup + reference parity runner                          |

See `parity/README.md` for full usage details.

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
