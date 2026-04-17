#!/usr/bin/env python3
"""Merge cached predictions into a single parity RunReport JSON.

The original parity run produced per-benchmark prediction jsonls but was
killed before the consolidated --output JSON was written. The follow-up
resume run produced its own RunReport for the missing benchmarks. This
script:

  1. Loads the resume run's RunReport (already scored).
  2. For every benchmark not in that report, loads the cached prediction
     jsonl and re-scores it via ``bench.evaluate(predictions, gts)``.
  3. Writes a unified RunReport to outputs/parity_claude_code.json.

Usage:
    .venv/bin/python scripts/merge_parity_results.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gdb import BenchmarkRegistry, BenchmarkResult, RunReport  # noqa: E402

PREDICTIONS_DIR = REPO_ROOT / "outputs/claude-code-media/predictions"
RESUME_REPORT = REPO_ROOT / "outputs/parity_claude_code_resume.json"
OUTPUT = REPO_ROOT / "outputs/parity_claude_code.json"
MODEL_NAME = "claude-sonnet-4-20250514"


def _load_resume_report() -> dict:
    if not RESUME_REPORT.is_file():
        return {"results": {}}
    return json.loads(RESUME_REPORT.read_text())


def _load_predictions(jsonl: Path) -> tuple[list, list, list]:
    sample_ids, preds, gts = [], [], []
    for line in jsonl.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        sample_ids.append(rec.get("sample_id"))
        preds.append(rec.get("prediction", ""))
        gts.append(rec.get("ground_truth"))
    return sample_ids, preds, gts


def _score_one(bench, preds: list, gts: list) -> dict:
    scores = bench.evaluate(preds, gts)
    # Some metrics return numpy types; coerce for JSON serializability.
    out: dict = {}
    for k, v in scores.items():
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            out[k] = v
    return out


def main() -> int:
    registry = BenchmarkRegistry()
    registry.discover()

    resume = _load_resume_report()
    resume_results = resume.get("results", {})

    final = RunReport(
        metadata={
            "benchmarks": [],
            "models": [MODEL_NAME],
            "input_modality": None,
            "source": (
                "Merged: parity_claude_code_resume.json + cached "
                "predictions/*.jsonl from initial run."
            ),
        }
    )

    all_jsonls = sorted(PREDICTIONS_DIR.glob(f"*_{MODEL_NAME}.jsonl"))
    for jsonl in all_jsonls:
        bid = jsonl.name.removesuffix(f"_{MODEL_NAME}.jsonl")

        if bid in resume_results:
            entry = resume_results[bid][MODEL_NAME]
            scores = entry["scores"]
            count = entry.get("count", 0)
            success = entry.get("success_count", count)
            fail = entry.get("failure_count", 0)
            print(f"[resume] {bid}: {len(scores)} metrics  (n={count})")
        else:
            try:
                bench = registry.get(bid)
            except KeyError:
                print(f"[skip ] {bid}: unknown benchmark", file=sys.stderr)
                continue
            sids, preds, gts = _load_predictions(jsonl)
            try:
                scores = _score_one(bench, preds, gts)
            except Exception as exc:
                print(f"[fail ] {bid}: rescoring crashed: {exc}", file=sys.stderr)
                continue
            count = len(sids)
            fail = sum(1 for p in preds if not p)
            success = count - fail
            print(f"[merge] {bid}: {len(scores)} metrics  (n={count}, fail={fail})")

        final.results[bid] = {
            MODEL_NAME: BenchmarkResult(
                benchmark_id=bid,
                model=MODEL_NAME,
                scores=scores,
                count=count,
                success_count=success,
                failure_count=fail,
            )
        }
        final.metadata["benchmarks"].append(bid)

    final.save(str(OUTPUT))
    print(f"\nMerged {len(final.results)} benchmarks -> {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
