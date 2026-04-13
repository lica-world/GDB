"""Per-sample evaluation tracker for debugging and detailed reporting.

The tracker captures what happened at each sample during a benchmark run:
which samples passed, which failed, what the model produced, and how long
it took.  This is invaluable when a benchmark scores poorly — you can
inspect individual predictions instead of just seeing an aggregate number.

Usage::

    tracker = EvaluationTracker()

    # During a run (BenchmarkRunner does this automatically):
    tracker.log(benchmark_id="svg-1", model="gemini",
                sample_id="s001", prediction="left", ground_truth="right",
                elapsed_s=1.2)

    # After:
    tracker.summary()          # Print per-benchmark stats
    tracker.save("run.jsonl")  # Detailed per-sample JSONL
    tracker.failures()         # Just the mismatches
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SampleLog:
    """Record of a single sample's evaluation."""

    benchmark_id: str
    model: str
    sample_id: str
    prediction: Any = None
    ground_truth: Any = None
    model_output: Optional[str] = None
    elapsed_s: float = 0.0
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "benchmark_id": self.benchmark_id,
            "model": self.model,
            "sample_id": self.sample_id,
            "elapsed_s": round(self.elapsed_s, 3),
        }
        if self.error:
            d["error"] = self.error
        else:
            d["prediction"] = _safe_str(self.prediction)
            d["ground_truth"] = _safe_str(self.ground_truth)
        if self.model_output is not None:
            d["model_output"] = _safe_str(self.model_output, max_len=2000)
        return d


def _safe_str(val: Any, max_len: int = 500) -> str:
    """Convert a value to a string, truncating if too long."""
    s = str(val)
    if len(s) > max_len:
        return s[:max_len] + f"... ({len(s)} chars)"
    return s


class EvaluationTracker:
    """Accumulates per-sample logs during a benchmark run.

    Integrates with Python logging — sample-level details go to DEBUG,
    per-benchmark summaries go to INFO.
    """

    def __init__(self) -> None:
        self._logs: List[SampleLog] = []

    def log(
        self,
        benchmark_id: str,
        model: str,
        sample_id: str,
        prediction: Any = None,
        ground_truth: Any = None,
        model_output: Optional[str] = None,
        elapsed_s: float = 0.0,
        error: Optional[str] = None,
    ) -> SampleLog:
        """Record one sample's result."""
        entry = SampleLog(
            benchmark_id=benchmark_id,
            model=model,
            sample_id=sample_id,
            prediction=prediction,
            ground_truth=ground_truth,
            model_output=model_output,
            elapsed_s=elapsed_s,
            error=error,
        )
        self._logs.append(entry)

        if error:
            logger.warning(
                "[%s] %s sample=%s ERROR: %s (%.1fs)",
                benchmark_id, model, sample_id, error, elapsed_s,
            )
        else:
            logger.debug(
                "[%s] %s sample=%s pred=%s gt=%s (%.1fs)",
                benchmark_id, model, sample_id,
                _safe_str(prediction, 80), _safe_str(ground_truth, 80),
                elapsed_s,
            )
        return entry

    def log_batch(
        self,
        benchmark_id: str,
        model: str,
        sample_ids: List[str],
        predictions: List[Any],
        ground_truths: List[Any],
        model_outputs: Optional[List[str]] = None,
        errors: Optional[List[Optional[str]]] = None,
        elapsed_s: float = 0.0,
    ) -> None:
        """Record results for a batch of samples."""
        per_sample = elapsed_s / max(len(sample_ids), 1)
        _outputs = model_outputs or [None] * len(sample_ids)
        _errors = errors or [None] * len(sample_ids)
        for sid, pred, gt, raw, err in zip(
            sample_ids, predictions, ground_truths, _outputs, _errors
        ):
            self.log(
                benchmark_id=benchmark_id,
                model=model,
                sample_id=sid,
                prediction=pred,
                ground_truth=gt,
                model_output=raw,
                error=err,
                elapsed_s=per_sample,
            )
        logger.info(
            "[%s] %s batch: %d samples in %.1fs (%.2fs/sample)",
            benchmark_id, model, len(sample_ids), elapsed_s, per_sample,
        )

    @property
    def logs(self) -> List[SampleLog]:
        return list(self._logs)

    def failures(
        self, benchmark_id: Optional[str] = None,
    ) -> List[SampleLog]:
        """Return logs where prediction != ground_truth or had an error."""
        out = []
        for entry in self._logs:
            if benchmark_id and entry.benchmark_id != benchmark_id:
                continue
            if entry.error or str(entry.prediction) != str(entry.ground_truth):
                out.append(entry)
        return out

    def summary(self) -> str:
        """Per-benchmark/model summary stats."""
        from collections import defaultdict

        stats: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(
            lambda: defaultdict(lambda: {"total": 0, "errors": 0, "elapsed": 0.0})
        )
        for entry in self._logs:
            s = stats[entry.benchmark_id][entry.model]
            s["total"] += 1
            if entry.error:
                s["errors"] += 1
            s["elapsed"] += entry.elapsed_s

        lines = []
        for bid in sorted(stats):
            for model in sorted(stats[bid]):
                s = stats[bid][model]
                lines.append(
                    f"  [{bid}] {model}: {s['total']} samples, "
                    f"{s['errors']} errors, {s['elapsed']:.1f}s total"
                )
        return "\n".join(lines) if lines else "(no samples logged)"

    def save(self, path: str) -> None:
        """Save detailed per-sample logs as JSONL."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            for entry in self._logs:
                f.write(json.dumps(entry.to_dict(), default=str) + "\n")
        logger.info("Saved %d sample logs to %s", len(self._logs), path)

    def __len__(self) -> int:
        return len(self._logs)
