"""
``gdb.evaluation`` — tracking and reporting.

Sub-modules
-----------
- ``tracker``     — EvaluationTracker: per-sample log writer (JSONL)
- ``reporting``   — BenchmarkResult, RunReport: aggregate result export (CSV / JSON)

Metrics live in ``gdb.metrics``.

Quick usage::

    from gdb.evaluation.tracker import EvaluationTracker
    from gdb.evaluation.reporting import RunReport
    tracker = EvaluationTracker()
    tracker.log(benchmark_id="category-1", model="gpt-4o", sample_id="s1",
                prediction="social", ground_truth="social")
    tracker.save("runs/run.jsonl")
"""

from gdb.evaluation.reporting import BenchmarkResult, RunReport
from gdb.evaluation.tracker import EvaluationTracker

__all__ = ["EvaluationTracker", "BenchmarkResult", "RunReport"]
