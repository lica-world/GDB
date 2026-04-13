"""
``design_benchmarks.evaluation`` — tracking and reporting.

Sub-modules
-----------
- ``tracker``     — EvaluationTracker: per-sample log writer (JSONL)
- ``reporting``   — BenchmarkResult, RunReport: aggregate result export (CSV / JSON)

Metrics live in ``design_benchmarks.metrics``.

Quick usage::

    from design_benchmarks.evaluation.tracker import EvaluationTracker
    from design_benchmarks.evaluation.reporting import RunReport
    tracker = EvaluationTracker()
    tracker.log(benchmark_id="category-1", model="gpt-4o", sample_id="s1",
                prediction="social", ground_truth="social")
    tracker.save("runs/run.jsonl")
"""

from design_benchmarks.evaluation.reporting import BenchmarkResult, RunReport
from design_benchmarks.evaluation.tracker import EvaluationTracker

__all__ = ["EvaluationTracker", "BenchmarkResult", "RunReport"]
