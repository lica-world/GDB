"""Data structures for benchmark results and reports."""

import csv as csv_mod
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import numpy as np


@dataclass
class BenchmarkResult:
    """Results for a single benchmark run against one model."""

    benchmark_id: str
    model: str
    count: int
    success_count: int
    failure_count: int
    scores: Dict[str, float] = field(default_factory=dict)

    @property
    def failure_rate(self) -> float:
        if self.count <= 0:
            return 0.0
        return self.failure_count / self.count


@dataclass
class RunReport:
    """Aggregated results across multiple benchmarks and models."""

    results: Dict[str, Dict[str, BenchmarkResult]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = ["=" * 80, "BENCHMARK RESULTS", "=" * 80]
        for bid, model_results in sorted(self.results.items()):
            lines.append(f"\n{bid}:")
            for model, result in sorted(model_results.items()):
                score_strs = [
                    f"{k}={v:.4f}" for k, v in sorted(result.scores.items())
                ]
                lines.append(
                    f"  {model:25s}: {', '.join(score_strs)}  "
                    f"(n={result.count}, ok={result.success_count}, "
                    f"fail={result.failure_count}, fail_rate={result.failure_rate:.1%})"
                )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"metadata": self.metadata, "results": {}}
        for bid, model_results in self.results.items():
            out["results"][bid] = {}
            for model, result in model_results.items():
                out["results"][bid][model] = {
                    "scores": result.scores,
                    "count": result.count,
                    "success_count": result.success_count,
                    "failure_count": result.failure_count,
                    "failure_rate": result.failure_rate,
                }
        return out

    def save(self, path: str) -> None:
        """Save report. Format inferred from extension (.csv or .json)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        if p.suffix.lower() == ".csv":
            rows: list = []
            all_keys: dict = {}
            for bid, model_results in self.results.items():
                for model_name, result in model_results.items():
                    row = {
                        "benchmark": bid,
                        "model": model_name,
                        "n": result.count,
                        "success_count": result.success_count,
                        "failure_count": result.failure_count,
                        "failure_rate": result.failure_rate,
                    }
                    row.update(result.scores)
                    rows.append(row)
                    for k in row:
                        all_keys[k] = None
            if rows:
                with open(p, "w", newline="", encoding="utf-8") as f:
                    writer = csv_mod.DictWriter(
                        f, fieldnames=list(all_keys), extrasaction="ignore",
                    )
                    writer.writeheader()
                    writer.writerows(rows)
        else:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2, default=str)


def load_from_csv(
    csv_path: str,
    benchmarks: Dict[str, Any],
    task_column: str = "task",
    expected_column: str = "expected_output",
) -> RunReport:
    """Evaluate benchmarks from a pre-computed CSV of model outputs.

    Parameters
    ----------
    csv_path : str
        Path to CSV with columns: task, expected_output, <model>_output.
    benchmarks : dict
        Mapping of benchmark ID/name → benchmark instance (must have ``evaluate``).
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv_mod.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    models = [
        fn[: -len("_output")]
        for fn in fieldnames
        if fn.endswith("_output") and fn != "expected_output"
    ]

    report = RunReport(
        metadata={
            "csv_path": csv_path,
            "total_rows": len(rows),
            "models": models,
        }
    )

    accumulators: Dict[str, Dict[str, Dict[str, list]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    counts: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"total": 0, "failures": 0})
    )

    for row in rows:
        task = row.get(task_column, "")
        expected = row.get(expected_column, "")
        bench = benchmarks.get(task)
        if bench is None:
            continue

        for model in models:
            counts[bench.meta.id][model]["total"] += 1
            predicted = row.get(f"{model}_output", "")
            if str(predicted).startswith("ERROR"):
                counts[bench.meta.id][model]["failures"] += 1
                continue

            scores = bench.evaluate([predicted], [expected])
            for metric, value in scores.items():
                if value != float("inf"):
                    accumulators[bench.meta.id][model][metric].append(value)

    for bid, model_counts in counts.items():
        report.results[bid] = {}
        for model, summary in model_counts.items():
            metric_lists = accumulators[bid][model]
            scores = {k: float(np.mean(v)) for k, v in metric_lists.items()}
            total = summary["total"]
            failures = summary["failures"]
            successes = max(total - failures, 0)
            report.results[bid][model] = BenchmarkResult(
                benchmark_id=bid,
                model=model,
                scores=scores,
                count=total,
                success_count=successes,
                failure_count=failures,
            )

    return report
