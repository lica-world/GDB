"""Design benchmarks for layout understanding, editing, and generation."""

from importlib.metadata import PackageNotFoundError, version

from .base import BaseBenchmark, BenchmarkMeta, TaskType, benchmark
from .evaluation.reporting import BenchmarkResult, RunReport
from .evaluation.tracker import EvaluationTracker
from .registry import BenchmarkRegistry
from .runner import BenchmarkRunner

try:
    __version__ = version("lica-bench")
except PackageNotFoundError:  # e.g. running from a checkout without install
    __version__ = "0.1.0"

__all__ = [
    "__version__",
    "BaseBenchmark",
    "BenchmarkMeta",
    "BenchmarkRegistry",
    "BenchmarkResult",
    "BenchmarkRunner",
    "EvaluationTracker",
    "RunReport",
    "TaskType",
    "benchmark",
]
