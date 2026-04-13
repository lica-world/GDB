"""Auto-discovery registry for benchmark implementations."""

import importlib
import logging
import pkgutil
from types import ModuleType
from typing import Dict, List, Optional

from .base import _REGISTERED_BENCHMARKS, BaseBenchmark, TaskType

logger = logging.getLogger(__name__)


def _import_submodule(modname: str):
    """Import *modname*; log failures instead of failing discovery silently.

    ``ImportError`` / ``ModuleNotFoundError`` are logged at DEBUG (optional
    dependencies). Other errors are logged at WARNING (bugs or environment).
    """
    try:
        return importlib.import_module(modname)
    except ImportError as exc:
        logger.debug("Skipping %s: %s", modname, exc)
        return None
    except Exception as exc:
        logger.warning(
            "Failed to import %s (%s): %s",
            modname,
            type(exc).__name__,
            exc,
        )
        return None


class BenchmarkRegistry:
    """Central registry that discovers and indexes benchmark classes.

    Usage::

        registry = BenchmarkRegistry()
        registry.discover()

        bench = registry.get("svg-1")
        svg_benches = registry.list(domain="svg")
    """

    def __init__(self) -> None:
        self._benchmarks: Dict[str, BaseBenchmark] = {}

    def register(self, benchmark: BaseBenchmark) -> None:
        bid = benchmark.meta.id
        if bid in self._benchmarks:
            existing = self._benchmarks[bid]
            # Same class registered twice (e.g. module re-import) — skip
            if type(existing) is type(benchmark):
                return
            raise ValueError(
                f"Duplicate benchmark id '{bid}': "
                f"{existing!r} vs {benchmark!r}"
            )
        self._benchmarks[bid] = benchmark

    def discover(self) -> None:
        """Import task subpackages and register ``@benchmark``-decorated classes.

        Submodules that fail with ``ImportError`` (e.g. missing optional
        dependencies) are skipped and logged at DEBUG for
        ``design_benchmarks.registry``. Other import failures are logged at
        WARNING.
        """
        import design_benchmarks as root_pkg

        import_cache: Dict[str, Optional[ModuleType]] = {}

        def _cached_import(modname: str) -> Optional[ModuleType]:
            if modname not in import_cache:
                import_cache[modname] = _import_submodule(modname)
            return import_cache[modname]

        for _importer, modname, _ispkg in pkgutil.walk_packages(
            root_pkg.__path__, prefix=root_pkg.__name__ + "."
        ):
            _cached_import(modname)

        for obj in _REGISTERED_BENCHMARKS:
            self.register(obj)

    def get(self, benchmark_id: str) -> BaseBenchmark:
        if benchmark_id not in self._benchmarks:
            available = ", ".join(sorted(self._benchmarks)) or "(none)"
            raise KeyError(
                f"Unknown benchmark '{benchmark_id}'. Available: {available}"
            )
        return self._benchmarks[benchmark_id]

    def list(
        self,
        domain: Optional[str] = None,
        task_type: Optional[TaskType] = None,
        tag: Optional[str] = None,
    ) -> List[BaseBenchmark]:
        results = list(self._benchmarks.values())

        if domain is not None:
            results = [b for b in results if b.meta.domain == domain]
        if task_type is not None:
            results = [b for b in results if b.meta.task_type == task_type]
        if tag is not None:
            results = [b for b in results if tag in b.meta.tags]

        return sorted(results, key=lambda b: b.meta.id)

    def list_ids(self, **kwargs) -> List[str]:
        return [b.meta.id for b in self.list(**kwargs)]

    def domains(self) -> List[str]:
        return sorted({b.meta.domain for b in self._benchmarks.values()})

    def __len__(self) -> int:
        return len(self._benchmarks)

    def __contains__(self, benchmark_id: str) -> bool:
        return benchmark_id in self._benchmarks
