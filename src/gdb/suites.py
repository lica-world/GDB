"""Named benchmark suites.

A *suite* is a named list of benchmark IDs. Papers should cite a suite name
(e.g. ``gdb-v0-all``) alongside the ``lica-gdb`` package version, so numbers
reported in different papers refer to the same evaluation set.

.. note::

    The ``v0-*`` prefix is deliberate: **GDB's evaluation definitions are not
    yet frozen**. Tasks, prompts, sample selection, and metric wiring may still
    change between ``lica-gdb`` releases. A ``v1.0-*`` suite family will be
    introduced once those definitions are pinned with a documented fingerprint;
    until then, cite the suite name *and* ``lica-gdb`` package version together.

Two kinds of suites:

* **Dynamic suites** are derived from the registry at call time
  (``v0-all``, ``v0-understanding``, ``v0-generation``). They stay in sync
  with whatever the installed ``lica-gdb`` version knows how to run, so
  ``v0-all`` on package 0.2.0 and ``v0-all`` on 0.2.1 may differ.

* **Static suites** are hardcoded lists (``v0-smoke``). The exact set of
  tasks is fixed in source and does not grow unexpectedly.

The public entry point is :func:`resolve_suite`, which takes a suite name and
a discovered :class:`~gdb.registry.BenchmarkRegistry` and returns a concrete
list of benchmark IDs.
"""

from __future__ import annotations

from typing import Dict, List

from .base import TaskType
from .registry import BenchmarkRegistry

_SMOKE_SUITE: List[str] = [
    "category-1",
    "layout-4",
    "layout-5",
    "typography-1",
    "svg-1",
    "template-1",
]

_STATIC_SUITES: Dict[str, List[str]] = {
    "v0-smoke": _SMOKE_SUITE,
}

_DYNAMIC_SUITES = {
    "v0-all",
    "v0-understanding",
    "v0-generation",
}


def list_suites() -> List[str]:
    """Return all known suite names (static + dynamic)."""
    return sorted(set(_STATIC_SUITES) | _DYNAMIC_SUITES)


def resolve_suite(name: str, registry: BenchmarkRegistry) -> List[str]:
    """Resolve a suite name to a sorted list of benchmark IDs.

    Raises
    ------
    KeyError
        If ``name`` is not a known suite.
    """
    if name in _STATIC_SUITES:
        return list(_STATIC_SUITES[name])

    if name == "v0-all":
        return sorted(b.meta.id for b in registry.list())
    if name == "v0-understanding":
        return sorted(
            b.meta.id for b in registry.list(task_type=TaskType.UNDERSTANDING)
        )
    if name == "v0-generation":
        return sorted(
            b.meta.id for b in registry.list(task_type=TaskType.GENERATION)
        )

    raise KeyError(
        f"Unknown suite {name!r}. Known suites: {', '.join(list_suites())}"
    )


def describe_suite(name: str, registry: BenchmarkRegistry) -> Dict[str, object]:
    """Return a metadata dict describing the suite (name, size, task IDs)."""
    ids = resolve_suite(name, registry)
    return {
        "name": name,
        "kind": "static" if name in _STATIC_SUITES else "dynamic",
        "n_tasks": len(ids),
        "task_ids": ids,
    }
