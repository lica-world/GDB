"""Shared pytest fixtures for the GDB test suite."""

from __future__ import annotations

import pytest

from gdb.registry import BenchmarkRegistry


@pytest.fixture(scope="session")
def registry() -> BenchmarkRegistry:
    """A fully-discovered registry shared across the test session.

    Discovery walks :mod:`gdb.tasks` once; tests that only need to read the
    registry (``list``, ``get``, ``list_ids``) can reuse this fixture freely.
    """
    reg = BenchmarkRegistry()
    reg.discover()
    return reg
