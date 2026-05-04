"""Tests for :mod:`gdb.suites`.

These tests use a real discovered registry so suite routing is validated
against the actual task set. That means adding a task will also need a suite
expectation update here if the task changes the understanding/generation
balance, which we consider a feature rather than a bug.
"""

from __future__ import annotations

import pytest

from gdb.suites import describe_suite, list_suites, resolve_suite

KNOWN_SUITES = {"v0-all", "v0-smoke", "v0-understanding", "v0-generation"}


def test_list_suites_contains_all_known():
    names = set(list_suites())
    assert KNOWN_SUITES.issubset(names)


def test_list_suites_is_sorted_and_unique():
    names = list_suites()
    assert names == sorted(set(names))


def test_v0_smoke_returns_fixed_task_ids(registry):
    # Static suite: must match the hardcoded list in gdb.suites. If this ever
    # changes, the bundled verify fixture in src/gdb/_verify_data/ probably
    # needs to change too.
    expected = {
        "category-1",
        "layout-4",
        "layout-5",
        "typography-1",
        "svg-1",
        "template-1",
    }
    assert set(resolve_suite("v0-smoke", registry)) == expected


def test_v0_all_matches_registry_contents(registry):
    ids = resolve_suite("v0-all", registry)
    assert set(ids) == {b.meta.id for b in registry.list()}
    assert ids == sorted(ids)


def test_v0_understanding_and_generation_partition_v0_all(registry):
    u = set(resolve_suite("v0-understanding", registry))
    g = set(resolve_suite("v0-generation", registry))
    all_ = set(resolve_suite("v0-all", registry))
    assert u | g == all_
    assert u & g == set()


def test_v0_understanding_is_nonempty_and_sorted(registry):
    ids = resolve_suite("v0-understanding", registry)
    assert ids
    assert ids == sorted(ids)


def test_v0_generation_is_nonempty_and_sorted(registry):
    ids = resolve_suite("v0-generation", registry)
    assert ids
    assert ids == sorted(ids)


def test_resolve_suite_returns_list_copy(registry):
    # Mutating the returned list must not corrupt the cached static suite.
    a = resolve_suite("v0-smoke", registry)
    a.append("not-a-real-benchmark")
    b = resolve_suite("v0-smoke", registry)
    assert "not-a-real-benchmark" not in b


def test_resolve_suite_unknown_raises_keyerror(registry):
    with pytest.raises(KeyError) as excinfo:
        resolve_suite("v99-all", registry)
    msg = str(excinfo.value)
    assert "v99-all" in msg
    # The message should list the known suites to help users.
    assert "v0-all" in msg
    assert "v0-smoke" in msg


def test_describe_suite_static_shape(registry):
    info = describe_suite("v0-smoke", registry)
    assert info["name"] == "v0-smoke"
    assert info["kind"] == "static"
    assert info["n_tasks"] == len(info["task_ids"])
    assert info["n_tasks"] > 0


def test_describe_suite_dynamic_shape(registry):
    info = describe_suite("v0-all", registry)
    assert info["name"] == "v0-all"
    assert info["kind"] == "dynamic"
    assert info["n_tasks"] == len(info["task_ids"])
    assert info["n_tasks"] == len(list(registry.list()))


def test_describe_suite_unknown_propagates_keyerror(registry):
    with pytest.raises(KeyError):
        describe_suite("nope", registry)
