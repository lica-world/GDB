"""End-to-end smoke tests for the ``gdb`` CLI.

These are intentionally cheap: they invoke ``python -m gdb`` as a subprocess
and check that a handful of read-only commands exit cleanly and print the
expected headings. Anything that requires model inference belongs in
:mod:`tests.test_cli_helpers` (pure logic) or in the ``gdb verify`` runtime
smoke path (executed out of band).
"""

from __future__ import annotations

import subprocess
import sys


def _run(*args: str) -> subprocess.CompletedProcess:
    """Invoke ``python -m gdb`` with the given args."""
    return subprocess.run(
        [sys.executable, "-m", "gdb", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_version_flag_prints_lica_gdb_version():
    result = _run("--version")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().startswith("lica-gdb ")


def test_help_lists_core_subcommands():
    result = _run("--help")
    assert result.returncode == 0, result.stderr
    for subcmd in ("list", "info", "suites", "eval", "verify", "submit", "collect"):
        assert subcmd in result.stdout, f"`{subcmd}` missing from --help output"


def test_list_prints_39_benchmarks():
    result = _run("list")
    assert result.returncode == 0, result.stderr
    # The exact integer is worth asserting: it's the number the paper and
    # README both reference, and the CI smoke test also pins it.
    assert "39 benchmark(s)" in result.stdout


def test_suites_lists_all_v0_suites():
    result = _run("suites")
    assert result.returncode == 0, result.stderr
    for suite in ("v0-all", "v0-smoke", "v0-understanding", "v0-generation"):
        assert suite in result.stdout, f"suite `{suite}` missing from `gdb suites`"


def test_info_reports_known_benchmark():
    result = _run("info", "category-1")
    assert result.returncode == 0, result.stderr
    assert "category-1" in result.stdout
    assert "understanding" in result.stdout.lower()


def test_info_unknown_benchmark_exits_nonzero():
    result = _run("info", "does-not-exist")
    assert result.returncode != 0
