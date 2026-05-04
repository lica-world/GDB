"""Tests for pure-function helpers in :mod:`gdb.cli`.

The CLI has a lot of orchestration code; these tests cover the small, pure
pieces that sit under argparse and above the model/runner layers. They don't
exercise any network, model loading, or benchmark execution.
"""

from __future__ import annotations

import argparse
import json

import pytest

from gdb.cli import (
    _parse_json_dict_arg,
    _parse_model_spec,
    _render_markdown_report,
    _resolve_benchmark_ids,
)

# ---------------------------------------------------------------------------
# _parse_model_spec
# ---------------------------------------------------------------------------


def test_parse_model_spec_basic():
    name, provider, model_id = _parse_model_spec("openai:gpt-4o")
    assert name == "openai:gpt-4o"
    assert provider == "openai"
    assert model_id == "gpt-4o"


def test_parse_model_spec_with_alias():
    name, provider, model_id = _parse_model_spec("myalias=openai:gpt-4o")
    assert name == "myalias"
    assert provider == "openai"
    assert model_id == "gpt-4o"


def test_parse_model_spec_strips_whitespace():
    name, provider, model_id = _parse_model_spec(
        "  myalias  =  anthropic  :  claude-haiku-4-5  "
    )
    assert name == "myalias"
    assert provider == "anthropic"
    assert model_id == "claude-haiku-4-5"


def test_parse_model_spec_custom_entrypoint_with_colon():
    # model_id can itself contain colons (e.g. ``module.path:attr``) because
    # we split only on the first ``:``. This is the shape used by the
    # ``custom`` provider.
    name, provider, model_id = _parse_model_spec("custom:my_pkg.wrapper:build")
    assert provider == "custom"
    assert model_id == "my_pkg.wrapper:build"
    assert name == "custom:my_pkg.wrapper:build"


def test_parse_model_spec_missing_colon_raises():
    with pytest.raises(ValueError, match="Invalid --multi-models spec"):
        _parse_model_spec("openai-gpt-4o")


def test_parse_model_spec_unknown_provider_raises():
    with pytest.raises(ValueError) as excinfo:
        _parse_model_spec("bogus:some-model")
    msg = str(excinfo.value)
    assert "Unknown provider" in msg
    # The error should enumerate the valid providers.
    assert "openai" in msg
    assert "anthropic" in msg


# ---------------------------------------------------------------------------
# _parse_json_dict_arg
# ---------------------------------------------------------------------------


def test_parse_json_dict_arg_none_returns_empty_dict():
    assert _parse_json_dict_arg(None, field_name="x") == {}


def test_parse_json_dict_arg_empty_string_returns_empty_dict():
    assert _parse_json_dict_arg("", field_name="x") == {}
    assert _parse_json_dict_arg("   ", field_name="x") == {}


def test_parse_json_dict_arg_dict_is_passthrough():
    d = {"foo": 1, "bar": [2, 3]}
    assert _parse_json_dict_arg(d, field_name="x") is d


def test_parse_json_dict_arg_json_string():
    assert _parse_json_dict_arg('{"k": "v"}', field_name="x") == {"k": "v"}


def test_parse_json_dict_arg_reads_file(tmp_path):
    path = tmp_path / "init.json"
    path.write_text('{"checkpoint": "/models/foo", "dtype": "bfloat16"}')
    result = _parse_json_dict_arg(str(path), field_name="custom init kwargs")
    assert result == {"checkpoint": "/models/foo", "dtype": "bfloat16"}


def test_parse_json_dict_arg_rejects_non_dict_json():
    with pytest.raises(ValueError, match="JSON object/dict"):
        _parse_json_dict_arg("[1, 2, 3]", field_name="x")


def test_parse_json_dict_arg_rejects_wrong_type():
    with pytest.raises(ValueError, match="JSON object/dict"):
        _parse_json_dict_arg(42, field_name="x")


# ---------------------------------------------------------------------------
# _resolve_benchmark_ids
# ---------------------------------------------------------------------------


def _ns(**kwargs) -> argparse.Namespace:
    return argparse.Namespace(**kwargs)


def test_resolve_benchmark_ids_from_suite(registry):
    ids = _resolve_benchmark_ids(
        _ns(suite="v0-smoke", benchmarks=None), registry
    )
    assert "category-1" in ids
    assert "svg-1" in ids


def test_resolve_benchmark_ids_from_explicit_list(registry):
    ids = _resolve_benchmark_ids(
        _ns(suite=None, benchmarks=["layout-4", "svg-1"]), registry
    )
    assert ids == ["layout-4", "svg-1"]


def test_resolve_benchmark_ids_conflicting_args(registry):
    with pytest.raises(SystemExit, match="either --suite or --benchmarks"):
        _resolve_benchmark_ids(
            _ns(suite="v0-all", benchmarks=["layout-4"]), registry
        )


def test_resolve_benchmark_ids_no_selection(registry):
    with pytest.raises(SystemExit, match="One of --suite or --benchmarks"):
        _resolve_benchmark_ids(_ns(suite=None, benchmarks=None), registry)


def test_resolve_benchmark_ids_unknown_suite(registry):
    with pytest.raises(SystemExit, match="Unknown suite"):
        _resolve_benchmark_ids(_ns(suite="v99-bogus", benchmarks=None), registry)


# ---------------------------------------------------------------------------
# _render_markdown_report
# ---------------------------------------------------------------------------


def test_render_markdown_report_empty_results():
    md = _render_markdown_report({"metadata": {"run_id": "abc"}, "results": {}})
    assert md.startswith("# GDB run report")
    assert "## Metadata" in md
    assert "- **run_id**: abc" in md
    assert "## Results" in md
    assert "_(empty)_" in md


def test_render_markdown_report_includes_table_header_and_rows():
    report = {
        "metadata": {},
        "results": {
            "category-1": {
                "openai:gpt-4o": {
                    "count": 10,
                    "failure_rate": 0.1,
                    "scores": {"accuracy": 0.8, "macro_f1": 0.75},
                },
            },
        },
    }
    md = _render_markdown_report(report)
    # Header row
    assert "| Benchmark | Model | n | fail_rate | accuracy | macro_f1 |" in md
    # Separator row
    assert "|---|---|---|---|---|---|" in md
    # Data row
    assert "| category-1 | openai:gpt-4o | 10 | 10.0% | 0.8000 | 0.7500 |" in md


def test_render_markdown_report_handles_missing_metric_as_em_dash():
    # One benchmark reports accuracy; another reports only f1. Each column
    # should render "—" when the metric is absent for a given row.
    report = {
        "metadata": {},
        "results": {
            "bench-a": {
                "stub": {
                    "count": 2,
                    "failure_rate": 0.0,
                    "scores": {"accuracy": 1.0},
                },
            },
            "bench-b": {
                "stub": {
                    "count": 2,
                    "failure_rate": 0.0,
                    "scores": {"f1": 0.5},
                },
            },
        },
    }
    md = _render_markdown_report(report)
    assert "| bench-a | stub | 2 | 0.0% | 1.0000 | — |" in md
    assert "| bench-b | stub | 2 | 0.0% | — | 0.5000 |" in md


def test_render_markdown_report_missing_metadata_omits_section():
    md = _render_markdown_report({"results": {}})
    assert "## Metadata" not in md


def test_render_markdown_report_output_is_parseable_markdown_table():
    # Sanity: the header + separator + at least one row are present in the
    # right order with consistent pipe counts.
    report = {
        "metadata": {"model": "stub"},
        "results": {
            "t1": {"stub": {"count": 1, "failure_rate": 0.0, "scores": {"x": 1.0}}}
        },
    }
    lines = _render_markdown_report(report).splitlines()
    table_lines = [line for line in lines if line.startswith("|")]
    assert len(table_lines) >= 3  # header, separator, at least one row
    pipe_counts = [line.count("|") for line in table_lines]
    assert len(set(pipe_counts)) == 1  # all rows have the same column count


# ---------------------------------------------------------------------------
# Sanity: we can still JSON-roundtrip the report renderer's input contract
# (guards against accidental changes to the expected dict shape).
# ---------------------------------------------------------------------------


def test_render_markdown_report_accepts_json_roundtripped_input():
    report = {
        "metadata": {"suite": "v0-smoke"},
        "results": {
            "category-1": {
                "stub": {"count": 2, "failure_rate": 0.0, "scores": {"acc": 0.5}}
            }
        },
    }
    roundtripped = json.loads(json.dumps(report))
    assert _render_markdown_report(roundtripped) == _render_markdown_report(report)
