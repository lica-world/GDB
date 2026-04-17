#!/usr/bin/env python3
"""Normalize string-encoded layouts in template-4/template-5 task JSON.

Historically a few fields in the template generation tasks were written to
disk as JSON-encoded strings:

  template-4.json: context_layouts[i], skeleton, ground_truth
  template-5.json: designated_layout, context_layouts[i], ground_truth

The corresponding benchmark code in ``gdb/tasks/template.py`` usually
assumes these fields are already dicts (``build_model_input`` passes them
straight through ``json.dumps``, and ``_evaluate_template_generation``
calls ``.get(...)`` on them). The mismatch manifests as:

  * garbled prompts (doubly-escaped JSON in the context the model sees), and
  * a hard crash for ``template-5`` evaluation:
    ``AttributeError: 'str' object has no attribute 'get'``.

This script rewrites the affected files in place so every layout-shaped
field is a plain dict / list[dict]. Loaders that previously worked on
strings continue to work (``json.loads(dict)`` obviously doesn't fire, but
the existing ``isinstance(gt_raw, dict)`` branch already handles the dict
shape).

Run from repo root:
    python scripts/normalize_template_data.py                 # default paths
    python scripts/normalize_template_data.py --dry-run       # report only
    python scripts/normalize_template_data.py --root DIR      # custom dataset dir
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Tuple


def _maybe_parse(value: Any) -> Tuple[Any, bool]:
    """If ``value`` is a JSON-encoded object or array, return the parsed
    value and ``True``; otherwise return it unchanged with ``False``."""
    if not isinstance(value, str):
        return value, False
    stripped = value.lstrip()
    if not stripped.startswith(("{", "[")):
        return value, False
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return value, False
    if isinstance(parsed, (dict, list)):
        return parsed, True
    return value, False


def _normalize_scalar(
    record: dict, field: str, touched: Iterable[str],
) -> bool:
    if field not in record:
        return False
    new_val, changed = _maybe_parse(record[field])
    if changed:
        record[field] = new_val
        touched_list = list(touched)
        touched_list.append(field)
        return True
    return False


def _normalize_list(
    record: dict, field: str,
) -> int:
    """For list fields like context_layouts, parse each item if string."""
    if field not in record or not isinstance(record[field], list):
        return 0
    changed_count = 0
    out: list = []
    for item in record[field]:
        new_val, changed = _maybe_parse(item)
        out.append(new_val)
        if changed:
            changed_count += 1
    if changed_count:
        record[field] = out
    return changed_count


# Per-task schema: which fields are layout-shaped scalars vs lists.
SCHEMAS: dict[str, dict[str, list[str]]] = {
    "template-4": {
        "scalars": ["skeleton", "ground_truth"],
        "lists": ["context_layouts"],
    },
    "template-5": {
        "scalars": ["designated_layout", "ground_truth"],
        "lists": ["context_layouts"],
    },
}


def normalize_file(path: Path, dry_run: bool = False) -> int:
    """Rewrite ``path`` in place and return the number of fields changed."""
    task_id = path.stem
    schema = SCHEMAS.get(task_id)
    if schema is None:
        return 0

    data = json.loads(path.read_text())
    problems = data.get("problems", [])
    total_changed = 0
    per_field: dict[str, int] = {}

    for prob in problems:
        if not isinstance(prob, dict):
            continue
        for field in schema["scalars"]:
            val, changed = _maybe_parse(prob.get(field))
            if changed:
                prob[field] = val
                total_changed += 1
                per_field[field] = per_field.get(field, 0) + 1
        for field in schema["lists"]:
            n = _normalize_list(prob, field)
            if n:
                total_changed += n
                per_field[field] = per_field.get(field, 0) + n

    if total_changed and not dry_run:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")

    mode = "[dry-run]" if dry_run else "[write  ]"
    details = (
        ", ".join(f"{f}={n}" for f, n in sorted(per_field.items()))
        if per_field else "no changes"
    )
    print(f"{mode} {path.name}: {total_changed} field(s) updated  ({details})")
    return total_changed


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default="data/lica-benchmarks-dataset/benchmarks/template",
        help="Directory containing template-*.json files",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    root = Path(args.root)
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 2

    total = 0
    for task_id in sorted(SCHEMAS):
        p = root / f"{task_id}.json"
        if not p.is_file():
            print(f"warn: {p} not found, skipping")
            continue
        total += normalize_file(p, dry_run=args.dry_run)

    verb = "would update" if args.dry_run else "updated"
    print(f"\nTotal: {verb} {total} field(s) across {len(SCHEMAS)} file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
