#!/usr/bin/env python3
"""Build the tiny ``_verify_data/`` fixture bundled inside the ``gdb`` package.

Run from the repo root **when you change the smoke suite or the sample formats**::

    python scripts/build_verify_dataset.py

The output lives at ``src/gdb/_verify_data/`` and is committed to the repo so
that ``pip install lica-gdb && gdb verify`` works with no downloads and no
API keys.

The fixture covers the ``v0-smoke`` suite only:

* category-1, layout-4, layout-5, typography-1  (CSV-based tasks)
* svg-1                                         (JSON + assets/)
* template-1                                    (JSON + lica-data/)

Images are downsampled to ``MAX_PX`` on the longest side to keep the wheel
small. Scores produced against this fixture are **meaningless** — the stub
model predicts empty strings — so ``gdb verify`` is strictly an install-time
smoke test, not a benchmark run.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image

SOURCE_ROOT = Path("data/gdb-dataset")
DEST_ROOT = Path("src/gdb/_verify_data")

# Samples per task kept in the fixture. 2 is enough to exercise the full
# load → predict → score path without blowing up wheel size.
N_PER_TASK = 2

# Longest-edge cap for bundled images. The stub model doesn't care about
# content; at ~6 PNGs this keeps the fixture well under 100 KB.
MAX_PX = 128


# ----------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------


def _downsample_image(src: Path, dest: Path) -> None:
    """Copy ``src`` to ``dest`` downsampled to ``MAX_PX`` on the long edge."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = im.convert("RGB")
        im.thumbnail((MAX_PX, MAX_PX))
        im.save(dest, format="PNG", optimize=True)


def _copy_text(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


# ----------------------------------------------------------------------------
# Simple CSV tasks (one image per row)
# ----------------------------------------------------------------------------

_CSV_TASKS: List[Tuple[str, str, str]] = [
    # (task_id, source subpath under benchmarks/, dest subpath under benchmarks/)
    ("category-1",   "category/CategoryClassification",   "category/CategoryClassification"),
    ("layout-4",     "layout/AspectRatioClassification",  "layout/AspectRatioClassification"),
    ("layout-5",     "layout/ComponentCount",             "layout/ComponentCount"),
    ("typography-1", "typography/FontFamilyClassification","typography/FontFamilyClassification"),
]


def _build_csv_task(task_id: str, src_subpath: str, dest_subpath: str) -> None:
    src_csv = SOURCE_ROOT / "benchmarks" / src_subpath / "samples.csv"
    dest_csv = DEST_ROOT / "benchmarks" / dest_subpath / "samples.csv"
    dest_csv.parent.mkdir(parents=True, exist_ok=True)

    with src_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        kept: List[Dict[str, str]] = []
        for row in reader:
            img_rel = row.get("image_path", "")
            if not img_rel:
                continue
            src_img = SOURCE_ROOT / img_rel
            if not src_img.is_file():
                continue
            dest_img = DEST_ROOT / img_rel
            _downsample_image(src_img, dest_img)
            kept.append(row)
            if len(kept) >= N_PER_TASK:
                break

    if not kept:
        raise RuntimeError(f"No usable samples found for {task_id} in {src_csv}")

    with dest_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)
    print(f"  {task_id}: {len(kept)} samples → {dest_csv.relative_to(DEST_ROOT)}")


# ----------------------------------------------------------------------------
# svg-1 (JSON + per-record PNG + per-record SVG)
# ----------------------------------------------------------------------------


def _build_svg1() -> None:
    src_json = SOURCE_ROOT / "benchmarks/svg/svg-1.json"
    dest_json = DEST_ROOT / "benchmarks/svg/svg-1.json"
    records_full = json.loads(src_json.read_text(encoding="utf-8"))

    kept: List[dict] = []
    for rec in records_full:
        img_rel = rec.get("image_path", "")
        svg_rel = rec.get("svg_path", "")
        if not img_rel or not svg_rel:
            continue
        src_img = SOURCE_ROOT / "benchmarks/svg" / img_rel
        src_svg = SOURCE_ROOT / "benchmarks/svg" / svg_rel
        if not src_img.is_file() or not src_svg.is_file():
            continue
        dest_img = DEST_ROOT / "benchmarks/svg" / img_rel
        dest_svg = DEST_ROOT / "benchmarks/svg" / svg_rel
        _downsample_image(src_img, dest_img)
        _copy_text(src_svg, dest_svg)
        kept.append(rec)
        if len(kept) >= N_PER_TASK:
            break

    if not kept:
        raise RuntimeError("No usable svg-1 records found")

    dest_json.parent.mkdir(parents=True, exist_ok=True)
    dest_json.write_text(json.dumps(kept, indent=2), encoding="utf-8")
    print(f"  svg-1: {len(kept)} records → {dest_json.relative_to(DEST_ROOT)}")


# ----------------------------------------------------------------------------
# template-1 (JSON + lica-data/layouts + lica-data/images)
# ----------------------------------------------------------------------------


def _layout_template_id(layout_index: Dict[str, str], layout_id: str) -> str:
    return layout_index.get(layout_id, "")


def _copy_layout_assets(
    layout_id: str,
    template_id: str,
    src_data_root: Path,
    dest_data_root: Path,
) -> None:
    """Copy the per-layout files the benchmark may consume, if they exist."""
    if not template_id:
        return
    rel_layout = Path("layouts") / template_id / f"{layout_id}.json"
    rel_image = Path("images") / template_id / f"{layout_id}.png"
    rel_annot = Path("annotations") / template_id / f"{layout_id}.json"

    src_layout = src_data_root / rel_layout
    if src_layout.is_file():
        _copy_text(src_layout, dest_data_root / rel_layout)
    src_image = src_data_root / rel_image
    if src_image.is_file():
        _downsample_image(src_image, dest_data_root / rel_image)
    src_annot = src_data_root / rel_annot
    if src_annot.is_file():
        _copy_text(src_annot, dest_data_root / rel_annot)


def _build_template1() -> None:
    src_json = SOURCE_ROOT / "benchmarks/template/template-1.json"
    dest_json = DEST_ROOT / "benchmarks/template/template-1.json"
    full = json.loads(src_json.read_text(encoding="utf-8"))

    src_data_root = SOURCE_ROOT / full.get("data_root", "lica-data")
    dest_data_root = DEST_ROOT / full.get("data_root", "lica-data")

    pairs = full.get("pairs", [])
    # Take one label=1 and one label=0 pair if we can — exercises both paths.
    chosen: List[dict] = []
    seen_labels: set = set()
    for p in pairs:
        if p.get("label") in seen_labels:
            continue
        chosen.append(p)
        seen_labels.add(p["label"])
        if len(chosen) >= N_PER_TASK:
            break
    if not chosen:
        chosen = pairs[:N_PER_TASK]

    used_layout_ids: List[str] = []
    for p in chosen:
        used_layout_ids.append(p["layout_a"])
        used_layout_ids.append(p["layout_b"])

    mini_index: Dict[str, str] = {}
    for lid in used_layout_ids:
        tid = _layout_template_id(full.get("layout_index", {}), lid)
        if tid:
            mini_index[lid] = tid
        _copy_layout_assets(lid, tid, src_data_root, dest_data_root)

    out = {
        "data_root": full.get("data_root", "lica-data"),
        "layout_index": mini_index,
        "pairs": chosen,
    }
    dest_json.parent.mkdir(parents=True, exist_ok=True)
    dest_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(
        f"  template-1: {len(chosen)} pairs (labels={sorted(seen_labels)}) "
        f"→ {dest_json.relative_to(DEST_ROOT)}"
    )


# ----------------------------------------------------------------------------
# Top-level driver
# ----------------------------------------------------------------------------


def _write_readme() -> None:
    readme = DEST_ROOT / "README.md"
    readme.parent.mkdir(parents=True, exist_ok=True)
    readme.write_text(
        "# gdb verify fixture\n\n"
        "Tiny bundled dataset used by `gdb verify` to confirm an install is\n"
        "functional without any downloads or API keys. Covers the\n"
        "`v0-smoke` suite only.\n\n"
        "**Do not edit by hand.** Regenerate with:\n\n"
        "```bash\n"
        "python scripts/build_verify_dataset.py\n"
        "```\n\n"
        f"Images are downsampled to {MAX_PX}px on the long edge; scores\n"
        "produced against this fixture are **meaningless** by design.\n",
        encoding="utf-8",
    )


def _clean() -> None:
    if DEST_ROOT.exists():
        shutil.rmtree(DEST_ROOT)


def _tree_size(root: Path) -> int:
    total = 0
    for p in root.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def _iter_files(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*")):
        if p.is_file():
            yield p


def main() -> None:
    global SOURCE_ROOT, DEST_ROOT

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-root", default=str(SOURCE_ROOT))
    parser.add_argument("--dest-root", default=str(DEST_ROOT))
    parser.add_argument("--keep-existing", action="store_true",
                        help="Don't wipe dest-root first (default: wipe).")
    args = parser.parse_args()

    SOURCE_ROOT = Path(args.source_root)
    DEST_ROOT = Path(args.dest_root)

    if not SOURCE_ROOT.is_dir():
        raise SystemExit(f"Source not found: {SOURCE_ROOT}")

    if not args.keep_existing:
        _clean()

    DEST_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Building verify fixture at {DEST_ROOT} (source: {SOURCE_ROOT})")

    print("\n[csv tasks]")
    for task_id, src_sub, dest_sub in _CSV_TASKS:
        _build_csv_task(task_id, src_sub, dest_sub)

    print("\n[svg-1]")
    _build_svg1()

    print("\n[template-1]")
    _build_template1()

    _write_readme()

    total = _tree_size(DEST_ROOT)
    n_files = sum(1 for _ in _iter_files(DEST_ROOT))
    print(f"\nDone. {n_files} files, {total / 1024:.1f} KiB at {DEST_ROOT}")


if __name__ == "__main__":
    main()
