#!/usr/bin/env python3
"""Mine hard template-variant pairs by font-family Jaccard similarity.

Among layouts with font metadata, hard positives are siblings (same template)
whose ``fontFamily`` sets have Jaccard similarity below 0.5, and hard negatives
are layouts from different templates with similarity of at least 0.5.

Usage::

    python scripts/mine_template_hard_pairs.py --dataset-root data/gdb-dataset \\
        --output hard_pairs.json
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", "-o", required=True)
    args = parser.parse_args()

    lica = Path(args.dataset_root) / "lica-data"
    with open(lica / "metadata.csv", encoding="utf-8") as f:
        meta = {r["layout_id"]: r for r in csv.DictReader(f)}
    rendered = {p.stem for p in (lica / "images").glob("*/*.png")}

    fonts = {}
    for path in (lica / "layouts").glob("*/*.json"):
        layout = json.loads(path.read_text(encoding="utf-8"))
        family = {c["fontFamily"] for c in layout.get("components", []) if c.get("fontFamily")}
        if family:
            fonts[path.stem] = family

    positives, negatives = [], []
    for a, b in itertools.combinations(sorted(fonts), 2):
        j = len(fonts[a] & fonts[b]) / len(fonts[a] | fonts[b])
        sibling = meta[a]["template_id"] == meta[b]["template_id"]
        if sibling and j < 0.5:
            positives.append([a, b, round(j, 3)])
        elif not sibling and j >= 0.5:
            negatives.append([a, b, round(j, 3)])

    def usable(lid: str) -> bool:
        return lid in rendered and meta[lid]["category"] != "Videos"

    print(f"layouts with font metadata: {len(fonts)}")
    print(
        f"hard positives: {len(positives)} "
        f"({sum(usable(a) and usable(b) for a, b, _ in positives)} without Videos, rendered)"
    )
    print(
        f"hard negatives: {len(negatives)} "
        f"({sum(j == 1.0 for _, _, j in negatives)} with identical font sets)"
    )
    Path(args.output).write_text(
        json.dumps({"hard_positives": positives, "hard_negatives": negatives}) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
