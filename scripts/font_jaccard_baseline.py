#!/usr/bin/env python3
"""Font-family Jaccard baseline for template-1, template-2 and template-3.

Each layout is reduced to the set of ``fontFamily`` values of its components.
template-1 predicts "same template" when the Jaccard similarity of the two sets
reaches the threshold (0.05 grid) with the highest accuracy, template-2 ranks the
candidates by similarity to the reference, and template-3 runs average-linkage
agglomerative clustering on Jaccard distance. Scores come from each task's
``evaluate``.

Usage::

    python scripts/font_jaccard_baseline.py --dataset-root data/gdb-dataset
    python scripts/font_jaccard_baseline.py --dataset-root data/gdb-dataset \\
        --data path/to/task-dir --output font_jaccard.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import average_precision_score, roc_auc_score

from gdb import BenchmarkRegistry

THRESHOLDS = [i / 20 for i in range(21)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--data", default=None, help="Task data directory (default: benchmarks/template)"
    )
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    root = Path(args.dataset_root)
    fonts = {}
    for path in (root / "lica-data" / "layouts").glob("*/*.json"):
        layout = json.loads(path.read_text(encoding="utf-8"))
        fonts[path.stem] = {c["fontFamily"] for c in layout.get("components", []) if c.get("fontFamily")}

    def jaccard(a: str, b: str) -> float:
        union = fonts[a] | fonts[b]
        return len(fonts[a] & fonts[b]) / len(union) if union else 1.0

    registry = BenchmarkRegistry()
    registry.discover()
    report = {}
    for task_id in ("template-1", "template-2", "template-3"):
        bench = registry.get(task_id)
        samples = bench.load_data(args.data or bench.resolve_data_dir(root), dataset_root=root)
        gt = [s["ground_truth"] for s in samples]
        if task_id == "template-1":
            sims = [jaccard(s["layout_a"], s["layout_b"]) for s in samples]
            t = max(THRESHOLDS, key=lambda t: sum((x >= t) == bool(g) for x, g in zip(sims, gt)))
            scores = bench.evaluate(["1" if x >= t else "0" for x in sims], gt)
            scores["threshold"] = t
            scores["auc_roc"] = float(roc_auc_score(gt, sims))
            scores["average_precision"] = float(average_precision_score(gt, sims))
        elif task_id == "template-2":
            ranked = [sorted(s["candidates"], key=lambda c: -jaccard(s["reference"], c)) for s in samples]
            scores = bench.evaluate(ranked, gt)
        else:
            labels = []
            for s in samples:
                vocab = sorted(set().union(*(fonts[lid] for lid in s["layouts"])))
                vectors = np.array([[f in fonts[lid] for f in vocab] for lid in s["layouts"]])
                model = AgglomerativeClustering(
                    n_clusters=s["n_clusters"], metric="jaccard", linkage="average"
                )
                labels.append(model.fit_predict(vectors).tolist())
            scores = bench.evaluate(labels, gt)
        report[task_id] = scores
        print(task_id, "  ".join(f"{k} {v:.4f}" for k, v in scores.items()))

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
