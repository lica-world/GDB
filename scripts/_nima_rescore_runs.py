"""Post-hoc NIMA rescoring helper for upstream parity runs.

``layout-8`` (``LayerAwareObjectInsertionAndAssetSynthesis``) does not emit
``nima_score`` from its built-in ``evaluate()`` because the metric requires
``pyiqa`` which is an optional evaluator dependency. This helper runs NIMA
against the prediction PNGs that ``claude_code_agent`` persisted under
``outputs/claude-code-media/`` for a parity run and patches the resulting
``nima_score`` into ``outputs/parity_claude_code_run{N}.json`` so downstream
aggregators see a complete score set.

Usage
-----

The script groups predictions by mtime into disjoint windows defined in
``RUN_WINDOWS`` and writes ``results["layout-8"][<model>]["scores"]
["nima_score"]`` on each matching run JSON. Edit ``RUN_WINDOWS`` for the
specific parity run timestamps being rescored.

Requires ``pyiqa`` and ``torch`` in the active environment.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from statistics import fmean

REPO = Path(__file__).resolve().parent.parent
MEDIA = REPO / "outputs" / "claude-code-media"
MODEL = "claude-sonnet-4-20250514"

# Per-run mtime windows for upstream claude-code parity PNGs. Edit when
# rescoring a different set of runs.
RUN_WINDOWS = {
    2: (
        time.mktime(time.strptime("2026-04-20 12:00", "%Y-%m-%d %H:%M")),
        time.mktime(time.strptime("2026-04-20 13:00", "%Y-%m-%d %H:%M")),
    ),
    3: (
        time.mktime(time.strptime("2026-04-20 13:00", "%Y-%m-%d %H:%M")),
        time.mktime(time.strptime("2026-04-20 23:59", "%Y-%m-%d %H:%M")),
    ),
}


def main() -> None:
    import torch
    from PIL import Image
    import pyiqa

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[nima] device={device}")
    metric = pyiqa.create_metric("nima", device=device)

    pngs = sorted(MEDIA.glob("layout-8_*.png"))
    per_run: dict[int, list[Path]] = {run_idx: [] for run_idx in RUN_WINDOWS}
    for p in pngs:
        mt = os.path.getmtime(p)
        for run_idx, (lo, hi) in RUN_WINDOWS.items():
            if lo <= mt < hi:
                per_run[run_idx].append(p)

    for run_idx in RUN_WINDOWS:
        paths = sorted(per_run[run_idx])
        scores: list[float] = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            s = float(metric(img).item())
            scores.append(s)
            print(f"  run{run_idx} {p.name}  nima={s:.4f}")
        if not scores:
            raise RuntimeError(f"no layout-8 PNGs found for run {run_idx}")
        mean = fmean(scores)
        print(f"[run{run_idx}] layout-8 nima_score mean = {mean:.6f} (n={len(scores)})")

        run_json = REPO / "outputs" / f"parity_claude_code_run{run_idx}.json"
        doc = json.loads(run_json.read_text())
        bench = doc["results"]["layout-8"][MODEL]
        bench["scores"]["nima_score"] = mean
        run_json.write_text(json.dumps(doc, indent=2))
        print(f"[patched] {run_json.name}: layout-8.scores.nima_score = {mean:.6f}")


if __name__ == "__main__":
    main()
