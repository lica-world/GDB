#!/usr/bin/env python3
"""Build the layer-order, image-rotation, crop-shape and frame-detection samples (layout-9 to layout-12).

These tasks need fields that only the v1 Lica layouts carry: the nested component tree, image
alt text and element subtypes. The v1 layouts are read from ``--v1-dir`` (``manifest.jsonl`` and
``layouts/``) and downloaded there first when missing. Samples cover the static layouts of the
CategoryClassification split, with up to three images per layout for the image tasks, and are
written as ``samples.csv`` under ``<dataset-root>/benchmarks/layout/``.

Usage::

    python scripts/build_lica_v1_tasks.py --dataset-root data/gdb-dataset
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from urllib.request import urlopen

V1_URL = "https://storage.googleapis.com/lica-ml-data/datasets/lica-data-public/v1"
VECTOR_TYPES = {"frame_or_svg", "lottie_svg", "graph_svg", "svg_stroke"}

LAYER_ORDER_PROMPT = (
    "You are a design layout analyst. Look at this rendered design template.\n\n"
    "The following elements are present in this layout:\n"
    "{element_list}\n\n"
    "List these elements in order from BACK (bottom layer, drawn first) "
    "to FRONT (top layer, drawn last), based on their visual stacking.\n\n"
    "Respond with ONLY the element identifiers in order, separated by commas.\n"
    'Example: "E1, E3, E2, E4"\n\n'
    "Do not include any explanation."
)
IMAGE_PROMPT = (
    "You are a design expert. Look at this rendered design template.\n\n"
    'Focus on the image element described as: "{description}"\n\n'
)
ROTATION_PROMPT = IMAGE_PROMPT + (
    "Is this image rotated from its normal axis-aligned orientation? If so, estimate the rotation angle "
    "in degrees (0° = normal; positive = clockwise; negative = counter-clockwise; range -180 to +180).\n\n"
    'Respond with ONLY a JSON object. Example:\n{{"is_rotated": true, "angle": -15}}'
)
CROP_PROMPT = IMAGE_PROMPT + (
    "Is this image cropped to a non-rectangular shape? If so, classify the crop shape.\n\n"
    'Shape categories: "none" (standard rectangular), "rectangle" (different aspect ratio crop), '
    '"rounded_rectangle", "circle" (or elliptical), "polygon" (star, hexagon, etc.), "organic" (freeform curved).\n\n'
    'Respond with ONLY a JSON object. Example:\n{{"is_cropped": true, "crop_shape": "circle"}}'
)
FRAME_PROMPT = IMAGE_PROMPT + (
    "Is this image placed inside a decorative frame? A decorative frame is a non-rectangular or ornamented "
    "visual border/container around the image (e.g. a circular mask, a shaped cutout, a border with "
    "decorative elements). A plain rectangular bounding box does NOT count as a frame.\n\n"
    'Respond with ONLY a JSON object. Example:\n{{"is_framed": true}}'
)

PATH_TOKEN = re.compile(r"([MLHVCSQTAZmlhvcsqtaz])|(-?\d*\.?\d+(?:e-?\d+)?)")
PATH_ARITY = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "T": 2, "A": 7}


def sample_indices(n: int, k: int, seed: str) -> list:
    if n <= k:
        return list(range(n))
    return sorted(range(n), key=lambda i: hashlib.md5(f"{seed}_{i}".encode()).hexdigest())[:k]


def describe(component: dict, idx: int) -> str:
    ctype = str(component.get("type") or "").upper()
    if ctype in ("TEXT", "IMAGE"):
        label = ctype.lower()
    elif ctype == "GROUP":
        subtype = str(component.get("data0_element_type") or "").lower()
        label = "vector" if subtype in VECTOR_TYPES else "group"
    else:
        label = "unknown"
    text = str(component.get("text") or "")[:50]
    if label == "text" and text:
        return f'E{idx + 1}: text — "{text}"'
    alt = str(component.get("alt") or (component.get("element") or {}).get("alt", "") or "")[:60]
    return f"E{idx + 1}: {label} — {alt}" if alt else f"E{idx + 1}: {label}"


def walk(components, ancestors=()):
    for c in components or []:
        yield c, ancestors
        yield from walk(c.get("components"), ancestors + (c,))


def rotation(component: dict) -> float:
    m = re.search(r"rotate\(\s*(-?[\d.]+)deg", str((component.get("style") or {}).get("transform", "")))
    return float(m.group(1)) if m else 0.0


def clip_path(component: dict, ancestors: tuple):
    for c in (component, *reversed(ancestors)):
        clip = (c.get("style") or {}).get("clipPath")
        if clip:
            return clip
    return None


def path_segments(d: str) -> list:
    tokens = [(m.group(1), m.group(2)) for m in PATH_TOKEN.finditer(d)]
    segs, cur, start, i = [], (0.0, 0.0), (0.0, 0.0), 0
    while i < len(tokens):
        cmd, i = tokens[i][0], i + 1
        nums = []
        while i < len(tokens) and tokens[i][0] is None:
            nums.append(float(tokens[i][1]))
            i += 1
        op, rel = cmd.upper(), cmd.islower()
        if op == "Z":
            if cur != start:
                segs.append(("line", cur, start))
            cur = start
            continue
        k = PATH_ARITY[op]
        for j in range(0, len(nums) - k + 1, k):
            a = nums[j:j + k]
            if op == "H":
                end = (a[0] + (cur[0] if rel else 0.0), cur[1])
            elif op == "V":
                end = (cur[0], a[0] + (cur[1] if rel else 0.0))
            else:
                end = (a[-2] + cur[0], a[-1] + cur[1]) if rel else (a[-2], a[-1])
            if op == "M" and j == 0:
                start = end
            else:
                segs.append(("line" if op in "MLHV" else "curve", cur, end))
            cur = end
    return [s for s in segs if math.dist(s[1], s[2]) > 1e-6]


def crop_shape(clip) -> str:
    if not clip:
        return "none"
    m = re.search(r'path\("([^"]*)"\)', str(clip))
    if not m:
        return "organic"
    segs = path_segments(m.group(1))
    lines = [s for s in segs if s[0] == "line"]
    curves = [s for s in segs if s[0] == "curve"]
    axis_aligned = all(abs(s[1][0] - s[2][0]) < 0.5 or abs(s[1][1] - s[2][1]) < 0.5 for s in lines)
    if not curves:
        return "rectangle" if len(lines) == 4 and axis_aligned else "polygon"
    if not lines:
        return "circle" if len(curves) in (4, 8) else "organic"
    return "rounded_rectangle" if axis_aligned and len(curves) <= 8 and len(lines) <= 6 else "organic"


def load_layout(v1_dir: Path, layout_id: str) -> dict:
    path = v1_dir / "layouts" / f"{layout_id}.json"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(f"{V1_URL}/layouts/{layout_id}.json") as resp:
            path.write_bytes(resp.read())
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--v1-dir", default="data/lica-data-public-v1")
    args = parser.parse_args()

    root, v1_dir = Path(args.dataset_root), Path(args.v1_dir)
    with open(root / "benchmarks/category/CategoryClassification/samples.csv", encoding="utf-8") as f:
        layout_ids = [r["sample_id"] for r in csv.DictReader(f)]
    with open(root / "lica-data/metadata.csv", encoding="utf-8") as f:
        template = {r["layout_id"]: r["template_id"] for r in csv.DictReader(f)}

    rows = {"LayerOrder": [], "ImageRotation": [], "ImageCropShape": [], "FrameDetection": []}
    for lid in layout_ids:
        layout = load_layout(v1_dir, lid)
        image_path = f"lica-data/images/{template[lid]}/{lid}.png"
        meta = layout.get("layout_metadata", {})
        top = (layout.get("layout_config") or {}).get("components") or []
        if int(float(meta.get("width", 0))) > 0 and int(float(meta.get("height", 0))) > 0 and len(top) >= 2:
            rows["LayerOrder"].append([
                lid,
                image_path,
                LAYER_ORDER_PROMPT.format(element_list="\n".join(describe(c, i) for i, c in enumerate(top))),
                json.dumps([f"E{i + 1}" for i in range(len(top))]),
            ])
        images = [(c, a) for c, a in walk(top) if c.get("type") == "IMAGE"]
        raster = [i for i, (c, _) in enumerate(images) if ".svg" not in str(c.get("src", "")).lower()]
        for i in sample_indices(len(images), 3, lid):
            c, anc = images[i]
            desc = str(c.get("alt") or "")
            angle = rotation(c)
            shape = crop_shape(clip_path(c, anc))
            rows["ImageRotation"].append([
                f"{lid}_img{i}", image_path, ROTATION_PROMPT.format(description=desc),
                json.dumps({"is_rotated": angle != 0.0, "angle": angle}),
            ])
            rows["ImageCropShape"].append([
                f"{lid}_img{i}", image_path, CROP_PROMPT.format(description=desc),
                json.dumps({"is_cropped": shape != "none", "crop_shape": shape}),
            ])
        for j in sample_indices(len(raster), 3, lid):
            c, anc = images[raster[j]]
            rows["FrameDetection"].append([
                f"{lid}_img{raster[j]}", image_path, FRAME_PROMPT.format(description=str(c.get("alt") or "")),
                json.dumps({"is_framed": clip_path(c, anc) is not None}),
            ])

    for task, task_rows in rows.items():
        out_dir = root / "benchmarks" / "layout" / task
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "samples.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_id", "image_path", "prompt", "expected_output"])
            writer.writerows(task_rows)
        print(f"{task}: {len(task_rows)} samples -> {out_dir / 'samples.csv'}")


if __name__ == "__main__":
    main()
