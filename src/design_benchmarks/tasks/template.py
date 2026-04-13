"""Template benchmarks (template-1 through template-5).

All benchmarks in this module are implemented.

Data contract: each task reads ``{task-id}.json`` from the ``--data`` directory.
Template JSON files may contain a ``data_root`` key pointing to the Lica dataset
tree (layouts/, images/, annotations/) — resolved against ``dataset_root`` passed
to ``load_data`` (see ``template_layout_paths.parse_data_root``).
"""

from __future__ import annotations

import colorsys
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from design_benchmarks.base import BaseBenchmark, BenchmarkMeta, TaskType, benchmark
from design_benchmarks.utils.data_helpers import load_task_json
from design_benchmarks.utils.template_layout_paths import (
    load_layout_content,
    parse_data_root,
    resolve_layout_paths,
)
from design_benchmarks.utils.text_helpers import parse_json_from_text, strip_thinking

# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------


def _mean_reciprocal_rank(ranked: List[List[str]], relevant: List[Any]) -> float:
    total = 0.0
    for ranked_list, rel in zip(ranked, relevant):
        rel_set = set(rel) if isinstance(rel, list) else set(rel.keys()) if isinstance(rel, dict) else set()
        for i, item in enumerate(ranked_list):
            if item in rel_set:
                total += 1.0 / (i + 1)
                break
    return total / len(ranked) if ranked else 0.0


def _mean_average_precision(ranked: List[List[str]], relevant: List[Any]) -> float:
    total = 0.0
    for ranked_list, rel in zip(ranked, relevant):
        rel_set = set(rel) if isinstance(rel, list) else set(rel.keys()) if isinstance(rel, dict) else set()
        hits = 0
        ap = 0.0
        for i, item in enumerate(ranked_list):
            if item in rel_set:
                hits += 1
                ap += hits / (i + 1)
        if rel_set:
            total += ap / len(rel_set)
    return total / len(ranked) if ranked else 0.0


def _ndcg_at_k(ranked: List[List[str]], relevant: List[Any], k: int) -> float:
    total = 0.0
    for ranked_list, rel in zip(ranked, relevant):
        rel_map = rel if isinstance(rel, dict) else {r: 1 for r in rel}
        dcg = sum(
            rel_map[item] / math.log2(i + 2)
            for i, item in enumerate(ranked_list[:k])
            if item in rel_map
        )
        ideal_rels = sorted(rel_map.values(), reverse=True)[:k]
        idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal_rels))
        total += dcg / idcg if idcg > 0 else 0.0
    return total / len(ranked) if ranked else 0.0


def _recall_at_k(ranked: List[List[str]], relevant: List[Any], k: int) -> float:
    total = 0.0
    for ranked_list, rel in zip(ranked, relevant):
        rel_set = set(rel) if isinstance(rel, list) else set(rel.keys()) if isinstance(rel, dict) else set()
        if rel_set:
            total += len(set(ranked_list[:k]) & rel_set) / len(rel_set)
    return total / len(ranked) if ranked else 0.0


# ---------------------------------------------------------------------------
# Binary classification helpers (require sklearn — return 0.0 if unavailable)
# ---------------------------------------------------------------------------


def _auc_roc(scores: List[float], gt: List[int]) -> float:
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(gt, scores))
    except Exception:
        return 0.0


def _average_precision(scores: List[float], gt: List[int]) -> float:
    try:
        from sklearn.metrics import average_precision_score

        return float(average_precision_score(gt, scores))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Clustering metrics (require sklearn — return 0.0 if unavailable)
# ---------------------------------------------------------------------------


def _clustering_metrics(pred_labels: List, gt_labels: List) -> Dict[str, float]:
    try:
        from sklearn.metrics import (
            adjusted_mutual_info_score,
            adjusted_rand_score,
            fowlkes_mallows_score,
            homogeneity_completeness_v_measure,
            normalized_mutual_info_score,
        )

        h, c, v = homogeneity_completeness_v_measure(gt_labels, pred_labels)
        return {
            "ari": float(adjusted_rand_score(gt_labels, pred_labels)),
            "nmi": float(normalized_mutual_info_score(gt_labels, pred_labels)),
            "ami": float(adjusted_mutual_info_score(gt_labels, pred_labels)),
            "fowlkes_mallows": float(fowlkes_mallows_score(gt_labels, pred_labels)),
            "homogeneity": float(h),
            "completeness": float(c),
            "v_measure": float(v),
        }
    except ImportError:
        return {k: 0.0 for k in ("ari", "nmi", "ami", "fowlkes_mallows", "homogeneity", "completeness", "v_measure")}


# ---------------------------------------------------------------------------
# Template generation evaluation helpers
# ---------------------------------------------------------------------------

_COLOR_RE = re.compile(r"rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)")


def _parse_color_rgb(color_str: str) -> Optional[Tuple[float, float, float]]:
    if not color_str:
        return None
    m = _COLOR_RE.search(color_str)
    if m:
        return (float(m.group(1)) / 255.0, float(m.group(2)) / 255.0, float(m.group(3)) / 255.0)
    color_str = color_str.strip()
    if color_str.startswith("#"):
        h = color_str.lstrip("#")
        if len(h) == 6:
            try:
                return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0)
            except ValueError:
                pass
    return None


def _ciede2000_distance(rgb1: Tuple[float, ...], rgb2: Tuple[float, ...]) -> float:
    try:
        import numpy as np
        from skimage.color import deltaE_ciede2000, rgb2lab

        lab1 = rgb2lab(np.array([[rgb1]])).flatten()
        lab2 = rgb2lab(np.array([[rgb2]])).flatten()
        return float(deltaE_ciede2000(lab1.reshape(1, 3), lab2.reshape(1, 3))[0])
    except ImportError:
        dr = (rgb1[0] - rgb2[0]) * 255
        dg = (rgb1[1] - rgb2[1]) * 255
        db = (rgb1[2] - rgb2[2]) * 255
        return math.sqrt(dr * dr + dg * dg + db * db)


def _extract_all_colors(layout: Dict) -> List[str]:
    colors: list = []
    bg = layout.get("background", "")
    if bg and _parse_color_rgb(bg):
        colors.append(bg)
    for comp in layout.get("components", []):
        for prop in ("color", "backgroundColor", "background"):
            val = comp.get(prop, "")
            if val and _parse_color_rgb(val):
                colors.append(val)
    return colors


def _extract_font_families(layout: Dict) -> List[str]:
    return [c.get("fontFamily", "") for c in layout.get("components", []) if c.get("fontFamily")]


def _extract_font_sizes(layout: Dict) -> List[float]:
    sizes: list = []
    for c in layout.get("components", []):
        fs = c.get("fontSize")
        if fs is not None:
            try:
                sizes.append(float(str(fs).replace("px", "")))
            except (ValueError, TypeError):
                pass
    return sorted(sizes)


def _cosine_sim(a: List[float], b: List[float]) -> float:
    if not a and not b:
        return 1.0
    n = max(len(a), len(b))
    ap = a + [0.0] * (n - len(a))
    bp = b + [0.0] * (n - len(b))
    dot = sum(x * y for x, y in zip(ap, bp))
    na = math.sqrt(sum(x * x for x in ap))
    nb = math.sqrt(sum(x * x for x in bp))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return dot / (na * nb)


def _parse_numeric(val: Any, default: float = 0.0) -> float:
    s = str(val).strip().replace("px", "")
    if s.endswith("%"):
        s = s[:-1]
    try:
        return float(s)
    except (ValueError, TypeError):
        return default


def _extract_positions(layout: Dict) -> List[float]:
    pos: list = []
    for c in layout.get("components", []):
        pos.extend([_parse_numeric(c.get("left", "0")), _parse_numeric(c.get("top", "0"))])
    return pos


def _extract_areas(layout: Dict) -> List[float]:
    areas: list = []
    for c in layout.get("components", []):
        try:
            w = float(str(c.get("width", "0")).replace("px", ""))
            h = float(str(c.get("height", "0")).replace("px", ""))
            if w > 0 and h > 0:
                areas.append(w * h)
        except (ValueError, TypeError):
            pass
    return sorted(areas, reverse=True)


def _color_harmony_score(colors: List[str]) -> float:
    rgbs = [_parse_color_rgb(c) for c in colors]
    rgbs_clean = [c for c in rgbs if c is not None]
    if len(rgbs_clean) < 2:
        return 1.0
    hues: list = []
    for r, g, b in rgbs_clean:
        h_val, _, _ = colorsys.rgb_to_hsv(r, g, b)
        hues.append(h_val * 360)
    if not hues:
        return 1.0
    hues.sort()
    gaps = [(hues[(i + 1) % len(hues)] - hues[i]) % 360 for i in range(len(hues))]
    ideal = 360.0 / len(hues)
    var = sum((g - ideal) ** 2 for g in gaps) / len(gaps)
    max_var = (360 - ideal) ** 2
    return max(0.0, 1.0 - var / max_var) if max_var > 1e-6 else 1.0


def _contrast_ratio(rgb1: Tuple[float, ...], rgb2: Tuple[float, ...]) -> float:
    def _rl(r: float, g: float, b: float) -> float:
        def _lin(c: float) -> float:
            return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
        return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)
    l1 = _rl(*rgb1)
    l2 = _rl(*rgb2)
    return (max(l1, l2) + 0.05) / (min(l1, l2) + 0.05)


def _wcag_contrast_score(layout: Dict) -> float:
    bg_rgb = _parse_color_rgb(layout.get("background", "")) or (1.0, 1.0, 1.0)
    passes = total = 0
    for c in layout.get("components", []):
        if c.get("type") != "TEXT":
            continue
        txt_rgb = _parse_color_rgb(c.get("color", ""))
        if not txt_rgb:
            continue
        comp_bg_str = c.get("backgroundColor") or c.get("background", "")
        comp_bg = _parse_color_rgb(comp_bg_str) if comp_bg_str else bg_rgb
        total += 1
        if _contrast_ratio(txt_rgb, comp_bg) >= 4.5:
            passes += 1
    return passes / total if total > 0 else 1.0


def _typography_hierarchy_score(sizes: List[float]) -> float:
    if len(sizes) < 2:
        return 1.0
    unique = sorted(set(sizes), reverse=True)
    if len(unique) < 2:
        return 1.0
    mono = sum(1 for i in range(len(unique) - 1) if unique[i] > unique[i + 1])
    return mono / (len(unique) - 1)


# -- Dispatch for template generation evaluate ------------------------------


def _evaluate_template_generation(
    predictions: List[Any], ground_truth: List[Any], task: str,
) -> Dict[str, float]:
    accum: Dict[str, List[float]] = {}

    def _acc(name: str, value: float) -> None:
        if math.isfinite(value):
            accum.setdefault(name, []).append(value)

    for pred_raw, gt_raw in zip(predictions, ground_truth):
        pred_layouts = pred_raw if isinstance(pred_raw, list) else [pred_raw] if pred_raw else []

        gt_bundle = gt_layout = None
        if gt_raw:
            parsed_gt = gt_raw if isinstance(gt_raw, dict) else None
            if parsed_gt is None and isinstance(gt_raw, str):
                try:
                    parsed_gt = json.loads(gt_raw)
                except json.JSONDecodeError:
                    pass
            if parsed_gt:
                if task == "color_transfer" and (
                    "designated_layout" in parsed_gt or "source_layouts" in parsed_gt
                ):
                    gt_bundle = parsed_gt
                    gt_layout = parsed_gt.get("designated_layout") or (
                        parsed_gt["source_layouts"][0] if parsed_gt.get("source_layouts") else None
                    )
                else:
                    gt_layout = parsed_gt

        if not pred_layouts:
            _acc("json_valid", 0.0)
            continue

        for pred in pred_layouts:
            if not isinstance(pred, dict):
                _acc("json_valid", 0.0)
                continue
            _acc("json_valid", 1.0)
            has_components = "components" in pred and isinstance(pred.get("components"), list)

            if gt_layout and has_components:
                gt_comps = gt_layout.get("components", [])
                pred_comps = pred.get("components", [])
                _acc("component_count_match", 1.0 if len(pred_comps) == len(gt_comps) else 0.0)
                pred_fonts = set(_extract_font_families(pred))
                gt_fonts = set(_extract_font_families(gt_layout))
                if pred_fonts or gt_fonts:
                    jacc = len(pred_fonts & gt_fonts) / len(pred_fonts | gt_fonts) if (pred_fonts | gt_fonts) else 1.0
                    _acc("font_adherence", jacc)
                pred_sizes = _extract_font_sizes(pred)
                gt_sizes = _extract_font_sizes(gt_layout)
                if pred_sizes and gt_sizes:
                    _acc("font_size_cosine", _cosine_sim(pred_sizes, gt_sizes))
                if task == "color_transfer":
                    pp = _extract_positions(pred)
                    gp = _extract_positions(gt_layout)
                    if pp and gp:
                        _acc("position_fidelity", _cosine_sim(pp, gp))
                    pa = _extract_areas(pred)
                    ga = _extract_areas(gt_layout)
                    if pa and ga:
                        _acc("area_fidelity", _cosine_sim(pa, ga))

            if has_components:
                if task == "color_transfer":
                    _compute_color_transfer(pred, gt_bundle, gt_layout, _acc)
                elif task == "style_completion" and gt_layout:
                    _compute_style_completion(pred, gt_layout, _acc)
                    pred_bg = _parse_color_rgb(pred.get("background", ""))
                    gt_bg = _parse_color_rgb(gt_layout.get("background", ""))
                    if pred_bg and gt_bg:
                        _acc("bg_color_distance", _ciede2000_distance(pred_bg, gt_bg))

            if has_components:
                all_colors = _extract_all_colors(pred)
                if all_colors:
                    _acc("color_harmony_score", _color_harmony_score(all_colors))
                _acc("contrast_score", _wcag_contrast_score(pred))
                if task in ("style_completion", "color_transfer"):
                    sizes = _extract_font_sizes(pred)
                    if sizes:
                        _acc("typography_hierarchy_score", _typography_hierarchy_score(sizes))

    return {name: sum(vals) / len(vals) if vals else 0.0 for name, vals in accum.items()}


def _compute_color_transfer(
    pred: dict, gt_bundle: Optional[dict], source_layout: Optional[dict], acc: Any,
) -> None:
    if not gt_bundle:
        return
    target_palette = gt_bundle.get("target_palette", [])
    target_rgbs: set = set()
    for entry in target_palette:
        rgb = _parse_color_rgb(entry.get("color", ""))
        if rgb:
            target_rgbs.add(rgb)
    pred_colors = _extract_all_colors(pred)
    if target_rgbs and pred_colors:
        adherent = 0
        for pc_str in pred_colors:
            pc = _parse_color_rgb(pc_str)
            if pc is None:
                continue
            if min(_ciede2000_distance(pc, tp) for tp in target_rgbs) < 10.0:
                adherent += 1
        acc("color_palette_adherence", adherent / len(pred_colors))
        pred_rgbs = {_parse_color_rgb(s) for s in pred_colors}
        pred_rgbs.discard(None)
        covered = sum(
            1 for tp in target_rgbs if any(_ciede2000_distance(pp, tp) < 10.0 for pp in pred_rgbs)
        )
        acc("palette_coverage", covered / len(target_rgbs))
    if target_palette:
        tgt_bg = _parse_color_rgb(target_palette[0].get("color", ""))
        pred_bg = _parse_color_rgb(pred.get("background", ""))
        if tgt_bg and pred_bg:
            acc("bg_color_distance", _ciede2000_distance(pred_bg, tgt_bg))


def _compute_style_completion(pred: dict, gt_layout: dict, acc: Any) -> None:
    gt_comps = gt_layout.get("components", [])
    pred_comps = pred.get("components", [])
    fm = ft = cm = ct = tam = tat = 0
    fse: List[float] = []
    oe: List[float] = []
    for pc, gc in zip(pred_comps, gt_comps):
        gt_ff = gc.get("fontFamily", "")
        pred_ff = pc.get("fontFamily", "")
        if gt_ff:
            ft += 1
            if pred_ff.strip().lower() == gt_ff.strip().lower():
                fm += 1
        gt_c = gc.get("color", "")
        pred_c = pc.get("color", "")
        if gt_c:
            ct += 1
            gr = _parse_color_rgb(gt_c)
            pr = _parse_color_rgb(pred_c)
            if gr and pr and gr == pr:
                cm += 1
        gt_fs, pred_fs = gc.get("fontSize"), pc.get("fontSize")
        if gt_fs is not None and pred_fs is not None:
            try:
                fse.append(abs(float(str(pred_fs).replace("px", "")) - float(str(gt_fs).replace("px", ""))))
            except (ValueError, TypeError):
                pass
        gt_op, pred_op = gc.get("opacity"), pc.get("opacity")
        if gt_op is not None and pred_op is not None:
            try:
                oe.append(abs(float(pred_op) - float(gt_op)))
            except (ValueError, TypeError):
                pass
        gt_ta = gc.get("textAlign", "")
        if gt_ta:
            tat += 1
            if pc.get("textAlign", "") == gt_ta:
                tam += 1
    if ft:
        acc("font_exact_match", fm / ft)
    if ct:
        acc("color_exact_match", cm / ct)
    if fse:
        acc("font_size_mae", sum(fse) / len(fse))
    if oe:
        acc("opacity_mae", sum(oe) / len(oe))
    if tat:
        acc("text_align_accuracy", tam / tat)
    pred_colors = _extract_all_colors(pred)
    gt_colors = _extract_all_colors(gt_layout)
    gt_palette = {_parse_color_rgb(c) for c in gt_colors}
    gt_palette.discard(None)
    if gt_palette and pred_colors:
        adherent = sum(
            1
            for pc_s in pred_colors
            if (pc_r := _parse_color_rgb(pc_s)) is not None
            and min(_ciede2000_distance(pc_r, gp) for gp in gt_palette) < 10.0
        )
        acc("color_palette_adherence", adherent / len(pred_colors))


# ===========================================================================
# Understanding tasks — template-1, template-2, template-3
# ===========================================================================


@benchmark
class PairwiseLayoutMatching(BaseBenchmark):
    """template-1 — Determine if two layouts originate from the same template."""

    pipeline_implemented = True

    PROMPT = (
        "You are given two layouts (A and B). "
        "Determine whether they originate from the same template. "
        "Answer with a single digit: 1 if same template, 0 if different."
    )

    meta = BenchmarkMeta(
        id="template-1",
        name="Pairwise Layout Matching",
        task_type=TaskType.UNDERSTANDING,
        domain="template",
        description="Determine whether two layouts originate from the same template",
        input_spec="Two layouts",
        output_spec="Binary label (0 or 1) and optional confidence score",
        metrics=["accuracy", "precision", "recall", "f1", "auc_roc", "average_precision"],
    )

    def load_data(self, data_dir, *, n=None, dataset_root: Union[str, Path]):
        data, data_root = load_task_json(data_dir, self.meta.id)
        layout_index = data.get("layout_index", {})
        data_root_resolved = parse_data_root(data.get("data_root"), dataset_root)
        samples: list = []
        for i, pair in enumerate(data["pairs"]):
            pa = resolve_layout_paths(pair["layout_a"], layout_index, data_root_resolved)
            pb = resolve_layout_paths(pair["layout_b"], layout_index, data_root_resolved)
            samples.append({
                "sample_id": f"pair_{i:03d}",
                "ground_truth": pair["label"],
                "layout_a": pair["layout_a"],
                "layout_b": pair["layout_b"],
                "image_path_a": pa.get("image_path", ""),
                "image_path_b": pb.get("image_path", ""),
                "layout_path_a": pa.get("layout_path", ""),
                "layout_path_b": pb.get("layout_path", ""),
            })
            if n is not None and len(samples) >= n:
                break
        return samples

    def build_model_input(self, sample, *, modality=None):
        from design_benchmarks.models.base import ModelInput

        la = load_layout_content(sample["layout_a"], {"layout_path": sample.get("layout_path_a", "")})
        lb = load_layout_content(sample["layout_b"], {"layout_path": sample.get("layout_path_b", "")})
        images: list = []
        for key in ("image_path_a", "image_path_b"):
            img = sample.get(key, "")
            if img and Path(img).is_file():
                images.append(img)
        return ModelInput(text=f"{self.PROMPT}\n\nLayout A:\n{la}\n\nLayout B:\n{lb}", images=images)

    def parse_model_output(self, output):
        text = strip_thinking(output.text)
        for ch in text:
            if ch in ("0", "1"):
                return ch
        return text.strip()

    def evaluate(self, predictions, ground_truth):
        hard_preds = [str(p) for p in predictions]
        hard_gt = [str(g) for g in ground_truth]
        if not hard_gt:
            acc = 0.0
        else:
            acc = sum(1 for p, g in zip(hard_preds, hard_gt) if p == g) / len(hard_gt)
        result: Dict[str, float] = {"accuracy": acc}
        try:
            from sklearn.metrics import precision_recall_fscore_support

            p, r, f, _ = precision_recall_fscore_support(hard_gt, hard_preds, average="macro", zero_division=0)
            result.update({"precision": float(p), "recall": float(r), "f1": float(f)})
        except ImportError:
            result.update({"precision": acc, "recall": acc, "f1": acc})
        try:
            fp = [float(p) for p in predictions]
            ig = [int(g) for g in ground_truth]
            result["auc_roc"] = _auc_roc(fp, ig)
            result["average_precision"] = _average_precision(fp, ig)
        except (ValueError, TypeError):
            pass
        return result


@benchmark
class LayoutRetrieval(BaseBenchmark):
    """template-2 — Rank candidate layouts by similarity to a reference."""

    pipeline_implemented = True

    PROMPT = (
        "You are given a reference layout and a set of candidate layouts. "
        "Rank the candidates from most similar to least similar to the reference. "
        "Return the candidate IDs as a comma-separated list, most similar first."
    )

    meta = BenchmarkMeta(
        id="template-2",
        name="Layout Retrieval",
        task_type=TaskType.UNDERSTANDING,
        domain="template",
        description="Rank candidate layouts by similarity to a reference layout",
        input_spec="Reference layout + set of candidate layouts",
        output_spec="Ranked list of candidate IDs (most similar first)",
        metrics=["mrr", "map", "ndcg@5", "ndcg@10", "recall@1", "recall@5", "recall@10"],
    )

    def load_data(self, data_dir, *, n=None, dataset_root: Union[str, Path]):
        data, data_root = load_task_json(data_dir, self.meta.id)
        layout_index = data.get("layout_index", {})
        data_root_resolved = parse_data_root(data.get("data_root"), dataset_root)
        samples: list = []
        for i, q in enumerate(data["queries"]):
            ref_paths = resolve_layout_paths(q["reference"], layout_index, data_root_resolved)
            samples.append({
                "sample_id": f"retr_{i:03d}",
                "ground_truth": q.get("relevant", []),
                "reference": q["reference"],
                "reference_image_path": ref_paths.get("image_path", ""),
                "reference_layout_path": ref_paths.get("layout_path", ""),
                "candidates": q.get("candidates", []),
                "_layout_index": layout_index,
                "_data_root": data_root_resolved,
            })
            if n is not None and len(samples) >= n:
                break
        return samples

    def build_model_input(self, sample, *, modality=None):
        from design_benchmarks.models.base import ModelInput

        ref = load_layout_content(sample["reference"], {"layout_path": sample.get("reference_layout_path", "")})
        images: list = []
        ref_img = sample.get("reference_image_path", "")
        if ref_img and Path(ref_img).is_file():
            images.append(ref_img)
        layout_index = sample.get("_layout_index", {})
        data_root = sample.get("_data_root")
        cand_parts: list = []
        for cid in sample["candidates"]:
            cpaths = resolve_layout_paths(cid, layout_index, data_root)
            content = load_layout_content(cid, cpaths)
            cand_parts.append(f"[{cid}]\n{content}")
            cimg = cpaths.get("image_path", "")
            if cimg and Path(cimg).is_file():
                images.append(cimg)
        return ModelInput(
            text=f"{self.PROMPT}\n\nReference:\n{ref}\n\nCandidates:\n" + "\n\n".join(cand_parts),
            images=images,
        )

    def parse_model_output(self, output):
        text = strip_thinking(output.text.strip())
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except json.JSONDecodeError:
            pass
        tokens = [t.strip().strip('"').strip("'") for t in text.split(",")]
        return [t for t in tokens if len(t) >= 15 and t.isalnum()]

    def evaluate(self, predictions, ground_truth):
        ranked = [p if isinstance(p, list) else [] for p in predictions]
        return {
            "mrr": _mean_reciprocal_rank(ranked, ground_truth),
            "map": _mean_average_precision(ranked, ground_truth),
            "ndcg@5": _ndcg_at_k(ranked, ground_truth, 5),
            "ndcg@10": _ndcg_at_k(ranked, ground_truth, 10),
            "recall@1": _recall_at_k(ranked, ground_truth, 1),
            "recall@5": _recall_at_k(ranked, ground_truth, 5),
            "recall@10": _recall_at_k(ranked, ground_truth, 10),
        }


@benchmark
class LayoutClustering(BaseBenchmark):
    """template-3 — Group layouts into clusters by underlying template."""

    pipeline_implemented = True

    PROMPT = (
        "You are given a collection of design layouts. "
        "Each layout was created from a template. Multiple layouts may share "
        "the same template. Your task is to group the layouts by their "
        "underlying template — assign the same integer label to layouts "
        "that come from the same template."
    )

    OUTPUT_INSTRUCTION = (
        "Assign the same integer label to layouts from the same template. "
        "Return ONLY a comma-separated list of integer labels, one per layout, "
        "in the order the layouts were presented. "
        "Example: 0,0,1,1,2  means layouts 1&2 share a template, 3&4 share another, "
        "and 5 is from a third template."
    )

    meta = BenchmarkMeta(
        id="template-3",
        name="Layout Clustering",
        task_type=TaskType.UNDERSTANDING,
        domain="template",
        description="Group layouts into clusters that correspond to underlying templates",
        input_spec="Collection of layouts",
        output_spec="Cluster assignment for each layout (integer label)",
        metrics=["ari", "nmi", "ami", "homogeneity", "completeness", "v_measure", "fowlkes_mallows"],
    )

    def load_data(self, data_dir, *, n=None, dataset_root: Union[str, Path]):
        data, data_root = load_task_json(data_dir, self.meta.id)
        problems = data.get("problems", data.get("layouts"))
        if isinstance(problems, dict):
            problems = [data]
        layout_index = data.get("layout_index", {})
        data_root_resolved = parse_data_root(data.get("data_root"), dataset_root)
        samples: list = []
        for i, prob in enumerate(problems):
            layouts = prob.get("layouts", [])
            labels = prob.get("cluster_labels", [])
            n_clusters = prob.get("n_clusters", len(set(labels)) if labels else 0)
            image_paths: list = []
            for lid in layouts:
                p = resolve_layout_paths(lid, layout_index, data_root_resolved)
                image_paths.append(p.get("image_path", ""))
            samples.append({
                "sample_id": f"clust_{i:03d}",
                "ground_truth": labels,
                "layouts": layouts,
                "n_clusters": n_clusters,
                "image_paths": image_paths,
                "_layout_index": layout_index,
                "_data_root": data_root_resolved,
            })
            if n is not None and len(samples) >= n:
                break
        return samples

    def build_model_input(self, sample, *, modality=None):
        from design_benchmarks.models.base import ModelInput

        layout_index = sample.get("_layout_index", {})
        data_root = sample.get("_data_root")
        parts: list = []
        for lid in sample["layouts"]:
            lp = resolve_layout_paths(lid, layout_index, data_root)
            content = load_layout_content(lid, lp)
            parts.append(f"[{lid}]\n{content}")
        images = [ip for ip in sample.get("image_paths", []) if ip and Path(ip).is_file()]
        n_clusters = sample.get("n_clusters", "unknown")
        text = (
            f"{self.PROMPT}\n\nThere are {n_clusters} templates.\n\n"
            + "\n\n".join(parts) + f"\n\n{self.OUTPUT_INSTRUCTION}"
        )
        return ModelInput(text=text, images=images)

    def parse_model_output(self, output):
        text = strip_thinking(output.text.strip())
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [int(x) for x in parsed]
        except (json.JSONDecodeError, ValueError):
            pass
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [s.strip() for s in line.split(",") if s.strip()]
            try:
                return [int(p) for p in parts]
            except ValueError:
                continue
        m = re.search(r"(?:\d+\s*,\s*)+\d+", text)
        if m:
            try:
                return [int(x.strip()) for x in m.group().split(",")]
            except ValueError:
                pass
        return []

    def evaluate(self, predictions, ground_truth):
        keys = ["ari", "nmi", "ami", "fowlkes_mallows", "homogeneity", "completeness", "v_measure"]
        sums: Dict[str, float] = {k: 0.0 for k in keys}
        total = len(ground_truth)
        for pred_labels, gt_labels in zip(predictions, ground_truth):
            if not gt_labels or not pred_labels:
                continue
            min_len = min(len(pred_labels), len(gt_labels))
            pl, gl = pred_labels[:min_len], gt_labels[:min_len]
            metrics = _clustering_metrics(pl, gl)
            for k in keys:
                sums[k] += metrics.get(k, 0.0)
        return {k: v / total for k, v in sums.items()} if total else {k: 0.0 for k in keys}


# ===========================================================================
# Generation tasks — template-4, template-5
# ===========================================================================


@benchmark
class StyleCompletion(BaseBenchmark):
    """template-4 — Complete a skeleton layout with visual style properties."""

    pipeline_implemented = True

    PROMPT = (
        "You are a design system assistant. Given several example layouts "
        "from the same template as style context, and one skeleton layout "
        "with structural data but missing visual styles, complete the skeleton "
        "by filling in the missing style properties (fontFamily, fontSize, color, "
        "backgroundColor, opacity, textAlign) to match the template's style. "
        "Return ONLY a valid JSON layout object."
    )

    meta = BenchmarkMeta(
        id="template-4",
        name="Style Completion",
        task_type=TaskType.GENERATION,
        domain="template",
        description="Complete a skeleton layout with visual style properties to match a template",
        input_spec="Style-context sibling layouts + skeleton layout",
        output_spec="Styled JSON layout object",
        metrics=["json_valid", "font_exact_match", "color_exact_match", "font_size_mae", "text_align_accuracy"],
    )

    def load_data(self, data_dir, *, n=None, dataset_root: Union[str, Path]):
        data, data_root = load_task_json(data_dir, self.meta.id)
        data_root_resolved = parse_data_root(data.get("data_root"), dataset_root)
        samples: list = []
        for i, prob in enumerate(data.get("problems", [])):
            ctx_img: list = []
            if data_root_resolved:
                ctx_img = [str(data_root_resolved / p) for p in prob.get("context_image_paths", [])]
            samples.append({
                "sample_id": f"style_{i:03d}",
                "ground_truth": prob.get("ground_truth", {}),
                "context_layouts": prob.get("context_layouts", []),
                "skeleton": prob.get("skeleton", {}),
                "image_srcs": prob.get("image_srcs", []),
                "context_image_paths": ctx_img,
            })
            if n is not None and len(samples) >= n:
                break
        return samples

    def build_model_input(self, sample, *, modality=None):
        from design_benchmarks.models.base import ModelInput

        ctx = json.dumps(sample["context_layouts"], indent=2)
        skel = json.dumps(sample["skeleton"], indent=2)
        images = [ip for ip in sample.get("context_image_paths", []) if ip and Path(ip).is_file()]
        img_note = ""
        if sample.get("image_srcs"):
            img_note = "\n\nThe skeleton references these assets (URLs, use as-is):\n" + json.dumps(sample["image_srcs"])
        text = f"{self.PROMPT}\n\nStyle context (sibling layouts):\n{ctx}\n\nSkeleton to style:\n{skel}{img_note}"
        return ModelInput(text=text, images=images)

    def parse_model_output(self, output):
        parsed = parse_json_from_text(output.text)
        if isinstance(parsed, list) and parsed:
            return parsed[0] if isinstance(parsed[0], dict) else {}
        if isinstance(parsed, dict):
            return parsed
        return {}

    def evaluate(self, predictions, ground_truth):
        return _evaluate_template_generation(predictions, ground_truth, "style_completion")


@benchmark
class ColorTransfer(BaseBenchmark):
    """template-5 — Recolor a layout to use a target color palette."""

    pipeline_implemented = True

    PROMPT = (
        "You are a design system assistant. Given a layout and a target color palette, "
        "recolor the layout to use the target palette while preserving the layout structure "
        "(positions, sizes, fonts, text content). "
        "Return ONLY a valid JSON layout object."
    )

    meta = BenchmarkMeta(
        id="template-5",
        name="Color Transfer",
        task_type=TaskType.GENERATION,
        domain="template",
        description="Recolor a layout to use a target palette while preserving structure",
        input_spec="Layout + target color palette + optional color mapping hint",
        output_spec="Recolored JSON layout object",
        metrics=["json_valid", "color_palette_adherence", "palette_coverage", "position_fidelity", "area_fidelity"],
    )

    def load_data(self, data_dir, *, n=None, dataset_root: Union[str, Path]):
        data, data_root = load_task_json(data_dir, self.meta.id)
        data_root_resolved = parse_data_root(data.get("data_root"), dataset_root)
        samples: list = []
        for i, prob in enumerate(data.get("problems", [])):
            src_img: list = []
            if data_root_resolved:
                src_img = [str(data_root_resolved / p) for p in prob.get("source_image_paths", [])]
            else:
                src_img = prob.get("source_image_paths", [])
            samples.append({
                "sample_id": f"color_{i:03d}",
                "ground_truth": prob,
                "designated_layout": prob.get("designated_layout", {}),
                "context_layouts": prob.get("context_layouts", []),
                "source_palette": prob.get("source_palette", []),
                "target_palette": prob.get("target_palette", []),
                "color_mapping": prob.get("color_mapping"),
                "difficulty": prob.get("difficulty", ""),
                "source_image_paths": src_img,
            })
            if n is not None and len(samples) >= n:
                break
        return samples

    def build_model_input(self, sample, *, modality=None):
        from design_benchmarks.models.base import ModelInput

        designated = json.dumps(sample["designated_layout"], indent=2)
        ctx = json.dumps(sample["context_layouts"], indent=2)
        target = json.dumps(sample["target_palette"], indent=2)
        images = [ip for ip in sample.get("source_image_paths", []) if ip and Path(ip).is_file()]
        mapping_note = ""
        if sample.get("color_mapping") and sample.get("difficulty") == "easy":
            mapping_note = f"\n\nColor mapping hint:\n{json.dumps(sample['color_mapping'], indent=2)}"
        text = (
            f"{self.PROMPT}{mapping_note}\n\n"
            f"Target palette:\n{target}\n\n"
            f"Siblings (style context):\n{ctx}\n\n"
            f"DESIGNATED layout to recolor:\n{designated}"
        )
        return ModelInput(text=text, images=images)

    def parse_model_output(self, output):
        parsed = parse_json_from_text(output.text)
        if isinstance(parsed, list) and parsed:
            return parsed[0] if isinstance(parsed[0], dict) else {}
        if isinstance(parsed, dict):
            return parsed
        return {}

    def evaluate(self, predictions, ground_truth):
        return _evaluate_template_generation(predictions, ground_truth, "color_transfer")
