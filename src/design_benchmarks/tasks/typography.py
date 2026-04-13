"""Typography benchmarks: typography-1 through typography-8.

Data contract for these tasks: ``samples.csv`` in the ``--data`` directory
with columns ``sample_id``, ``prompt``, ``image_path``, ``expected_output``.
``image_path`` is resolved against ``dataset_root`` (the bundle root passed to
``load_data``, e.g. ``data/lica-benchmarks-dataset``), not only against ``data_dir``.
"""

from __future__ import annotations

import base64
import colorsys
import csv
import hashlib
import html
import io
import json
import logging
import math
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from design_benchmarks.base import BaseBenchmark, BenchmarkMeta, TaskType, benchmark
from design_benchmarks.metrics.core import edit_distance, lpips_score
from design_benchmarks.metrics.core import ssim as ssim_metric
from design_benchmarks.metrics.text import normalize_font_name
from design_benchmarks.utils.data_helpers import build_vision_input, load_csv_samples
from design_benchmarks.utils.text_helpers import extract_json_obj

logger = logging.getLogger(__name__)
Box = Tuple[int, int, int, int]

# ---------------------------------------------------------------------------
# Shared helpers — color metrics
# ---------------------------------------------------------------------------


def _parse_color_string(text: str) -> Optional[Tuple[int, int, int]]:
    text = text.strip().lower()
    m = re.match(r"#([0-9a-f]{6})$", text)
    if m:
        h = m.group(1)
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    m = re.match(r"#([0-9a-f]{3})$", text)
    if m:
        h = m.group(1)
        return int(h[0] * 2, 16), int(h[1] * 2, 16), int(h[2] * 2, 16)
    m = re.match(r"rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", text)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return None


def _rgb_to_lab(r: int, g: int, b: int) -> Tuple[float, float, float]:
    def linearize(c: float) -> float:
        c /= 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    rl, gl, bl = linearize(float(r)), linearize(float(g)), linearize(float(b))
    x = rl * 0.4124564 + gl * 0.3575761 + bl * 0.1804375
    y = rl * 0.2126729 + gl * 0.7151522 + bl * 0.0721750
    z = rl * 0.0193339 + gl * 0.1191920 + bl * 0.9503041
    xn, yn, zn = 0.95047, 1.0, 1.08883

    def f(t: float) -> float:
        return t ** (1.0 / 3.0) if t > 0.008856 else 7.787 * t + 16.0 / 116.0

    fx, fy, fz = f(x / xn), f(y / yn), f(z / zn)
    return 116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)


def _color_distance(pred: str, gt: str) -> Dict[str, float]:
    pred_rgb = _parse_color_string(str(pred))
    gt_rgb = _parse_color_string(str(gt))
    if pred_rgb is None or gt_rgb is None:
        return {"rgb_l2_distance": float("inf"), "delta_e_distance": float("inf")}
    rgb_l2 = math.sqrt(sum((float(p) - float(g)) ** 2 for p, g in zip(pred_rgb, gt_rgb)))
    pred_lab = _rgb_to_lab(*pred_rgb)
    gt_lab = _rgb_to_lab(*gt_rgb)
    delta_e = math.sqrt(sum((p - g) ** 2 for p, g in zip(pred_lab, gt_lab)))
    return {"rgb_l2_distance": rgb_l2, "delta_e_distance": delta_e}


def _hue_bucket(r: int, g: int, b: int) -> str:
    if max(r, g, b) - min(r, g, b) < 30:
        if max(r, g, b) < 40:
            return "black"
        if min(r, g, b) > 215:
            return "white"
        return "gray"
    h, _s, _v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    if _s < 0.15:
        return "gray"
    h_deg = h * 360.0
    if h_deg < 15 or h_deg >= 345:
        return "red"
    if h_deg < 45:
        return "orange"
    if h_deg < 75:
        return "yellow"
    if h_deg < 165:
        return "green"
    if h_deg < 195:
        return "cyan"
    if h_deg < 255:
        return "blue"
    if h_deg < 285:
        return "purple"
    return "pink"


def _hue_bucket_accuracy(predictions: List[str], ground_truth: List[str]) -> float:
    correct = total = 0
    for pred, gt in zip(predictions, ground_truth):
        pred_rgb = _parse_color_string(str(pred))
        gt_rgb = _parse_color_string(str(gt))
        if pred_rgb is None or gt_rgb is None:
            continue
        total += 1
        if _hue_bucket(*pred_rgb) == _hue_bucket(*gt_rgb):
            correct += 1
    return correct / total if total > 0 else 0.0


def _normalize_font_name(raw: str) -> str:
    name = raw.strip().lower()
    name = re.sub(r"--\d+$", "", name)
    for suffix in ("-regular", " regular", "-bold", " bold"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name.strip()


# ===========================================================================
# typography-1 through typography-6
# ===========================================================================


@benchmark
class FontFamilyClassification(BaseBenchmark):
    """typography-1 — Identify the font family used in a text component."""

    pipeline_implemented = True

    meta = BenchmarkMeta(
        id="typography-1",
        name="Font Family Classification",
        task_type=TaskType.UNDERSTANDING,
        domain="typography",
        data_subpath="typography/FontFamilyClassification",
        description="Identify the font family used in text components",
        input_spec="Rendered text component (image)",
        output_spec="Font family name",
        metrics=["accuracy_top1", "f1_macro"],
    )

    def load_data(self, data_dir, *, n=None, dataset_root: Union[str, Path]):
        return load_csv_samples(data_dir, n=n, dataset_root=dataset_root)

    def build_model_input(self, sample, *, modality=None):
        return build_vision_input(sample, modality=modality)

    def parse_model_output(self, output):
        t = output.text.strip().strip('"').strip("'")
        if "," in t:
            t = t.split(",")[0].strip().strip('"').strip("'")
        return t

    def evaluate(self, predictions, ground_truth):
        norm_preds = [_normalize_font_name(str(p)) for p in predictions]
        norm_gts = [_normalize_font_name(str(g)) for g in ground_truth]
        n = max(len(predictions), 1)
        top1 = sum(1 for p, g in zip(norm_preds, norm_gts) if p == g)
        result: Dict[str, float] = {"accuracy_top1": top1 / n}
        families = sorted(set(norm_gts))
        per_f1: List[float] = []
        for fam in families:
            tp = sum(1 for p, g in zip(norm_preds, norm_gts) if p == fam and g == fam)
            fp = sum(1 for p, g in zip(norm_preds, norm_gts) if p == fam and g != fam)
            fn = sum(1 for p, g in zip(norm_preds, norm_gts) if p != fam and g == fam)
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            result[f"f1_{fam}"] = f1
            per_f1.append(f1)
        result["f1_macro"] = sum(per_f1) / max(len(per_f1), 1)
        return result


@benchmark
class TextColorEstimation(BaseBenchmark):
    """typography-2 — Predict the color of text components."""

    pipeline_implemented = True

    meta = BenchmarkMeta(
        id="typography-2",
        name="Text Color Estimation",
        task_type=TaskType.UNDERSTANDING,
        domain="typography",
        data_subpath="typography/TextColorEstimation",
        description="Predict the color of text components",
        input_spec="Rendered text component (image)",
        output_spec="Color as hex or rgb()",
        metrics=["rgb_l2_distance", "delta_e_distance", "delta_e_below_5", "hue_bucket_accuracy"],
    )

    def load_data(self, data_dir, *, n=None, dataset_root: Union[str, Path]):
        return load_csv_samples(data_dir, n=n, dataset_root=dataset_root)

    def build_model_input(self, sample, *, modality=None):
        return build_vision_input(sample, modality=modality)

    def parse_model_output(self, output):
        text = output.text.strip()
        m = re.search(r"#[0-9a-fA-F]{3,6}", text)
        if m:
            return m.group(0).upper()
        m = re.search(r"rgb\s*\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\)", text, re.IGNORECASE)
        if m:
            return m.group(0)
        return text

    def evaluate(self, predictions, ground_truth):
        rgb_l2_scores: List[float] = []
        de_scores: List[float] = []
        for pred, gt in zip(predictions, ground_truth):
            d = _color_distance(str(pred), str(gt))
            if d["rgb_l2_distance"] != float("inf"):
                rgb_l2_scores.append(d["rgb_l2_distance"])
                de_scores.append(d["delta_e_distance"])
        n = max(len(rgb_l2_scores), 1)
        result: Dict[str, float] = {
            "rgb_l2_distance": sum(rgb_l2_scores) / n,
            "delta_e_distance": sum(de_scores) / n,
            "delta_e_below_5": sum(1 for d in de_scores if d < 5.0) / n,
        }
        result["hue_bucket_accuracy"] = _hue_bucket_accuracy(
            [str(p) for p in predictions], [str(g) for g in ground_truth],
        )
        return result


@benchmark
class TextParamsEstimation(BaseBenchmark):
    """typography-3 — Extract font size, weight, alignment, letter spacing, and line height."""

    pipeline_implemented = True

    PROPERTIES = ["font_size", "font_weight", "text_align", "letter_spacing", "line_height"]
    CLASSIFICATION_PROPS = {"text_align", "font_weight"}

    meta = BenchmarkMeta(
        id="typography-3",
        name="Text Params Estimation",
        task_type=TaskType.UNDERSTANDING,
        domain="typography",
        data_subpath="typography/TextParamsEstimation",
        description=(
            "Extract font_size, font_weight, text_align, letter_spacing, and line_height "
            "as one JSON object"
        ),
        input_spec="Rendered text component (image)",
        output_spec=(
            "JSON object with keys font_size, font_weight, text_align, letter_spacing, line_height"
        ),
        metrics=[
            "font_size_mae",
            "font_weight_accuracy",
            "text_align_accuracy",
            "letter_spacing_mae",
            "line_height_mae",
        ],
    )

    def load_data(self, data_dir, *, n=None, dataset_root: Union[str, Path]):
        return load_csv_samples(data_dir, n=n, dataset_root=dataset_root)

    def build_model_input(self, sample, *, modality=None):
        return build_vision_input(sample, modality=modality)

    def parse_model_output(self, output):
        parsed = extract_json_obj(output.text.strip())
        return parsed if isinstance(parsed, dict) else {}

    def evaluate(self, predictions, ground_truth):
        result: Dict[str, float] = {}
        for prop in self.PROPERTIES:
            preds_v = [(p.get(prop) if isinstance(p, dict) else None) for p in predictions]
            gts_v = [(g.get(prop) if isinstance(g, dict) else None) for g in ground_truth]
            valid = [(p, g) for p, g in zip(preds_v, gts_v) if p is not None and g is not None]
            if not valid:
                continue
            if prop in self.CLASSIFICATION_PROPS:
                correct = sum(1 for p, g in valid if str(p).lower() == str(g).lower())
                result[f"{prop}_accuracy"] = correct / len(valid)
            else:
                errors = [abs(float(p) - float(g)) for p, g in valid]
                result[f"{prop}_mae"] = sum(errors) / len(errors)
        return result


@benchmark
class StyleRanges(BaseBenchmark):
    """typography-4 — Detect inline style ranges within a text block."""

    pipeline_implemented = True

    meta = BenchmarkMeta(
        id="typography-4",
        name="Style Ranges",
        task_type=TaskType.UNDERSTANDING,
        domain="typography",
        data_subpath="typography/StyleRanges",
        description="Detect inline style ranges within a text block",
        input_spec="Rendered text component (image)",
        output_spec="JSON array of spans with start, end, and style fields",
        metrics=["span_iou", "exact_match"],
    )

    def load_data(self, data_dir, *, n=None, dataset_root: Union[str, Path]):
        return load_csv_samples(data_dir, n=n, dataset_root=dataset_root)

    def build_model_input(self, sample, *, modality=None):
        return build_vision_input(sample, modality=modality)

    def parse_model_output(self, output):
        parsed = extract_json_obj(output.text.strip())
        if not isinstance(parsed, list):
            return []
        return [r for r in parsed if isinstance(r, dict) and "start" in r and "end" in r]

    def evaluate(self, predictions, ground_truth):
        iou_scores: List[float] = []
        exact = total = 0
        for pred_ranges, gt_ranges in zip(predictions, ground_truth):
            if not isinstance(pred_ranges, list) or not isinstance(gt_ranges, list):
                continue
            total += 1
            p_chars: set = set()
            for r in pred_ranges:
                p_chars.update(range(int(r.get("start", 0)), int(r.get("end", 0))))
            g_chars: set = set()
            for r in gt_ranges:
                g_chars.update(range(int(r.get("start", 0)), int(r.get("end", 0))))
            if not p_chars and not g_chars:
                iou_val = 1.0
            elif not p_chars or not g_chars:
                iou_val = 0.0
            else:
                iou_val = len(p_chars & g_chars) / len(p_chars | g_chars)
            iou_scores.append(iou_val)

            if iou_val >= 0.99 and self._styles_match(pred_ranges, gt_ranges):
                exact += 1

        n = max(total, 1)
        return {
            "span_iou": sum(iou_scores) / max(len(iou_scores), 1),
            "exact_match": exact / n,
        }

    @staticmethod
    def _styles_match(pr: list, gr: list) -> bool:
        if len(pr) != len(gr):
            return False
        for a, b in zip(
            sorted(pr, key=lambda x: x.get("start", 0)),
            sorted(gr, key=lambda x: x.get("start", 0)),
        ):
            for k in ("font_family", "font_weight", "font_size", "color"):
                if k in b and str(a.get(k, "")).lower() != str(b[k]).lower():
                    return False
        return True


@benchmark
class CurvedText(BaseBenchmark):
    """typography-5 — Detect and characterize curved or warped text."""

    pipeline_implemented = True

    meta = BenchmarkMeta(
        id="typography-5",
        name="Curved Text",
        task_type=TaskType.UNDERSTANDING,
        domain="typography",
        data_subpath="typography/CurvedText",
        description="Detect and characterize curved or warped text",
        input_spec="Rendered text component (image)",
        output_spec="JSON with is_curved and curvature",
        metrics=["is_curved_accuracy", "curvature_mae"],
    )

    def load_data(self, data_dir, *, n=None, dataset_root: Union[str, Path]):
        return load_csv_samples(data_dir, n=n, dataset_root=dataset_root)

    def build_model_input(self, sample, *, modality=None):
        return build_vision_input(sample, modality=modality)

    def parse_model_output(self, output):
        parsed = extract_json_obj(output.text.strip())
        if isinstance(parsed, dict):
            return parsed
        return {"is_curved": False, "curvature": 0}

    def evaluate(self, predictions, ground_truth):
        curved_correct = 0
        curvature_errors: List[float] = []
        curved_only_errors: List[float] = []
        curved_only_correct = 0
        curved_count = 0
        straight_only_correct = 0
        straight_count = 0
        total = 0
        for pred, gt in zip(predictions, ground_truth):
            if not isinstance(pred, dict) or not isinstance(gt, dict):
                continue
            total += 1
            pred_curved = bool(pred.get("is_curved"))
            gt_curved = bool(gt.get("is_curved"))
            if pred_curved == gt_curved:
                curved_correct += 1
            pred_curv = float(pred.get("curvature", 0))
            gt_curv = float(gt.get("curvature", 0))
            curvature_errors.append(abs(pred_curv - gt_curv))
            if gt_curved:
                curved_count += 1
                curved_only_errors.append(abs(pred_curv - gt_curv))
                if pred_curved == gt_curved:
                    curved_only_correct += 1
            else:
                straight_count += 1
                if pred_curved == gt_curved:
                    straight_only_correct += 1
        n = max(total, 1)
        return {
            "is_curved_accuracy": curved_correct / n,
            "curvature_mae": sum(curvature_errors) / max(len(curvature_errors), 1),
            "curvature_mae_curved_only": (
                sum(curved_only_errors) / max(len(curved_only_errors), 1)
                if curved_only_errors
                else 0.0
            ),
            "curved/count": float(curved_count),
            "curved/is_curved_accuracy": (
                curved_only_correct / curved_count if curved_count > 0 else 0.0
            ),
            "curved/curvature_mae": (
                sum(curved_only_errors) / curved_count if curved_count > 0 else 0.0
            ),
            "straight/count": float(straight_count),
            "straight/is_curved_accuracy": (
                straight_only_correct / straight_count if straight_count > 0 else 0.0
            ),
        }


@benchmark
class TextRotation(BaseBenchmark):
    """typography-6 — Detect text rotation and estimate angle."""

    pipeline_implemented = True

    meta = BenchmarkMeta(
        id="typography-6",
        name="Text Rotation",
        task_type=TaskType.UNDERSTANDING,
        domain="typography",
        data_subpath="typography/TextRotation",
        description="Detect text rotation and estimate angle",
        input_spec="Rendered text component (image)",
        output_spec="JSON with is_rotated and angle",
        metrics=["is_rotated_accuracy", "angle_mae"],
    )

    def load_data(self, data_dir, *, n=None, dataset_root: Union[str, Path]):
        return load_csv_samples(data_dir, n=n, dataset_root=dataset_root)

    def build_model_input(self, sample, *, modality=None):
        return build_vision_input(sample, modality=modality)

    def parse_model_output(self, output):
        parsed = extract_json_obj(output.text.strip())
        if isinstance(parsed, dict):
            return parsed
        return {"is_rotated": False, "angle": 0}

    def evaluate(self, predictions, ground_truth):
        rot_correct = 0
        angle_errors: List[float] = []
        rotated_only_errors: List[float] = []
        total = 0
        for pred, gt in zip(predictions, ground_truth):
            if not isinstance(pred, dict) or not isinstance(gt, dict):
                continue
            total += 1
            pred_rotated = bool(pred.get("is_rotated"))
            gt_rotated = bool(gt.get("is_rotated"))
            if pred_rotated == gt_rotated:
                rot_correct += 1
            err = abs(float(pred.get("angle", 0)) - float(gt.get("angle", 0)))
            angle_errors.append(err)
            if gt_rotated:
                rotated_only_errors.append(err)
        n = max(total, 1)
        return {
            "is_rotated_accuracy": rot_correct / n,
            "angle_mae": sum(angle_errors) / max(len(angle_errors), 1),
            "angle_mae_rotated_only": (
                sum(rotated_only_errors) / max(len(rotated_only_errors), 1)
                if rotated_only_errors
                else 0.0
            ),
        }


@benchmark
class StyledTextGeneration(BaseBenchmark):
    """typography-8 — G10 styled text element generation from text and style only."""

    pipeline_implemented = True

    meta = BenchmarkMeta(
        id="typography-8",
        name="G10 Styled Text Element Generation",
        task_type=TaskType.GENERATION,
        domain="typography",
        data_subpath="typography/Typography-6-Styled-Text-Generation",
        description="Generate a styled text element from text and typography specification",
        input_spec="Text string + typography specification (no image input)",
        output_spec="Generated image containing target styled text",
        metrics=[
            "ocr_accuracy",
            "cer",
            "edit_distance",
            "ocr_accuracy_alnum",
            "cer_alnum",
            "edit_distance_alnum",
            "bbox_iou",
            "bbox_f1",
            "bbox_precision",
            "bbox_recall",
            "bbox_detection_rate",
            "font_family_top1_accuracy",
            "font_family_top5_accuracy",
            "font_size_mae",
            "color_mae_rgb",
            "text_align_accuracy",
            "curvature_accuracy",
            "spacing_line_height_mae",
            "line_height_mae",
            "letter_spacing_mae",
            "property_accuracy",
            "property_mae",
            "property_coverage",
            "style_prediction_rate",
        ],
    )

    _text_param_predictor_bundle: Any = None
    USE_MANIFEST_PROMPT_ENV = "DESIGN_BENCHMARKS_G10_USE_MANIFEST_PROMPT"
    TEXTPARAM_API_URL_ENV = "DESIGN_BENCHMARKS_G10_TEXTPARAM_API_URL"
    TEXTPARAM_API_KEY_ENV = "DESIGN_BENCHMARKS_G10_TEXTPARAM_API_KEY"
    TEXTPARAM_API_TIMEOUT_ENV = "DESIGN_BENCHMARKS_G10_TEXTPARAM_API_TIMEOUT_SECONDS"
    TEXTPARAM_API_MODEL_ID_ENV = "DESIGN_BENCHMARKS_G10_TEXTPARAM_API_MODEL_ID"
    TEXTPARAM_API_GCS_BUCKET_ENV = "DESIGN_BENCHMARKS_G10_TEXTPARAM_GCS_BUCKET"
    TEXTPARAM_API_GCS_PREFIX_ENV = "DESIGN_BENCHMARKS_G10_TEXTPARAM_GCS_PREFIX"
    TEXTPARAM_API_GCS_CREDENTIALS_ENV = "DESIGN_BENCHMARKS_G10_TEXTPARAM_GCS_CREDENTIALS"
    ELEMENT_MANIFEST_JSON = "g10_text_element_manifest.json"
    ELEMENT_MANIFEST_CSV = "g10_text_element_manifest.csv"
    INPAINT_MANIFEST_JSON = "g10_text_inpaint_manifest.json"
    INPAINT_MANIFEST_CSV = "g10_text_inpaint_manifest.csv"
    BBOX_DETECTOR_MODEL_ID_ENV = "DESIGN_BENCHMARKS_G10_BBOX_DETECTOR_MODEL_ID"
    BBOX_DETECTOR_DEFAULT_MODEL_ID = "gpt-5.4"
    _bbox_detector_model: Any = None
    _bbox_detector_model_id: str = ""
    _bbox_detector_disabled: bool = False
    _bbox_detector_cache: Dict[str, Optional[Box]] = {}

    def load_data(
        self,
        data_dir: Union[str, Path],
        *,
        n: Optional[int] = None,
        dataset_root: Union[str, Path],
    ) -> List[Dict[str, Any]]:
        manifest_path = self._resolve_g10_manifest_path(
            data_dir=data_dir,
            json_name=self.ELEMENT_MANIFEST_JSON,
            csv_name=self.ELEMENT_MANIFEST_CSV,
            missing_message="G10 element manifest not found",
        )
        rows = self._read_g10_manifest_rows(manifest_path)

        root = manifest_path.parent
        samples: List[Dict[str, Any]] = []
        for i, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            text = self._clean_text(row.get("text"))
            if not text:
                continue

            gt_path = self._resolve_manifest_path(root, row.get("ground_truth_image"))
            input_path = self._resolve_manifest_path(root, row.get("input_image"))
            mask_path = self._resolve_manifest_path(root, row.get("mask"))
            context_path = self._resolve_manifest_path(root, row.get("layout_context_image"))
            if not gt_path or not mask_path:
                continue

            style_spec = row.get("style_spec") if isinstance(row.get("style_spec"), dict) else {}
            bbox_xywh = self._normalize_bbox_xywh(row.get("bbox_xywh_on_layout"))
            sample_id = str(row.get("sample_id") or f"g10_element_{i:04d}")
            row_prompt = str(row.get("prompt") or "").strip()
            if self._resolve_use_manifest_prompt() and row_prompt:
                prompt = row_prompt
            else:
                prompt = self._compose_element_prompt(text=text, style_spec=style_spec)

            gt_bundle = {
                "sample_id": sample_id,
                "text": text,
                "style_spec": style_spec,
                "ground_truth_image": gt_path,
                "mask": mask_path,
                "input_image": input_path,
                "evaluation_mode": "text_style_only",
                "target_bbox_xywh_on_layout": bbox_xywh,
            }
            samples.append(
                {
                    "sample_id": sample_id,
                    "text": text,
                    "style_spec": style_spec,
                    "prompt": prompt,
                    "input_image": input_path,
                    "layout_context_image": context_path,
                    "mask": mask_path,
                    "ground_truth_image": gt_path,
                    "ground_truth": gt_bundle,
                }
            )
            if n is not None and len(samples) >= n:
                break

        if not samples:
            raise ValueError(f"No valid samples in G10 element manifest: {manifest_path}")
        return samples

    @classmethod
    def _resolve_g10_manifest_path(
        cls,
        *,
        data_dir: Union[str, Path],
        json_name: str,
        csv_name: str,
        missing_message: str,
    ) -> Path:
        manifest_path = Path(data_dir).resolve()
        if manifest_path.is_file():
            return manifest_path
        if manifest_path.is_dir():
            json_path = manifest_path / json_name
            csv_path = manifest_path / csv_name
            if json_path.is_file():
                return json_path
            if csv_path.is_file():
                return csv_path
            raise FileNotFoundError(f"{missing_message}: {json_path} or {csv_path}")
        raise FileNotFoundError(f"{missing_message}: {manifest_path}")

    @classmethod
    def _read_g10_manifest_rows(cls, manifest_path: Path) -> List[Dict[str, Any]]:
        suffix = manifest_path.suffix.lower()
        if suffix == ".csv":
            try:
                with open(manifest_path, "r", encoding="utf-8-sig", newline="") as f:
                    reader = csv.DictReader(f)
                    if not reader.fieldnames:
                        raise ValueError(f"CSV manifest has no header row: {manifest_path}")
                    rows = [
                        cls._normalize_g10_manifest_csv_row(row)
                        for row in reader
                        if isinstance(row, dict)
                    ]
                    return rows
            except Exception as exc:
                raise ValueError(f"Failed to parse CSV manifest {manifest_path}: {exc}") from exc

        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = payload.get("samples", []) if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError(f"Manifest must be a list (or dict with samples): {manifest_path}")
        return [row for row in rows if isinstance(row, dict)]

    @classmethod
    def _normalize_g10_manifest_csv_row(cls, row: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(row)
        for key in (
            "sample_id",
            "source_layout_id",
            "text",
            "input_image",
            "ground_truth_image",
            "mask",
            "layout_context_image",
            "prompt",
            "mask_method",
        ):
            value = row.get(key)
            if isinstance(value, str):
                out[key] = value.replace("\\r\\n", "\n").replace("\\n", "\n").strip()

        out["source_component_index"] = cls._safe_int(row.get("source_component_index"), 0)

        for key in ("style_spec", "bbox_xywh_on_layout", "canvas_wh", "source_canvas_wh"):
            parsed = cls._parse_json_cell(row.get(key))
            if parsed is not None:
                out[key] = parsed
        return out

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(float(value))
        except Exception:
            return default

    @staticmethod
    def _parse_json_cell(value: Any) -> Optional[Any]:
        if not isinstance(value, str):
            return None
        text = value.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            return None

    def build_model_input(self, sample: Dict[str, Any], *, modality: Any = None) -> Any:
        from design_benchmarks.models.base import ModelInput

        return ModelInput(
            text=str(sample.get("prompt") or ""),
            images=[],
            metadata={
                "benchmark_id": self.meta.id,
                "task": "g10_styled_text_element_generation",
                "mask": str(sample.get("mask") or ""),
                "text": str(sample.get("text") or ""),
                "style_spec": sample.get("style_spec") or {},
                "prompt": str(sample.get("prompt") or ""),
            },
        )

    def parse_model_output(self, output: Any) -> Any:
        if output is None:
            return None
        images = getattr(output, "images", None)
        if isinstance(images, list) and images:
            return images[0]
        if isinstance(output, dict):
            for key in ("image", "image_path", "prediction"):
                if key in output:
                    return output[key]
        return output

    def evaluate(self, predictions: List[Any], ground_truth: List[Any]) -> Dict[str, float]:
        cer_scores: List[float] = []
        ocr_acc_scores: List[float] = []
        edit_distance_scores: List[float] = []
        cer_alnum_scores: List[float] = []
        ocr_acc_alnum_scores: List[float] = []
        edit_distance_alnum_scores: List[float] = []
        font_top1_scores: List[float] = []
        font_top5_scores: List[float] = []
        font_size_mae_scores: List[float] = []
        color_mae_rgb_scores: List[float] = []
        text_align_acc_scores: List[float] = []
        curvature_acc_scores: List[float] = []
        spacing_line_height_mae_scores: List[float] = []
        line_height_mae_scores: List[float] = []
        letter_spacing_mae_scores: List[float] = []
        prop_acc_scores: List[float] = []
        prop_mae_scores: List[float] = []
        prop_cov_scores: List[float] = []
        de_scores: List[float] = []
        lpips_scores: List[float] = []
        ssim_scores: List[float] = []
        style_pred_rate_scores: List[float] = []
        bbox_iou_scores: List[float] = []
        bbox_f1_scores: List[float] = []
        bbox_precision_scores: List[float] = []
        bbox_recall_scores: List[float] = []
        bbox_detection_scores: List[float] = []

        evaluated = 0
        has_reconstruction_metrics = False

        def _collect_style_scores(style_pred: Optional[Dict[str, Any]], expected_style: Dict[str, Any]) -> None:
            style_pred_rate_scores.append(1.0 if isinstance(style_pred, dict) else 0.0)
            style_scores = self._style_scores(
                predicted=style_pred,
                expected=expected_style,
            )
            self._append_if_finite(
                font_top1_scores,
                style_scores["font_family_top1_accuracy"],
            )
            self._append_if_finite(
                font_top5_scores,
                style_scores["font_family_top5_accuracy"],
            )
            self._append_if_finite(
                font_size_mae_scores,
                style_scores["font_size_mae"],
            )
            self._append_if_finite(
                color_mae_rgb_scores,
                style_scores["color_mae_rgb"],
            )
            self._append_if_finite(
                text_align_acc_scores,
                style_scores["text_align_accuracy"],
            )
            self._append_if_finite(
                curvature_acc_scores,
                style_scores["curvature_accuracy"],
            )
            self._append_if_finite(
                spacing_line_height_mae_scores,
                style_scores["spacing_line_height_mae"],
            )
            self._append_if_finite(
                line_height_mae_scores,
                style_scores["line_height_mae"],
            )
            self._append_if_finite(
                letter_spacing_mae_scores,
                style_scores["letter_spacing_mae"],
            )
            self._append_if_finite(prop_acc_scores, style_scores["property_accuracy"])
            self._append_if_finite(prop_mae_scores, style_scores["property_mae"])
            self._append_if_finite(prop_cov_scores, style_scores["property_coverage"])

        for pred_raw, gt_raw in zip(predictions, ground_truth):
            gt = self._normalize_gt(gt_raw)
            mode = str(gt.get("evaluation_mode") or "inpaint_reconstruction").strip().lower()
            pred_img = self._to_rgb_array(pred_raw)
            if pred_img is None:
                continue

            # typography-6 text-only mode:
            # avoid forced resize/alignment against GT patch and evaluate from generated image.
            if mode == "text_style_only":
                ocr = self._ocr_metrics(
                    image=pred_img,
                    expected_text=str(gt.get("text", "")),
                )
                self._append_if_finite(cer_scores, ocr["cer"])
                self._append_if_finite(ocr_acc_scores, ocr["ocr_accuracy"])
                self._append_if_finite(edit_distance_scores, ocr["edit_distance"])
                self._append_if_finite(cer_alnum_scores, ocr["cer_alnum"])
                self._append_if_finite(ocr_acc_alnum_scores, ocr["ocr_accuracy_alnum"])
                self._append_if_finite(edit_distance_alnum_scores, ocr["edit_distance_alnum"])

                mask = self._to_gray_mask(gt.get("mask"), pred_img.shape[:2])
                gt_bbox = self._resolve_target_bbox_xyxy(
                    gt=gt,
                    image_hw=pred_img.shape[:2],
                )
                if gt_bbox is not None:
                    pred_bbox = self._detect_text_bbox_llm(
                        image=pred_img,
                        expected_text=str(gt.get("text", "")),
                        mask_bbox=self._mask_bbox(mask),
                        sample_id=str(gt.get("sample_id", "")),
                    )
                    self._append_if_finite(
                        bbox_detection_scores,
                        1.0 if pred_bbox is not None else 0.0,
                    )
                    if pred_bbox is not None:
                        self._append_if_finite(
                            bbox_iou_scores,
                            self._box_iou(pred_bbox, gt_bbox),
                        )
                        precision, recall, f1 = self._box_precision_recall_f1(pred_bbox, gt_bbox)
                        self._append_if_finite(bbox_precision_scores, precision)
                        self._append_if_finite(bbox_recall_scores, recall)
                        self._append_if_finite(bbox_f1_scores, f1)

                style_pred = self._predict_style_proxy(pred_img)
                _collect_style_scores(style_pred, gt.get("style_spec") or {})
                evaluated += 1
                continue

            gt_img = self._to_rgb_array(gt.get("ground_truth_image", ""))
            if gt_img is None:
                continue
            has_reconstruction_metrics = True

            mask = self._to_gray_mask(gt.get("mask"), gt_img.shape[:2])
            pred_img = self._resize_to_match(pred_img, gt_img.shape[:2])

            pred_region = self._crop_to_mask_bbox(pred_img, mask)
            gt_region = self._crop_to_mask_bbox(gt_img, mask)
            if pred_region is None:
                pred_region = pred_img
            if gt_region is None:
                gt_region = gt_img

            ocr = self._ocr_metrics(
                image=pred_region,
                expected_text=str(gt.get("text", "")),
            )
            self._append_if_finite(cer_scores, ocr["cer"])
            self._append_if_finite(ocr_acc_scores, ocr["ocr_accuracy"])
            self._append_if_finite(edit_distance_scores, ocr["edit_distance"])
            self._append_if_finite(cer_alnum_scores, ocr["cer_alnum"])
            self._append_if_finite(ocr_acc_alnum_scores, ocr["ocr_accuracy_alnum"])
            self._append_if_finite(edit_distance_alnum_scores, ocr["edit_distance_alnum"])

            de = self._masked_color_delta_e(pred_img, gt_img, mask)
            self._append_if_finite(de_scores, de)

            try:
                self._append_if_finite(lpips_scores, float(lpips_score(pred_img, gt_img)))
            except Exception:  # noqa: BLE001
                pass
            try:
                self._append_if_finite(ssim_scores, float(ssim_metric(pred_img, gt_img)))
            except Exception:  # noqa: BLE001
                pass

            style_pred = self._predict_style_proxy(pred_region)
            _collect_style_scores(style_pred, gt.get("style_spec") or {})
            evaluated += 1

        scores: Dict[str, float] = {
            "ocr_accuracy": self._mean_or_nan(ocr_acc_scores),
            "cer": self._mean_or_nan(cer_scores),
            "edit_distance": self._mean_or_nan(edit_distance_scores),
            "ocr_accuracy_alnum": self._mean_or_nan(ocr_acc_alnum_scores),
            "cer_alnum": self._mean_or_nan(cer_alnum_scores),
            "edit_distance_alnum": self._mean_or_nan(edit_distance_alnum_scores),
            "bbox_iou": self._mean_or_nan(bbox_iou_scores),
            "bbox_f1": self._mean_or_nan(bbox_f1_scores),
            "bbox_precision": self._mean_or_nan(bbox_precision_scores),
            "bbox_recall": self._mean_or_nan(bbox_recall_scores),
            "bbox_detection_rate": self._mean_or_nan(bbox_detection_scores),
            "font_family_top1_accuracy": self._mean_or_nan(font_top1_scores),
            "font_family_top5_accuracy": self._mean_or_nan(font_top5_scores),
            "font_size_mae": self._mean_or_nan(font_size_mae_scores),
            "color_mae_rgb": self._mean_or_nan(color_mae_rgb_scores),
            "text_align_accuracy": self._mean_or_nan(text_align_acc_scores),
            "curvature_accuracy": self._mean_or_nan(curvature_acc_scores),
            "spacing_line_height_mae": self._mean_or_nan(spacing_line_height_mae_scores),
            "line_height_mae": self._mean_or_nan(line_height_mae_scores),
            "letter_spacing_mae": self._mean_or_nan(letter_spacing_mae_scores),
            "property_accuracy": self._mean_or_nan(prop_acc_scores),
            "property_mae": self._mean_or_nan(prop_mae_scores),
            "property_coverage": self._mean_or_nan(prop_cov_scores),
            "style_prediction_rate": self._mean_or_nan(style_pred_rate_scores),
            "evaluated_samples": float(evaluated),
        }
        if has_reconstruction_metrics:
            scores.update(
                {
                    "color_delta_e": self._mean_or_nan(de_scores),
                    "lpips": self._mean_or_nan(lpips_scores),
                    "ssim": self._mean_or_nan(ssim_scores),
                }
            )
        return scores

    @staticmethod
    def _resolve_manifest_path(root: Path, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        p = Path(text)
        if p.is_file():
            return str(p)
        rel = root / text
        if rel.is_file():
            return str(rel)
        return ""

    @staticmethod
    def _parse_css_number(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        m = re.search(r"[-+]?\d*\.?\d+", str(value))
        if not m:
            return None
        try:
            return float(m.group(0))
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _clean_text(value: Any) -> str:
        text = html.unescape(str(value or ""))
        text = text.replace("</li>", " ").replace("<br>", " ").replace("<br/>", " ").replace("<br />", " ")
        text = re.sub(r"<[^>]+>", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _to_rgb_array(raw: Any) -> Optional[np.ndarray]:
        try:
            from PIL import Image
        except ImportError:
            return None

        pil: Optional[Image.Image] = None
        if isinstance(raw, np.ndarray):
            arr = raw
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            if arr.ndim == 3 and arr.shape[2] == 4:
                arr = arr[:, :, :3]
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return arr

        if isinstance(raw, Image.Image):
            pil = raw
        elif isinstance(raw, (str, Path)):
            p = Path(raw)
            if not p.is_file():
                return None
            try:
                pil = Image.open(p)
            except Exception:  # noqa: BLE001
                return None
        elif isinstance(raw, (bytes, bytearray)):
            try:
                pil = Image.open(io.BytesIO(raw))
            except Exception:  # noqa: BLE001
                return None

        if pil is None:
            return None
        try:
            return np.asarray(pil.convert("RGB"), dtype=np.uint8)
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _to_gray_mask(mask_like: Any, target_hw: Tuple[int, int]) -> Optional[np.ndarray]:
        try:
            from PIL import Image
        except ImportError:
            return None

        mask: Optional[np.ndarray] = None
        if isinstance(mask_like, np.ndarray):
            mask = mask_like
        elif isinstance(mask_like, Image.Image):
            mask = np.asarray(mask_like.convert("L"))
        elif isinstance(mask_like, (str, Path)):
            p = Path(mask_like)
            if not p.is_file():
                return None
            try:
                mask = np.asarray(Image.open(p).convert("L"))
            except Exception:  # noqa: BLE001
                return None
        elif isinstance(mask_like, (bytes, bytearray)):
            try:
                mask = np.asarray(Image.open(io.BytesIO(mask_like)).convert("L"))
            except Exception:  # noqa: BLE001
                return None

        if mask is None:
            return None
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        if mask.shape[:2] != target_hw:
            try:
                resized = Image.fromarray(mask.astype(np.uint8)).resize(
                    (target_hw[1], target_hw[0]),
                    Image.NEAREST,
                )
                mask = np.asarray(resized)
            except Exception:  # noqa: BLE001
                return None
        return mask.astype(np.uint8)

    @staticmethod
    def _resize_to_match(image: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
        if image.shape[:2] == target_hw:
            return image
        try:
            from PIL import Image

            return np.asarray(
                Image.fromarray(image).resize(
                    (target_hw[1], target_hw[0]),
                    Image.BILINEAR,
                ),
                dtype=np.uint8,
            )
        except Exception:  # noqa: BLE001
            return np.resize(image, (target_hw[0], target_hw[1], image.shape[2]))

    @staticmethod
    def _crop_to_mask_bbox(image: np.ndarray, mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if mask is None:
            return None
        ys, xs = np.where(mask > 127)
        if ys.size == 0 or xs.size == 0:
            return None
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        if y2 <= y1 or x2 <= x1:
            return None
        return image[y1:y2, x1:x2]

    @staticmethod
    def _run_ocr(image: np.ndarray) -> Optional[str]:
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            return None
        try:
            return str(pytesseract.image_to_string(Image.fromarray(image), config="--psm 6"))
        except Exception:  # noqa: BLE001
            return None

    @classmethod
    def _ocr_cer(cls, *, image: np.ndarray, expected_text: str) -> Tuple[float, float]:
        metrics = cls._ocr_metrics(image=image, expected_text=expected_text)
        return metrics["cer"], metrics["ocr_accuracy"]

    @classmethod
    def _ocr_metrics(cls, *, image: np.ndarray, expected_text: str) -> Dict[str, float]:
        expected = cls._clean_text(expected_text)
        if not expected:
            return {
                "cer": float("nan"),
                "ocr_accuracy": float("nan"),
                "edit_distance": float("nan"),
                "cer_alnum": float("nan"),
                "ocr_accuracy_alnum": float("nan"),
                "edit_distance_alnum": float("nan"),
            }

        raw_pred = cls._run_ocr(image)
        if raw_pred is None:
            return {
                "cer": float("nan"),
                "ocr_accuracy": float("nan"),
                "edit_distance": float("nan"),
                "cer_alnum": float("nan"),
                "ocr_accuracy_alnum": float("nan"),
                "edit_distance_alnum": float("nan"),
            }

        pred = cls._clean_text(raw_pred)
        edit_dist, cer, ocr_acc = cls._text_edit_metrics(predicted=pred, expected=expected)

        expected_alnum = cls._normalize_ocr_text(expected)
        pred_alnum = cls._normalize_ocr_text(pred)
        edit_dist_alnum, cer_alnum, ocr_acc_alnum = cls._text_edit_metrics(
            predicted=pred_alnum,
            expected=expected_alnum,
        )
        return {
            "cer": cer,
            "ocr_accuracy": ocr_acc,
            "edit_distance": edit_dist,
            "cer_alnum": cer_alnum,
            "ocr_accuracy_alnum": ocr_acc_alnum,
            "edit_distance_alnum": edit_dist_alnum,
        }

    @staticmethod
    def _normalize_ocr_text(text: str) -> str:
        lowered = str(text or "").lower()
        lowered = re.sub(r"[^0-9a-z\s]+", " ", lowered)
        return re.sub(r"\s+", " ", lowered).strip()

    @staticmethod
    def _text_edit_metrics(*, predicted: str, expected: str) -> Tuple[float, float, float]:
        expected_clean = str(expected or "").strip()
        if not expected_clean:
            return float("nan"), float("nan"), float("nan")
        pred_clean = str(predicted or "").strip()
        edit_dist = float(edit_distance(pred_clean, expected_clean))
        cer = max(0.0, edit_dist / max(len(expected_clean), 1))
        ocr_acc = max(0.0, 1.0 - cer)
        return edit_dist, cer, ocr_acc

    @staticmethod
    def _rgb_to_lab(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
        r, g, b = [x / 255.0 for x in rgb]

        def gamma_correct(c: float) -> float:
            return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

        r_lin = gamma_correct(r)
        g_lin = gamma_correct(g)
        b_lin = gamma_correct(b)

        x = r_lin * 0.4124564 + g_lin * 0.3575761 + b_lin * 0.1804375
        y = r_lin * 0.2126729 + g_lin * 0.7151522 + b_lin * 0.0721750
        z = r_lin * 0.0193339 + g_lin * 0.1191920 + b_lin * 0.9503041

        x /= 0.95047
        y /= 1.00000
        z /= 1.08883

        def f(t: float) -> float:
            return t ** (1 / 3) if t > 0.008856 else (7.787 * t + 16 / 116)

        fx = f(x)
        fy = f(y)
        fz = f(z)
        lab_l = 116 * fy - 16
        a = 500 * (fx - fy)
        b = 200 * (fy - fz)
        return lab_l, a, b

    @classmethod
    def _delta_e(cls, rgb1: Tuple[int, int, int], rgb2: Tuple[int, int, int]) -> float:
        l1, a1, b1 = cls._rgb_to_lab(rgb1)
        l2, a2, b2 = cls._rgb_to_lab(rgb2)
        return float(((l1 - l2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2) ** 0.5)

    @classmethod
    def _masked_color_delta_e(
        cls,
        pred: np.ndarray,
        gt: np.ndarray,
        mask: Optional[np.ndarray],
    ) -> float:
        if mask is None:
            return float("nan")
        active = mask > 127
        if not np.any(active):
            return float("nan")
        pred_rgb = pred[active]
        gt_rgb = gt[active]
        if pred_rgb.size == 0 or gt_rgb.size == 0:
            return float("nan")
        pred_mean = tuple(int(round(v)) for v in pred_rgb.mean(axis=0)[:3])
        gt_mean = tuple(int(round(v)) for v in gt_rgb.mean(axis=0)[:3])
        return cls._delta_e(pred_mean, gt_mean)

    @classmethod
    def _maybe_load_text_param_predictor(cls) -> Any:
        if cls._text_param_predictor_bundle is not None:
            return cls._text_param_predictor_bundle

        model_dir_env = os.environ.get("DESIGN_BENCHMARKS_G10_TEXTPARAM_MODEL_DIR", "").strip()
        if not model_dir_env:
            cls._text_param_predictor_bundle = False
            return cls._text_param_predictor_bundle

        model_dir = Path(model_dir_env).expanduser().resolve()
        if not model_dir.is_dir():
            cls._text_param_predictor_bundle = False
            return cls._text_param_predictor_bundle

        module_path = os.environ.get(
            "DESIGN_BENCHMARKS_G10_TEXTPARAM_MODULE_PATH",
            "/home/ubuntu/ml-platform/pipelines/text-params/inference/model/model.py",
        ).strip()
        module_file = Path(module_path).expanduser().resolve()
        if not module_file.is_file():
            cls._text_param_predictor_bundle = False
            return cls._text_param_predictor_bundle

        vocab_path = model_dir / "vocabularies.json"
        checkpoint_candidates = [model_dir / "final_model.pt", model_dir / "best_model.pt"]
        checkpoint_path = next((p for p in checkpoint_candidates if p.is_file()), None)
        if checkpoint_path is None or not vocab_path.is_file():
            cls._text_param_predictor_bundle = False
            return cls._text_param_predictor_bundle

        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location("g10_text_param_predictor", module_file)
            if spec is None or spec.loader is None:
                cls._text_param_predictor_bundle = False
                return cls._text_param_predictor_bundle

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.MODEL_CHECKPOINT_PATH = str(checkpoint_path)
            module.VOCABULARIES_PATH = str(vocab_path)
            backbone = os.environ.get("DESIGN_BENCHMARKS_G10_TEXTPARAM_BACKBONE", "").strip()
            if backbone:
                module.BACKBONE_NAME = backbone

            predictor = module.Model()
            predictor.load()
            cls._text_param_predictor_bundle = predictor
            logger.info(
                "Loaded G10 text-param predictor: model=%s vocab=%s",
                checkpoint_path,
                vocab_path,
            )
            return cls._text_param_predictor_bundle
        except Exception as exc:  # noqa: BLE001
            logger.info("G10 text-param predictor unavailable: %s", exc)
            cls._text_param_predictor_bundle = False
            return cls._text_param_predictor_bundle

    @classmethod
    def _predict_style_proxy(cls, image: np.ndarray) -> Optional[Dict[str, Any]]:
        local_pred = cls._predict_style_proxy_local(image)
        if isinstance(local_pred, dict):
            return local_pred
        api_pred = cls._predict_style_proxy_api(image)
        if isinstance(api_pred, dict):
            return api_pred
        return None

    @classmethod
    def _predict_style_proxy_local(cls, image: np.ndarray) -> Optional[Dict[str, Any]]:
        predictor = cls._maybe_load_text_param_predictor()
        if not predictor:
            return None
        try:
            from PIL import Image
        except ImportError:
            return None
        tmp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            Image.fromarray(image).save(tmp_path)
            pred = predictor.predict({"image_path": str(tmp_path)})
            return pred if isinstance(pred, dict) else None
        except Exception as exc:  # noqa: BLE001
            logger.debug("G10 style proxy prediction failed: %s", exc)
            return None
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:  # noqa: BLE001
                    pass

    @classmethod
    def _predict_style_proxy_api(cls, image: np.ndarray) -> Optional[Dict[str, Any]]:
        api_url = cls._resolve_textparam_api_url()
        if not api_url:
            return None
        try:
            import requests
        except ImportError:
            return None

        image_path = cls._prepare_textparam_api_image_path(image)
        if not image_path:
            return None

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        api_key = cls._resolve_textparam_api_key()
        if api_key:
            headers["Authorization"] = f"Api-Key {api_key}"

        payload: Dict[str, Any] = {"image_path": image_path}
        model_id = os.environ.get(cls.TEXTPARAM_API_MODEL_ID_ENV, "").strip()
        if model_id:
            payload["model_id"] = model_id

        timeout_s = cls._resolve_textparam_api_timeout_seconds()
        try:
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=timeout_s,
            )
            if not response.ok:
                logger.debug(
                    "G10 style proxy API failed (status=%s): %s",
                    response.status_code,
                    response.text[:300],
                )
                return None
            parsed = response.json()
            return parsed if isinstance(parsed, dict) else None
        except Exception as exc:  # noqa: BLE001
            logger.debug("G10 style proxy API request failed: %s", exc)
            return None

    @classmethod
    def _resolve_textparam_api_url(cls) -> str:
        for key in (cls.TEXTPARAM_API_URL_ENV, "TEXT_PARAMS_MODEL_URL"):
            value = str(os.environ.get(key, "")).strip()
            if value:
                return value
        return ""

    @classmethod
    def _resolve_textparam_api_key(cls) -> str:
        for key in (cls.TEXTPARAM_API_KEY_ENV, "TEXT_PARAMS_PREDICTOR_API_KEY", "BASETEN_API_KEY"):
            value = str(os.environ.get(key, "")).strip()
            if value:
                return value
        return ""

    @classmethod
    def _resolve_textparam_api_timeout_seconds(cls) -> float:
        raw = str(os.environ.get(cls.TEXTPARAM_API_TIMEOUT_ENV, "60")).strip()
        try:
            return max(1.0, float(raw))
        except Exception:  # noqa: BLE001
            return 60.0

    @classmethod
    def _prepare_textparam_api_image_path(cls, image: np.ndarray) -> str:
        uploaded = cls._maybe_upload_textparam_patch_to_gcs(image)
        if uploaded:
            return uploaded
        return cls._encode_image_data_uri(image)

    @classmethod
    def _maybe_upload_textparam_patch_to_gcs(cls, image: np.ndarray) -> str:
        bucket = str(os.environ.get(cls.TEXTPARAM_API_GCS_BUCKET_ENV, "")).strip()
        if not bucket:
            return ""
        prefix = str(
            os.environ.get(cls.TEXTPARAM_API_GCS_PREFIX_ENV, "lica-bench/g10-textparam")
        ).strip("/")
        credentials = str(os.environ.get(cls.TEXTPARAM_API_GCS_CREDENTIALS_ENV, "")).strip() or None

        try:
            from PIL import Image

            from design_benchmarks.inference.gcs import upload_file_public
        except Exception as exc:  # noqa: BLE001
            logger.debug("G10 text-param GCS upload unavailable: %s", exc)
            return ""

        tmp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            Image.fromarray(image).save(tmp_path)
            blob_name = f"{prefix}/{uuid.uuid4().hex}.png" if prefix else f"{uuid.uuid4().hex}.png"
            return str(
                upload_file_public(
                    local_path=tmp_path,
                    bucket_name=bucket,
                    blob_name=blob_name,
                    credentials_path=credentials,
                    signed_url_hours=24,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("G10 text-param GCS upload failed: %s", exc)
            return ""
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:  # noqa: BLE001
                    pass

    @staticmethod
    def _encode_image_data_uri(image: np.ndarray) -> str:
        try:
            from PIL import Image
        except ImportError:
            return ""
        try:
            buffer = io.BytesIO()
            Image.fromarray(image).save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return f"data:image/png;base64,{encoded}"
        except Exception:  # noqa: BLE001
            return ""

    @classmethod
    def _style_scores(cls, *, predicted: Optional[Dict[str, Any]], expected: Dict[str, Any]) -> Dict[str, float]:
        scores = cls._empty_style_scores(property_coverage=float("nan"))
        if not isinstance(expected, dict) or not expected:
            return scores
        if not isinstance(predicted, dict):
            scores["property_coverage"] = 0.0
            return scores

        acc_values: List[float] = []
        mae_values: List[float] = []
        expected_count = 0
        covered_count = 0

        gt_font = expected.get("fontFamily")
        if cls._has_value(gt_font):
            expected_count += 1
            pred_font = predicted.get("fontFamily")
            candidates = cls._font_family_candidates(predicted)
            if candidates:
                covered_count += 1
                pred_top1 = (
                    cls._normalize_font_family_value(pred_font)
                    if cls._has_value(pred_font)
                    else candidates[0]
                )
                top1_ok = pred_top1 == cls._normalize_font_family_value(gt_font)
                top1_score = 1.0 if top1_ok else 0.0
                scores["font_family_top1_accuracy"] = top1_score
                acc_values.append(top1_score)

                gt_norm = cls._normalize_font_family_value(gt_font)
                if gt_norm and candidates:
                    top5_ok = gt_norm in candidates[:5]
                    scores["font_family_top5_accuracy"] = 1.0 if top5_ok else 0.0
                else:
                    scores["font_family_top5_accuracy"] = top1_score

        gt_align = expected.get("textAlign")
        if cls._has_value(gt_align):
            expected_count += 1
            pred_align = predicted.get("textAlign")
            if cls._has_value(pred_align):
                covered_count += 1
                align_ok = (
                    cls._normalize_text_align_value(pred_align)
                    == cls._normalize_text_align_value(gt_align)
                )
                align_score = 1.0 if align_ok else 0.0
                scores["text_align_accuracy"] = align_score
                acc_values.append(align_score)

        for key in ("fontStyle", "textTransform", "fontWeight"):
            gt_val = expected.get(key)
            if not cls._has_value(gt_val):
                continue
            expected_count += 1
            pred_val = predicted.get(key)
            if not cls._has_value(pred_val):
                continue
            covered_count += 1
            if key == "fontWeight":
                ok = cls._normalize_font_weight_value(pred_val) == cls._normalize_font_weight_value(gt_val)
            else:
                ok = str(pred_val).strip().lower() == str(gt_val).strip().lower()
            acc_values.append(1.0 if ok else 0.0)

        gt_curvature = expected.get("curvature")
        pred_curvature = predicted.get("curvature")
        if cls._has_value(gt_curvature) and cls._has_value(pred_curvature):
            gt_bin = cls._normalize_curvature_binary(gt_curvature)
            pred_bin = cls._normalize_curvature_binary(pred_curvature)
            if gt_bin is not None and pred_bin is not None:
                scores["curvature_accuracy"] = 1.0 if gt_bin == pred_bin else 0.0

        gt_font_size = cls._style_font_size_px(expected)
        pred_font_size = cls._style_font_size_px(predicted)
        if gt_font_size is not None:
            expected_count += 1
            if pred_font_size is not None:
                covered_count += 1
                font_size_mae = abs(pred_font_size - gt_font_size)
                scores["font_size_mae"] = float(font_size_mae)
                mae_values.append(float(font_size_mae))

        gt_line_height = cls._style_line_height_px(expected, font_size_px=gt_font_size)
        pred_line_height = cls._style_line_height_px(
            predicted,
            font_size_px=pred_font_size if pred_font_size is not None else gt_font_size,
        )
        if gt_line_height is not None:
            expected_count += 1
            if pred_line_height is not None:
                covered_count += 1
                scores["line_height_mae"] = float(abs(pred_line_height - gt_line_height))
                mae_values.append(scores["line_height_mae"])

        gt_letter_spacing = cls._style_letter_spacing_px(expected, font_size_px=gt_font_size)
        pred_letter_spacing = cls._style_letter_spacing_px(
            predicted,
            font_size_px=pred_font_size if pred_font_size is not None else gt_font_size,
        )
        if gt_letter_spacing is not None:
            expected_count += 1
            if pred_letter_spacing is not None:
                covered_count += 1
                scores["letter_spacing_mae"] = float(abs(pred_letter_spacing - gt_letter_spacing))
                mae_values.append(scores["letter_spacing_mae"])

        pairwise_mae: List[float] = []
        if math.isfinite(scores["line_height_mae"]):
            pairwise_mae.append(scores["line_height_mae"])
        if math.isfinite(scores["letter_spacing_mae"]):
            pairwise_mae.append(scores["letter_spacing_mae"])
        if pairwise_mae:
            scores["spacing_line_height_mae"] = float(sum(pairwise_mae) / len(pairwise_mae))

        gt_rgb = cls._parse_css_color_rgb(expected.get("color"))
        pred_rgb = cls._parse_css_color_rgb(predicted.get("color"))
        if gt_rgb is not None and pred_rgb is not None:
            scores["color_mae_rgb"] = cls._rgb_mae(pred_rgb, gt_rgb)

        coverage = float(covered_count / expected_count) if expected_count > 0 else float("nan")
        scores["property_accuracy"] = cls._mean_or_nan(acc_values)
        scores["property_mae"] = cls._mean_or_nan(mae_values)
        scores["property_coverage"] = coverage
        return scores

    @staticmethod
    def _empty_style_scores(property_coverage: float) -> Dict[str, float]:
        return {
            "font_family_top1_accuracy": float("nan"),
            "font_family_top5_accuracy": float("nan"),
            "font_size_mae": float("nan"),
            "color_mae_rgb": float("nan"),
            "text_align_accuracy": float("nan"),
            "curvature_accuracy": float("nan"),
            "spacing_line_height_mae": float("nan"),
            "line_height_mae": float("nan"),
            "letter_spacing_mae": float("nan"),
            "property_accuracy": float("nan"),
            "property_mae": float("nan"),
            "property_coverage": property_coverage,
        }

    @staticmethod
    def _has_value(value: Any) -> bool:
        return value is not None and str(value).strip() != ""

    @staticmethod
    def _normalize_font_family_value(value: Any) -> str:
        text = normalize_font_name(str(value or ""))
        text = re.sub(r"--\d{2,4}$", "", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _normalize_font_weight_value(value: Any) -> str:
        text = str(value or "").strip().lower()
        if not text:
            return ""
        mapping = {
            "thin": "100",
            "extralight": "200",
            "ultralight": "200",
            "light": "300",
            "normal": "400",
            "regular": "400",
            "book": "400",
            "medium": "500",
            "semibold": "600",
            "demibold": "600",
            "bold": "700",
            "extrabold": "800",
            "ultrabold": "800",
            "black": "900",
            "heavy": "900",
        }
        if text in mapping:
            return mapping[text]
        num = StyledTextGeneration._safe_float(text)
        if num is None:
            return text
        snapped = int(round(num / 100.0) * 100)
        snapped = min(900, max(100, snapped))
        return str(snapped)

    @staticmethod
    def _normalize_text_align_value(value: Any) -> str:
        text = str(value or "").strip().lower()
        mapping = {
            "start": "left",
            "end": "right",
            "middle": "center",
        }
        return mapping.get(text, text)

    @staticmethod
    def _normalize_curvature_binary(value: Any) -> Optional[bool]:
        number = StyledTextGeneration._safe_float(value)
        if number is not None:
            return abs(number) > 1e-3
        text = str(value or "").strip().lower()
        if text in {"true", "t", "yes", "y", "on", "curved"}:
            return True
        if text in {"false", "f", "no", "n", "off", "flat", "none"}:
            return False
        return None

    @classmethod
    def _font_family_candidates(cls, predicted: Dict[str, Any]) -> List[str]:
        sources: List[Any] = []
        if cls._has_value(predicted.get("fontFamily")):
            sources.append(predicted.get("fontFamily"))

        keys = (
            "fontFamily_top5",
            "fontFamilyTop5",
            "font_family_top5",
            "fontFamilyCandidates",
            "fontFamily_candidates",
            "top5_fonts",
        )
        for key in keys:
            if key in predicted:
                sources.append(predicted.get(key))
        raw = predicted.get("_raw")
        if isinstance(raw, dict):
            for key in keys:
                if key in raw:
                    sources.append(raw.get(key))

        out: List[str] = []
        seen: set = set()
        for value in sources:
            candidates: List[str] = []
            if isinstance(value, (list, tuple)):
                candidates = [str(v) for v in value]
            elif isinstance(value, str):
                text = value.strip()
                if not text:
                    continue
                if text.startswith("[") and text.endswith("]"):
                    try:
                        parsed = json.loads(text)
                        if isinstance(parsed, list):
                            candidates = [str(v) for v in parsed]
                    except Exception:  # noqa: BLE001
                        candidates = [seg.strip() for seg in text.split(",") if seg.strip()]
                else:
                    candidates = [seg.strip() for seg in text.split(",") if seg.strip()]
            else:
                continue

            for candidate in candidates:
                norm = cls._normalize_font_family_value(candidate)
                if not norm or norm in seen:
                    continue
                seen.add(norm)
                out.append(norm)
        return out

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        m = re.search(r"[-+]?\d*\.?\d+", str(value))
        if not m:
            return None
        try:
            return float(m.group(0))
        except Exception:  # noqa: BLE001
            return None

    @classmethod
    def _css_length_to_px(
        cls,
        value: Any,
        *,
        font_size_px: Optional[float],
        unitless_as_multiplier: bool = False,
        normal_as_zero: bool = False,
    ) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)

        text = str(value).strip().lower()
        if not text:
            return None
        if text == "normal":
            return 0.0 if normal_as_zero else None

        number = cls._safe_float(text)
        if number is None:
            return None

        if "%" in text:
            if font_size_px is None:
                return None
            return float(number * font_size_px / 100.0)
        if text.endswith("em") or text.endswith("rem"):
            if font_size_px is None:
                return None
            return float(number * font_size_px)
        if text.endswith("pt"):
            return float(number * 96.0 / 72.0)

        is_unitless = re.fullmatch(r"[-+]?\d*\.?\d+", text) is not None
        if unitless_as_multiplier and is_unitless and font_size_px is not None and abs(number) <= 10.0:
            return float(number * font_size_px)
        return float(number)

    @classmethod
    def _style_font_size_px(cls, spec: Dict[str, Any]) -> Optional[float]:
        if not isinstance(spec, dict):
            return None
        parsed = cls._safe_float(spec.get("fontSize_px"))
        if parsed is not None:
            return parsed
        return cls._css_length_to_px(spec.get("fontSize"), font_size_px=None)

    @classmethod
    def _style_line_height_px(cls, spec: Dict[str, Any], *, font_size_px: Optional[float]) -> Optional[float]:
        if not isinstance(spec, dict):
            return None
        parsed = cls._css_length_to_px(
            spec.get("lineHeight"),
            font_size_px=font_size_px,
            unitless_as_multiplier=True,
            normal_as_zero=False,
        )
        if parsed is not None:
            return parsed
        return cls._safe_float(spec.get("lineHeight_px"))

    @classmethod
    def _style_letter_spacing_px(cls, spec: Dict[str, Any], *, font_size_px: Optional[float]) -> Optional[float]:
        if not isinstance(spec, dict):
            return None
        parsed = cls._css_length_to_px(
            spec.get("letterSpacing"),
            font_size_px=font_size_px,
            unitless_as_multiplier=False,
            normal_as_zero=True,
        )
        if parsed is not None:
            return parsed
        return cls._safe_float(spec.get("letterSpacing_value"))

    @staticmethod
    def _parse_css_color_rgb(value: Any) -> Optional[Tuple[float, float, float]]:
        def _clamp(channel: float) -> float:
            return max(0.0, min(255.0, float(channel)))

        def _parse_channel(token: str) -> Optional[float]:
            tok = token.strip().lower()
            if not tok:
                return None
            if tok.endswith("%"):
                num = StyledTextGeneration._safe_float(tok[:-1])
                if num is None:
                    return None
                return _clamp(num * 255.0 / 100.0)
            num = StyledTextGeneration._safe_float(tok)
            if num is None:
                return None
            return _clamp(num)

        if value is None:
            return None
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            try:
                return (_clamp(float(value[0])), _clamp(float(value[1])), _clamp(float(value[2])))
            except Exception:  # noqa: BLE001
                return None

        text = str(value).strip().lower()
        if not text:
            return None

        if text.startswith("#"):
            hex_color = text[1:]
            if len(hex_color) == 3:
                hex_color = "".join(ch * 2 for ch in hex_color)
            if len(hex_color) == 6 and re.fullmatch(r"[0-9a-f]{6}", hex_color):
                return (
                    float(int(hex_color[0:2], 16)),
                    float(int(hex_color[2:4], 16)),
                    float(int(hex_color[4:6], 16)),
                )

        rgb_match = re.search(r"rgba?\(([^)]+)\)", text)
        if rgb_match:
            chunks = [seg.strip() for seg in rgb_match.group(1).split(",")]
            if len(chunks) >= 3:
                channels = [_parse_channel(chunks[0]), _parse_channel(chunks[1]), _parse_channel(chunks[2])]
                if all(ch is not None for ch in channels):
                    return (float(channels[0]), float(channels[1]), float(channels[2]))

        if "," in text:
            chunks = [seg.strip() for seg in text.split(",")]
            if len(chunks) >= 3:
                channels = [_parse_channel(chunks[0]), _parse_channel(chunks[1]), _parse_channel(chunks[2])]
                if all(ch is not None for ch in channels):
                    return (float(channels[0]), float(channels[1]), float(channels[2]))

        return None

    @staticmethod
    def _rgb_mae(pred: Tuple[float, float, float], gt: Tuple[float, float, float]) -> float:
        return float((abs(pred[0] - gt[0]) + abs(pred[1] - gt[1]) + abs(pred[2] - gt[2])) / 3.0)

    @classmethod
    def _normalize_gt(cls, raw: Any) -> Dict[str, Any]:
        if not isinstance(raw, dict):
            return {
                "sample_id": "",
                "text": "",
                "style_spec": {},
                "ground_truth_image": "",
                "mask": "",
                "evaluation_mode": "inpaint_reconstruction",
                "target_bbox_xywh_on_layout": None,
            }
        bbox_xywh = cls._normalize_bbox_xywh(
            raw.get("target_bbox_xywh_on_layout", raw.get("bbox_xywh_on_layout")),
        )
        return {
            "sample_id": str(raw.get("sample_id") or ""),
            "text": cls._clean_text(raw.get("text")),
            "style_spec": raw.get("style_spec") if isinstance(raw.get("style_spec"), dict) else {},
            "ground_truth_image": str(raw.get("ground_truth_image") or ""),
            "mask": str(raw.get("mask") or ""),
            "input_image": str(raw.get("input_image") or ""),
            "evaluation_mode": str(raw.get("evaluation_mode") or "inpaint_reconstruction").strip().lower(),
            "target_bbox_xywh_on_layout": bbox_xywh,
        }

    @staticmethod
    def _append_if_finite(bucket: List[float], value: float) -> None:
        try:
            val = float(value)
        except Exception:  # noqa: BLE001
            return
        if math.isfinite(val):
            bucket.append(val)

    @staticmethod
    def _mean_or_nan(values: List[float]) -> float:
        if not values:
            return float("nan")
        return float(sum(values) / len(values))

    @staticmethod
    def _normalize_bbox_xywh(value: Any) -> Optional[List[float]]:
        if not isinstance(value, (list, tuple)) or len(value) < 4:
            return None
        out: List[float] = []
        for raw in value[:4]:
            try:
                out.append(float(raw))
            except Exception:  # noqa: BLE001
                return None
        return out

    @staticmethod
    def _xywh_to_xyxy(x: float, y: float, w: float, h: float) -> Box:
        return (
            int(round(x)),
            int(round(y)),
            int(round(x + w)),
            int(round(y + h)),
        )

    @staticmethod
    def _clamp_box(box: Optional[Box], width: int, height: int) -> Optional[Box]:
        if box is None:
            return None
        x1, y1, x2, y2 = box
        x1 = max(0, min(width, int(x1)))
        y1 = max(0, min(height, int(y1)))
        x2 = max(0, min(width, int(x2)))
        y2 = max(0, min(height, int(y2)))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    @staticmethod
    def _box_area(box: Optional[Box]) -> float:
        if box is None:
            return 0.0
        x1, y1, x2, y2 = box
        return float(max(0, x2 - x1) * max(0, y2 - y1))

    @classmethod
    def _box_iou(cls, a: Optional[Box], b: Optional[Box]) -> float:
        if a is None or b is None:
            return 0.0
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        inter = cls._box_area((ix1, iy1, ix2, iy2))
        union = cls._box_area(a) + cls._box_area(b) - inter
        if union <= 0:
            return 0.0
        return float(inter / union)

    @classmethod
    def _box_precision_recall_f1(
        cls,
        pred: Optional[Box],
        gt: Optional[Box],
    ) -> Tuple[float, float, float]:
        if pred is None or gt is None:
            return 0.0, 0.0, 0.0
        px1, py1, px2, py2 = pred
        gx1, gy1, gx2, gy2 = gt
        ix1 = max(px1, gx1)
        iy1 = max(py1, gy1)
        ix2 = min(px2, gx2)
        iy2 = min(py2, gy2)
        inter = cls._box_area((ix1, iy1, ix2, iy2))
        pred_area = cls._box_area(pred)
        gt_area = cls._box_area(gt)
        precision = float(inter / pred_area) if pred_area > 0 else 0.0
        recall = float(inter / gt_area) if gt_area > 0 else 0.0
        if precision + recall <= 0:
            return precision, recall, 0.0
        f1 = 2.0 * precision * recall / (precision + recall)
        return float(precision), float(recall), float(f1)

    @staticmethod
    def _mask_bbox(mask: Optional[np.ndarray]) -> Optional[Box]:
        if mask is None:
            return None
        ys, xs = np.where(mask > 127)
        if ys.size == 0 or xs.size == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

    @classmethod
    def _resolve_target_bbox_xyxy(
        cls,
        *,
        gt: Dict[str, Any],
        image_hw: Tuple[int, int],
    ) -> Optional[Box]:
        bbox_xywh = cls._normalize_bbox_xywh(gt.get("target_bbox_xywh_on_layout"))
        if bbox_xywh is None:
            return None
        h, w = image_hw
        box = cls._xywh_to_xyxy(
            bbox_xywh[0],
            bbox_xywh[1],
            bbox_xywh[2],
            bbox_xywh[3],
        )
        return cls._clamp_box(box, width=w, height=h)

    @staticmethod
    def _encode_png_bytes(image: np.ndarray) -> bytes:
        from PIL import Image

        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="PNG")
        return buffer.getvalue()

    @staticmethod
    def _extract_json_object_text(raw: str) -> str:
        text = str(raw or "").strip()
        if not text:
            return ""
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            text = text.strip()
        if text.startswith("{") and text.endswith("}"):
            return text
        m = re.search(r"\{[\s\S]*\}", text)
        return m.group(0).strip() if m else text

    @classmethod
    def _parse_bbox_detector_response(
        cls,
        raw_text: str,
        *,
        width: int,
        height: int,
    ) -> Optional[Box]:
        text = cls._extract_json_object_text(raw_text)
        if not text:
            return None

        parsed: Dict[str, Any] = {}
        norm_text = re.sub(r'"\s+([xywh]\d?)"', r'"\1"', text)
        norm_text = re.sub(
            r'"\s+(x1|y1|x2|y2|x|y|w|h|width|height|bbox)"',
            r'"\1"',
            norm_text,
        )
        try:
            obj = json.loads(norm_text)
            if isinstance(obj, dict):
                parsed = obj
        except Exception:  # noqa: BLE001
            parsed = {}

        def _as_int(v: Any) -> Optional[int]:
            try:
                return int(round(float(v)))
            except Exception:  # noqa: BLE001
                return None

        x1 = _as_int(parsed.get("x1"))
        y1 = _as_int(parsed.get("y1"))
        x2 = _as_int(parsed.get("x2"))
        y2 = _as_int(parsed.get("y2"))

        if None in (x1, y1, x2, y2):
            x = _as_int(parsed.get("x"))
            y = _as_int(parsed.get("y"))
            w = _as_int(parsed.get("w") if parsed.get("w") is not None else parsed.get("width"))
            h = _as_int(parsed.get("h") if parsed.get("h") is not None else parsed.get("height"))
            if None not in (x, y, w, h):
                x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)

        if None in (x1, y1, x2, y2):
            bbox_list = parsed.get("bbox")
            if isinstance(bbox_list, list) and len(bbox_list) >= 4:
                vals = [_as_int(v) for v in bbox_list[:4]]
                if None not in vals:
                    x1, y1, x2, y2 = int(vals[0]), int(vals[1]), int(vals[2]), int(vals[3])

        if None in (x1, y1, x2, y2):
            pairs = list(
                re.finditer(
                    r'(?i)"?\s*(x1|y1|x2|y2|x|y)\s*"?\s*[:=]?\s*"?\s*(-?\d+(?:\.\d+)?)',
                    norm_text,
                ),
            )
            by_key: Dict[str, List[int]] = {}
            for m in pairs:
                key = str(m.group(1)).lower()
                val = _as_int(m.group(2))
                if val is None:
                    continue
                by_key.setdefault(key, []).append(int(val))

            if x1 is None and by_key.get("x1"):
                x1 = int(by_key["x1"][0])
            if y1 is None and by_key.get("y1"):
                y1 = int(by_key["y1"][0])
            if x2 is None and by_key.get("x2"):
                x2 = int(by_key["x2"][0])
            if y2 is None and by_key.get("y2"):
                y2 = int(by_key["y2"][0])

            if x2 is None and x1 is not None and by_key.get("x"):
                candidates = [v for v in by_key["x"] if v != x1]
                if candidates:
                    larger = [v for v in candidates if v > x1]
                    x2 = int(larger[0] if larger else candidates[-1])
            if y2 is None and y1 is not None and by_key.get("y"):
                candidates = [v for v in by_key["y"] if v != y1]
                if candidates:
                    larger = [v for v in candidates if v > y1]
                    y2 = int(larger[0] if larger else candidates[-1])

        if None in (x1, y1, x2, y2):
            return None
        return cls._clamp_box((int(x1), int(y1), int(x2), int(y2)), width=width, height=height)

    @staticmethod
    def _bbox_detector_prompt(
        *,
        expected_text: str,
        image_wh: Tuple[int, int],
        mask_bbox: Optional[Box],
    ) -> str:
        w, h = image_wh
        target = str(expected_text or "").strip()
        hint = "null" if mask_bbox is None else str(list(mask_bbox))
        return (
            "You are a visual grounding assistant.\n"
            "Find the bounding box of the target text in this image.\n"
            "The text may span multiple lines.\n"
            f"Image size: width={w}, height={h}\n"
            f"Target text:\n{target}\n"
            f"Mask hint (x1,y1,x2,y2): {hint}\n\n"
            "Return only one JSON object with fields:\n"
            '{"found": true|false, "x1": int, "y1": int, "x2": int, "y2": int}\n'
            "No markdown or extra text."
        )

    @classmethod
    def _get_bbox_detector_model(cls) -> Optional[Any]:
        model_id = str(
            os.environ.get(cls.BBOX_DETECTOR_MODEL_ID_ENV, cls.BBOX_DETECTOR_DEFAULT_MODEL_ID),
        ).strip() or cls.BBOX_DETECTOR_DEFAULT_MODEL_ID
        if cls._bbox_detector_model is not None and cls._bbox_detector_model_id == model_id:
            return cls._bbox_detector_model
        if cls._bbox_detector_disabled:
            return None
        if not os.environ.get("OPENAI_API_KEY"):
            cls._bbox_detector_disabled = True
            logger.warning("OPENAI_API_KEY missing; bbox detector metrics will be skipped.")
            return None
        try:
            from design_benchmarks.models import load_model

            cls._bbox_detector_model = load_model(
                "openai",
                model_id=model_id,
                temperature=0.0,
                max_tokens=512,
            )
            cls._bbox_detector_model_id = model_id
            return cls._bbox_detector_model
        except Exception as exc:  # noqa: BLE001
            cls._bbox_detector_disabled = True
            logger.warning("Failed to initialize bbox detector model: %s", exc)
            return None

    @classmethod
    def _detect_text_bbox_llm(
        cls,
        *,
        image: np.ndarray,
        expected_text: str,
        mask_bbox: Optional[Box],
        sample_id: str = "",
    ) -> Optional[Box]:
        model = cls._get_bbox_detector_model()
        if model is None:
            return None
        h, w = image.shape[:2]
        expected_clean = cls._clean_text(expected_text)
        if not expected_clean:
            return None

        png_bytes = cls._encode_png_bytes(image)
        image_hash = hashlib.sha1(png_bytes).hexdigest()
        text_hash = hashlib.sha1(expected_clean.encode("utf-8")).hexdigest()
        cache_key = f"{sample_id}|{w}x{h}|{image_hash}|{text_hash}"
        if cache_key in cls._bbox_detector_cache:
            return cls._bbox_detector_cache[cache_key]

        prompt = cls._bbox_detector_prompt(
            expected_text=expected_clean,
            image_wh=(w, h),
            mask_bbox=mask_bbox,
        )
        try:
            from design_benchmarks.models.base import ModelInput

            out = model.predict(ModelInput(text=prompt, images=[png_bytes]))
            raw_text = str(getattr(out, "text", "") or "")
            bbox = cls._parse_bbox_detector_response(raw_text, width=w, height=h)
        except Exception as exc:  # noqa: BLE001
            logger.debug("bbox detector request failed for sample %s: %s", sample_id, exc)
            bbox = None

        cls._bbox_detector_cache[cache_key] = bbox
        return bbox

    @staticmethod
    def _style_prompt_value(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            try:
                return json.dumps(value, ensure_ascii=False)
            except Exception:  # noqa: BLE001
                return str(value)
        return str(value).strip()

    @classmethod
    def _style_prompt_lines(
        cls,
        style_spec: Dict[str, Any],
        *,
        include_position_keys: bool = True,
    ) -> List[str]:
        if not isinstance(style_spec, dict) or not style_spec:
            return ["- (no explicit typography values provided)"]

        preferred_keys: List[str] = [
            "fontFamily",
            "fontSize",
            "fontWeight",
            "fontStyle",
            "color",
            "textAlign",
            "lineHeight",
            "letterSpacing",
            "textTransform",
            "curvature",
            "autoResizeHeight",
            "styleRanges",
            "fontSize_px",
            "lineHeight_px",
            "letterSpacing_value",
        ]
        if include_position_keys:
            preferred_keys.extend(["left", "top", "width", "height"])
        excluded_when_position_off = {"left", "top", "width", "height"}
        lines: List[str] = []
        seen: set[str] = set()

        for key in preferred_keys:
            if key not in style_spec:
                continue
            value_text = cls._style_prompt_value(style_spec.get(key))
            if not value_text:
                continue
            lines.append(f"- {key}: {value_text}")
            seen.add(key)

        for key in sorted(style_spec.keys()):
            if key in seen:
                continue
            if not include_position_keys and key in excluded_when_position_off:
                continue
            value_text = cls._style_prompt_value(style_spec.get(key))
            if not value_text:
                continue
            lines.append(f"- {key}: {value_text}")

        return lines or ["- (no explicit typography values provided)"]

    @classmethod
    def _compose_element_prompt(cls, *, text: str, style_spec: Dict[str, Any]) -> str:
        lines = [
            "You are an expert typography compositor for structured design layouts.",
            "Task: generate one styled text element from text and style specification only.",
            f'Target text (exact, case-sensitive): "{text}"',
            "",
            "Typography/style specification (layout schema values):",
        ]
        lines.extend(cls._style_prompt_lines(style_spec, include_position_keys=False))
        lines.extend(
            [
                "",
                "Input semantics:",
                "- No input image is provided for this task.",
                "- Synthesize the text element directly from text and style values.",
                "",
                "Requirements:",
                "- Render exactly the target text: preserve characters, spaces, punctuation, and symbols.",
                "- Never translate, paraphrase, or normalize the target text.",
                "- Respect textTransform and intended line breaks; do not normalize case.",
                "- Match fontFamily, fontWeight, fontStyle, and fontSize.",
                "- Match color, lineHeight, letterSpacing, and textAlign.",
                "- If curvature/autoResizeHeight/styleRanges are provided, follow them exactly.",
                "- Keep glyph edges crisp and naturally anti-aliased (no blur, halo, ringing, or jagged artifacts).",
                "- Keep text fully visible; avoid clipping or truncation artifacts.",
                "- If style constraints conflict, prioritize exact text fidelity and full visibility without clipping.",
                "- Do not add extra words, glyphs, logos, or decorative marks.",
                "",
                "Output: return one generated image only.",
            ]
        )
        return "\n".join(lines)

    @classmethod
    def _resolve_use_manifest_prompt(cls) -> bool:
        raw = os.environ.get(cls.USE_MANIFEST_PROMPT_ENV, "0")
        text = str(raw).strip().lower()
        return text in {"1", "true", "t", "yes", "y", "on"}


@benchmark
class MixedStyleTextGeneration(BaseBenchmark):
    """typography-7 — G10 styled text rendering to layout (text inpainting)."""

    pipeline_implemented = True

    meta = BenchmarkMeta(
        id="typography-7",
        name="G10 Styled Text Rendering to Layout",
        task_type=TaskType.GENERATION,
        domain="typography",
        data_subpath="typography/Typography-6-Styled-Text-Generation",
        description="Render missing styled text into a masked layout region",
        input_spec="Text string + typography specification + masked layout image + text mask",
        output_spec="Layout image with styled text restored in masked region",
        metrics=[
            "ocr_accuracy",
            "cer",
            "edit_distance",
            "ocr_accuracy_alnum",
            "cer_alnum",
            "edit_distance_alnum",
            "font_family_top1_accuracy",
            "font_family_top5_accuracy",
            "font_size_mae",
            "color_mae_rgb",
            "text_align_accuracy",
            "curvature_accuracy",
            "spacing_line_height_mae",
            "line_height_mae",
            "letter_spacing_mae",
            "property_accuracy",
            "property_mae",
            "property_coverage",
            "color_delta_e",
            "lpips",
            "ssim",
            "style_prediction_rate",
        ],
    )

    _delegate = StyledTextGeneration()

    def load_data(
        self,
        data_dir: Union[str, Path],
        *,
        n: Optional[int] = None,
        dataset_root: Union[str, Path],
    ) -> List[Dict[str, Any]]:
        manifest_path = self._delegate._resolve_g10_manifest_path(
            data_dir=data_dir,
            json_name=self._delegate.INPAINT_MANIFEST_JSON,
            csv_name=self._delegate.INPAINT_MANIFEST_CSV,
            missing_message="G10 inpaint manifest not found",
        )
        rows = self._delegate._read_g10_manifest_rows(manifest_path)

        root = manifest_path.parent
        samples: List[Dict[str, Any]] = []
        for i, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            text = self._delegate._clean_text(row.get("text"))
            if not text:
                continue
            input_path = self._delegate._resolve_manifest_path(root, row.get("input_image"))
            gt_path = self._delegate._resolve_manifest_path(root, row.get("ground_truth_image"))
            mask_path = self._delegate._resolve_manifest_path(root, row.get("mask"))
            if not input_path or not gt_path or not mask_path:
                continue

            style_spec = row.get("style_spec") if isinstance(row.get("style_spec"), dict) else {}
            row_prompt = str(row.get("prompt") or "").strip()
            if self._delegate._resolve_use_manifest_prompt() and row_prompt:
                prompt = row_prompt
            else:
                prompt = self._compose_inpaint_prompt(text=text, style_spec=style_spec)
            sample_id = str(row.get("sample_id") or f"g10_inpaint_{i:04d}")
            gt_bundle = {
                "sample_id": sample_id,
                "text": text,
                "style_spec": style_spec,
                "ground_truth_image": gt_path,
                "mask": mask_path,
                "input_image": input_path,
                "evaluation_mode": "inpaint_reconstruction",
            }
            samples.append(
                {
                    "sample_id": sample_id,
                    "text": text,
                    "style_spec": style_spec,
                    "prompt": prompt,
                    "input_image": input_path,
                    "mask": mask_path,
                    "ground_truth_image": gt_path,
                    "ground_truth": gt_bundle,
                }
            )
            if n is not None and len(samples) >= n:
                break

        if not samples:
            raise ValueError(f"No valid samples in G10 inpaint manifest: {manifest_path}")
        return samples

    def build_model_input(self, sample: Dict[str, Any], *, modality: Any = None) -> Any:
        from design_benchmarks.models.base import Modality, ModelInput

        images: List[str] = []
        if modality != Modality.TEXT:
            input_image = str(sample.get("input_image") or "").strip()
            if input_image:
                images.append(input_image)

        return ModelInput(
            text=str(sample.get("prompt") or ""),
            images=images,
            metadata={
                "benchmark_id": self.meta.id,
                "task": "g10_styled_text_layout_inpaint",
                "mask": str(sample.get("mask") or ""),
                "text": str(sample.get("text") or ""),
                "style_spec": sample.get("style_spec") or {},
                "prompt": str(sample.get("prompt") or ""),
            },
        )

    def parse_model_output(self, output: Any) -> Any:
        return self._delegate.parse_model_output(output)

    def evaluate(self, predictions: List[Any], ground_truth: List[Any]) -> Dict[str, float]:
        return self._delegate.evaluate(predictions, ground_truth)

    @classmethod
    def _compose_inpaint_prompt(cls, *, text: str, style_spec: Dict[str, Any]) -> str:
        lines = [
            "You are an expert typography inpainting model for design layouts.",
            "Goal: restore exactly one missing text layer in the provided layout image.",
            "",
            "Target text (render verbatim, case-sensitive):",
            f'"{text}"',
            "",
            "Typography/style specification (layout schema values):",
        ]
        lines.extend(StyledTextGeneration._style_prompt_lines(style_spec))
        lines.extend(
            [
                "",
                "Mask and edit rules:",
                "- A binary mask is provided by the task runtime.",
                "- White mask pixels are editable; black pixels must remain unchanged.",
                "- Edit only the masked region.",
                "- Use left/top/width/height as placement cues when present.",
                "- Keep the full text visible inside the intended region; avoid clipping.",
                "",
                "Rendering rules:",
                "- Render exactly the target text: preserve characters, spaces, punctuation, and symbols.",
                "- Never translate, paraphrase, or normalize the target text.",
                "- Respect textTransform and intended line breaks; do not normalize case.",
                "- Match fontFamily, fontWeight, fontStyle, fontSize, color, lineHeight, letterSpacing, and textAlign.",
                "- If curvature/autoResizeHeight/styleRanges are provided, follow them exactly.",
                "- Keep glyph edges crisp and naturally anti-aliased (no blur, halo, ringing, or jagged artifacts).",
                "- If style constraints conflict, prioritize exact text fidelity and clean readability.",
                "- Do not add extra words, glyphs, logos, or decorative marks.",
                "",
                "Output: return one edited image only.",
            ]
        )
        return "\n".join(lines)

