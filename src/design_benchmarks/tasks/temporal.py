"""Temporal benchmarks: temporal-1, temporal-2, temporal-3, temporal-4, temporal-5, temporal-6.

  temporal-1  Keyframe Ordering
  temporal-2  Motion Type Classification (video-level, all variants)
  temporal-3  Animation Property Extraction (per-component JSON; README counts three
              timing task lines—clip/video duration, component duration, start time—plus
              motion type, speed, direction in the same benchmark)
  temporal-4  Animation Parameter Generation (video gen from static layout)
  temporal-5  Motion Trajectory Generation (single-component entrance video)
  temporal-6  Short-Form Video Layout Generation (text-only video gen)

Data contract: ``samples.csv`` in the task directory under
``benchmarks/temporal/<TaskDirName>/``.  ``image_path`` (or ``video_path``)
values in the CSV are resolved against ``dataset_root``.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from design_benchmarks.base import BaseBenchmark, BenchmarkMeta, TaskType, benchmark

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KEYFRAME_ORDERING_NUM_FRAMES = 4

LICA_MOTION_TYPES: List[str] = [
    "ascend", "baseline", "block", "blur", "bounce", "breathe", "burst",
    "clarify", "drift", "fade", "flicker", "merge", "neon", "pan",
    "photoFlow", "photoRise", "pop", "pulse", "rise", "roll", "rotate",
    "scrapbook", "shift", "skate", "stomp", "succession", "tectonic",
    "tumble", "typewriter", "wiggle", "wipe", "none",
]

# ---------------------------------------------------------------------------
# Shared helpers — motion type normalisation & parsing
# ---------------------------------------------------------------------------


def normalize_motion_type(raw: str) -> str:
    """Normalize a motion type label against the canonical LICA set."""
    text = raw.strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    canon_map = {t.lower().replace("_", ""): t for t in LICA_MOTION_TYPES}
    if text in canon_map:
        return canon_map[text]
    for canon_key, canon_val in canon_map.items():
        if text in canon_key or canon_key in text:
            return canon_val
    return raw.strip().lower()


def _parse_motion_type_single(raw: str) -> str:
    """Parse a single motion type label from model output."""
    text = raw.strip()
    for line in text.splitlines():
        line = line.strip().strip('"\'.,;:')
        if line:
            return normalize_motion_type(line)
    return normalize_motion_type(text)


def _strip_json_fences(raw: str) -> str:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    bracket_start = text.find("[")
    bracket_end = text.rfind("]")
    if bracket_start != -1 and bracket_end != -1:
        text = text[bracket_start : bracket_end + 1]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text


def _parse_json_array(raw: str) -> List[Any]:
    text = _strip_json_fences(raw)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass
    return []


def _parse_motion_type_array(raw: str) -> List[str]:
    """Parse a JSON array of motion type strings from model output."""
    arr = _parse_json_array(raw)
    return [normalize_motion_type(str(x)) for x in arr]


def _safe_float(val: Any) -> float:
    """Extract a float from arbitrary text, returning 0.0 on failure."""
    try:
        text = str(val).strip()
        m = re.search(r"-?\d+(?:\.\d+)?", text)
        return float(m.group()) if m else 0.0
    except (ValueError, TypeError):
        return 0.0


def _parse_property_array(raw: str) -> List[Dict[str, Any]]:
    """Parse a JSON array of property dicts from model output."""
    text = _strip_json_fences(raw)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass
    return []


# ---------------------------------------------------------------------------
# Shared helpers — classification metrics
# ---------------------------------------------------------------------------


def _accuracy(predictions: List[str], ground_truth: List[str]) -> float:
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    return correct / len(predictions) if predictions else 0.0


def _macro_f1(predictions: List[str], ground_truths: List[str]) -> float:
    gt_classes = sorted(set(ground_truths))
    if not gt_classes:
        return 0.0
    f1_sum = 0.0
    for c in gt_classes:
        tp = sum(1 for p, g in zip(predictions, ground_truths) if p == c and g == c)
        fp = sum(1 for p, g in zip(predictions, ground_truths) if p == c and g != c)
        fn = sum(1 for p, g in zip(predictions, ground_truths) if p != c and g == c)
        denom = 2 * tp + fp + fn
        f1_sum += (2 * tp / denom) if denom > 0 else 0.0
    return f1_sum / len(gt_classes)


def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


# ---------------------------------------------------------------------------
# Shared helpers — keyframe ordering evaluation
# ---------------------------------------------------------------------------


def parse_keyframe_ordering(raw: str) -> List[int]:
    """Parse model output into a 4-element ordering [1-4].

    Tries JSON array first, then falls back to extracting digits.
    Returns [1, 2, 3, 4] (identity) if parsing fails.
    """
    n = KEYFRAME_ORDERING_NUM_FRAMES
    arr = _parse_json_array(raw)
    if len(arr) >= n:
        try:
            result = [int(x) for x in arr[:n]]
            if sorted(result) == list(range(1, n + 1)):
                return result
        except (ValueError, TypeError):
            pass

    nums = re.findall(r"\d+", raw.strip())
    candidates = [int(x) for x in nums if 1 <= int(x) <= n]
    seen: set = set()
    deduped: List[int] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            deduped.append(c)
    if len(deduped) == n:
        return deduped

    return list(range(1, n + 1))


def _kendalls_tau(pred: List[int], gt: List[int]) -> float:
    n = len(pred)
    if n < 2:
        return 1.0
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            pred_sign = pred[i] - pred[j]
            gt_sign = gt[i] - gt[j]
            if pred_sign * gt_sign > 0:
                concordant += 1
            elif pred_sign * gt_sign < 0:
                discordant += 1
    total = n * (n - 1) / 2
    return (concordant - discordant) / total if total > 0 else 0.0


def _pairwise_accuracy(pred: List[int], gt: List[int]) -> float:
    n = len(pred)
    if n < 2:
        return 1.0
    correct = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            if (pred[i] - pred[j]) * (gt[i] - gt[j]) > 0:
                correct += 1
    return correct / total if total > 0 else 0.0


def _first_frame_accuracy(pred: List[int], gt: List[int]) -> float:
    if not pred or not gt:
        return 0.0
    return 1.0 if pred[0] == gt[0] else 0.0


# ===================================================================
# temporal-1  Keyframe Ordering
# ===================================================================


@benchmark
class KeyframeOrdering(BaseBenchmark):
    """temporal-1 — Sort shuffled keyframes into the correct temporal sequence."""

    pipeline_implemented = True

    meta = BenchmarkMeta(
        id="temporal-1",
        name="Keyframe Ordering",
        task_type=TaskType.UNDERSTANDING,
        domain="temporal",
        data_subpath="temporal/KeyframeOrdering",
        description="Sort shuffled keyframes into the correct temporal sequence",
        input_spec="Four shuffled keyframe images",
        output_spec="Permutation of frame indices (correct order)",
        metrics=["percent_perfect", "mean_kendalls_tau", "pairwise_accuracy",
                 "first_frame_accuracy"],
    )

    def load_data(self, data_dir, *, n=None, dataset_root: Union[str, Path]):
        root = Path(data_dir).resolve()
        csv_path = root / "samples.csv"
        keyframes_dir = root / "keyframes"

        if not csv_path.is_file():
            raise FileNotFoundError(f"samples.csv not found in {root}")

        samples: List[Dict[str, Any]] = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row["sample_id"]
                perm = json.loads(row["shuffle_permutation"])
                gt = json.loads(row["expected_output"])
                num = KEYFRAME_ORDERING_NUM_FRAMES

                original_frames = [
                    str(keyframes_dir / sid / f"image_{i + 1}.jpg")
                    for i in range(num)
                ]
                shuffled = [original_frames[perm[j]] for j in range(num)]

                samples.append({
                    "sample_id": sid,
                    "ground_truth": gt,
                    "prompt": row["prompt"],
                    "shuffle_permutation": perm,
                    "shuffled_keyframe_paths": shuffled,
                })
                if n is not None and len(samples) >= n:
                    break
        return samples

    def build_model_input(self, sample, *, modality=None):
        from design_benchmarks.models.base import ModelInput

        return ModelInput(
            text=sample["prompt"],
            images=sample["shuffled_keyframe_paths"],
        )

    def parse_model_output(self, output):
        return parse_keyframe_ordering(output.text)

    def evaluate(self, predictions, ground_truth):
        n = max(len(predictions), 1)
        perfect = 0
        taus: List[float] = []
        pairwise: List[float] = []
        first_frame: List[float] = []
        for pred, gt in zip(predictions, ground_truth):
            p = pred if isinstance(pred, list) else []
            g = gt if isinstance(gt, list) else []
            if p == g:
                perfect += 1
            taus.append(_kendalls_tau(p, g))
            pairwise.append(_pairwise_accuracy(p, g))
            first_frame.append(_first_frame_accuracy(p, g))
        return {
            "percent_perfect": perfect / n,
            "mean_kendalls_tau": _mean(taus),
            "pairwise_accuracy": _mean(pairwise),
            "first_frame_accuracy": _mean(first_frame),
        }


# ===================================================================
# temporal-2  Motion Type Classification (video-level, all variants)
# ===================================================================


@benchmark
class MotionTypeClassificationVideo(BaseBenchmark):
    """temporal-2 — Classify animation entrance type from video (open-vocab and constrained)."""

    pipeline_implemented = True

    meta = BenchmarkMeta(
        id="temporal-2",
        name="Motion Type Classification (Video)",
        task_type=TaskType.UNDERSTANDING,
        domain="temporal",
        data_subpath="temporal/MotionTypeClassification",
        description="Classify primary animation entrance type from supported motion types",
        input_spec="Animation video (MP4)",
        output_spec="Motion type label",
        metrics=["accuracy", "macro_f1"],
    )

    def load_data(self, data_dir, *, n=None, dataset_root: Union[str, Path]):
        root = Path(data_dir).resolve()
        csv_path = root / "samples.csv"
        base = Path(dataset_root).resolve()

        if not csv_path.is_file():
            raise FileNotFoundError(f"samples.csv not found in {root}")

        group_counts: Dict[Tuple[str, str], int] = {}
        samples: List[Dict[str, Any]] = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row["sample_id"]
                prompt = row["prompt"]
                comp_id = row["component_id"]
                is_video = comp_id == "video"
                variant = "constrained" if "Choose" in prompt else "open-vocab"
                level = "video" if is_video else "component"
                row_id = f"{sid}|{comp_id}|{variant}"

                key = (sid, prompt)
                idx = group_counts.get(key, 0)
                group_counts[key] = idx + 1

                video_path = str((base / row["image_path"]).resolve())
                samples.append({
                    "sample_id": sid,
                    "row_id": row_id,
                    "component_id": comp_id,
                    "video_path": video_path,
                    "prompt": prompt,
                    "ground_truth": row["expected_output"],
                    "is_video_level": is_video,
                    "variant": f"{variant}-{level}",
                    "component_index": idx,
                })
                if n is not None and len(samples) >= n:
                    break
        return samples

    def build_model_input(self, sample, *, modality=None):
        from design_benchmarks.models.base import ModelInput

        return ModelInput(
            text=sample["prompt"],
            images=[sample["video_path"]],
        )

    def parse_model_output(self, output):
        return output.text.strip()

    def evaluate(self, predictions, ground_truth):
        preds = [normalize_motion_type(str(p)) for p in predictions]
        gts = [normalize_motion_type(str(g)) for g in ground_truth]
        return {
            "accuracy": _accuracy(preds, gts),
            "macro_f1": _macro_f1(preds, gts),
        }


# ===================================================================
# temporal-3  Animation Property Extraction
# ===================================================================


@benchmark
class AnimationPropertyExtraction(BaseBenchmark):
    """temporal-3 — Extract per-component animation properties from video."""

    pipeline_implemented = True

    meta = BenchmarkMeta(
        id="temporal-3",
        name="Animation Property Extraction",
        task_type=TaskType.UNDERSTANDING,
        domain="temporal",
        data_subpath="temporal/AnimationPropertyExtraction",
        description=(
            "Extract per-component animation properties (motion type, duration, start offset, "
            "speed, direction). For reporting, duration is split into video/clip vs "
            "per-component duration task lines; start time is a separate task line from duration."
        ),
        input_spec="Animation video (MP4)",
        output_spec="JSON array of per-component animation property objects",
        metrics=["motion_type_accuracy", "duration_mae", "start_time_mae",
                 "speed_mae", "direction_accuracy", "component_count_mae"],
    )

    def load_data(self, data_dir, *, n=None, dataset_root: Union[str, Path]):
        root = Path(data_dir).resolve()
        csv_path = root / "samples.csv"
        base = Path(dataset_root).resolve()

        if not csv_path.is_file():
            raise FileNotFoundError(f"samples.csv not found in {root}")

        group_counts: Dict[Tuple[str, str], int] = {}
        samples: List[Dict[str, Any]] = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row["sample_id"]
                prompt = row["prompt"]
                comp_id = row["component_id"]
                row_id = f"{sid}|{comp_id}"

                key = (sid, prompt)
                idx = group_counts.get(key, 0)
                group_counts[key] = idx + 1

                video_path = str((base / row["image_path"]).resolve())
                expected = json.loads(row["expected_output"])
                samples.append({
                    "sample_id": sid,
                    "row_id": row_id,
                    "component_id": comp_id,
                    "video_path": video_path,
                    "prompt": prompt,
                    "ground_truth": expected,
                    "component_index": idx,
                })
                if n is not None and len(samples) >= n:
                    break
        return samples

    def build_model_input(self, sample, *, modality=None):
        from design_benchmarks.models.base import ModelInput

        return ModelInput(
            text=sample["prompt"],
            images=[sample["video_path"]],
        )

    def parse_model_output(self, output):
        return output.text.strip()

    def evaluate(self, predictions, ground_truth):
        video_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for i, (raw_pred, gt) in enumerate(zip(predictions, ground_truth)):
            parsed = _parse_property_array(str(raw_pred))
            gt_obj = gt if isinstance(gt, dict) else {}
            sid = f"sample_{i}"

            pred_obj: Dict[str, Any] = {}
            if parsed:
                pred_obj = parsed[0] if isinstance(parsed[0], dict) else {}

            pred_mt = normalize_motion_type(str(pred_obj.get("motion_type", "")))
            gt_mt = normalize_motion_type(str(gt_obj.get("motion_type", "")))

            video_groups[sid].append({
                "motion_type_match": pred_mt == gt_mt,
                "duration_error": abs(
                    _safe_float(pred_obj.get("duration_seconds", 0))
                    - _safe_float(gt_obj.get("duration_seconds", 0))
                ),
                "start_time_error": abs(
                    _safe_float(pred_obj.get("start_time_seconds", 0))
                    - _safe_float(gt_obj.get("start_time_seconds", 0))
                ),
                "speed_error": abs(
                    _safe_float(pred_obj.get("speed", 1))
                    - _safe_float(gt_obj.get("speed", 1))
                ),
                "direction_match": (
                    str(pred_obj.get("direction", "none")).strip().lower()
                    == str(gt_obj.get("direction", "none")).strip().lower()
                ),
            })

        all_type_acc: List[float] = []
        all_dur_mae: List[float] = []
        all_start_mae: List[float] = []
        all_speed_mae: List[float] = []
        all_dir_acc: List[float] = []

        for entries in video_groups.values():
            n_v = len(entries)
            all_type_acc.append(
                sum(1 for e in entries if e["motion_type_match"]) / n_v
            )
            all_dur_mae.append(
                sum(e["duration_error"] for e in entries) / n_v
            )
            all_start_mae.append(
                sum(e["start_time_error"] for e in entries) / n_v
            )
            all_speed_mae.append(
                sum(e["speed_error"] for e in entries) / n_v
            )
            all_dir_acc.append(
                sum(1 for e in entries if e["direction_match"]) / n_v
            )

        return {
            "motion_type_accuracy": _mean(all_type_acc),
            "duration_mae": _mean(all_dur_mae),
            "start_time_mae": _mean(all_start_mae),
            "speed_mae": _mean(all_speed_mae),
            "direction_accuracy": _mean(all_dir_acc),
        }


# ===================================================================
# temporal-4  Animation Parameter Generation
# ===================================================================


@benchmark
class AnimationParameterGeneration(BaseBenchmark):
    """temporal-4 — Generate animated video from a static layout and animation parameters."""

    pipeline_implemented = True

    meta = BenchmarkMeta(
        id="temporal-4",
        name="Animation Parameter Generation",
        task_type=TaskType.GENERATION,
        domain="temporal",
        data_subpath="temporal/AnimationParameterGeneration",
        description="Generate animated video from static layout and animation parameters",
        input_spec="Static layout image + animation prompt",
        output_spec="Generated video (path or URI)",
        metrics=["generation_success_rate"],
    )

    def load_data(self, data_dir, *, n=None, dataset_root: Union[str, Path]):
        root = Path(data_dir).resolve()
        csv_path = root / "samples.csv"
        base = Path(dataset_root).resolve()

        if not csv_path.is_file():
            raise FileNotFoundError(f"samples.csv not found in {root}")

        samples: List[Dict[str, Any]] = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                samples.append({
                    "sample_id": row["sample_id"],
                    "static_image_path": str((base / row["static_image_path"]).resolve()),
                    "prompt": row["prompt"],
                    "ground_truth": str((base / row["image_path"]).resolve()),
                    "gt_video_path": str((base / row["image_path"]).resolve()),
                })
                if n is not None and len(samples) >= n:
                    break
        return samples

    def build_model_input(self, sample, *, modality=None):
        from design_benchmarks.models.base import ModelInput

        return ModelInput(
            text=sample["prompt"],
            images=[sample["static_image_path"]],
        )

    def evaluate(self, predictions, ground_truth):
        n_ok = sum(1 for p in predictions if p)
        return {
            "generation_success_rate": n_ok / max(len(predictions), 1),
        }


# ===================================================================
# temporal-5  Motion Trajectory Generation
# ===================================================================


@benchmark
class MotionTrajectoryGeneration(BaseBenchmark):
    """temporal-5 — Generate single-component entrance animation video."""

    pipeline_implemented = True

    meta = BenchmarkMeta(
        id="temporal-5",
        name="Motion Trajectory Generation",
        task_type=TaskType.GENERATION,
        domain="temporal",
        data_subpath="temporal/MotionTrajectoryGeneration",
        description="Generate single-component entrance animation video",
        input_spec="Static layout image + single-component motion prompt",
        output_spec="Generated video (path or URI)",
        metrics=["generation_success_rate"],
    )

    def load_data(self, data_dir, *, n=None, dataset_root: Union[str, Path]):
        root = Path(data_dir).resolve()
        csv_path = root / "samples.csv"
        base = Path(dataset_root).resolve()

        if not csv_path.is_file():
            raise FileNotFoundError(f"samples.csv not found in {root}")

        samples: List[Dict[str, Any]] = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                samples.append({
                    "sample_id": row["sample_id"],
                    "static_image_path": str((base / row["static_image_path"]).resolve()),
                    "prompt": row["prompt"],
                    "ground_truth": str((base / row["image_path"]).resolve()),
                    "gt_video_path": str((base / row["image_path"]).resolve()),
                    "motion_type": row["motion_type"],
                    "component_index": int(row["component_index"]),
                    "component_type": row["component_type"],
                })
                if n is not None and len(samples) >= n:
                    break
        return samples

    def build_model_input(self, sample, *, modality=None):
        from design_benchmarks.models.base import ModelInput

        return ModelInput(
            text=sample["prompt"],
            images=[sample["static_image_path"]],
        )

    def evaluate(self, predictions, ground_truth):
        n_ok = sum(1 for p in predictions if p)
        return {
            "generation_success_rate": n_ok / max(len(predictions), 1),
        }


# ===================================================================
# temporal-6  Short-Form Video Layout Generation
# ===================================================================


@benchmark
class ShortFormVideoLayoutGeneration(BaseBenchmark):
    """temporal-6 — Generate short-form marketing video from a text-only brief."""

    pipeline_implemented = True

    meta = BenchmarkMeta(
        id="temporal-6",
        name="Short-Form Video Layout Generation",
        task_type=TaskType.GENERATION,
        domain="temporal",
        data_subpath="temporal/ShortFormVideoLayoutGeneration",
        description="Generate animated vertical video layout from marketing brief",
        input_spec="Marketing brief and aspect ratio (text only, no image input)",
        output_spec="Generated video (path or URI)",
        metrics=["generation_success_rate"],
    )

    def load_data(self, data_dir, *, n=None, dataset_root: Union[str, Path]):
        root = Path(data_dir).resolve()
        csv_path = root / "samples.csv"
        base = Path(dataset_root).resolve()

        if not csv_path.is_file():
            raise FileNotFoundError(f"samples.csv not found in {root}")

        samples: List[Dict[str, Any]] = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                samples.append({
                    "sample_id": row["sample_id"],
                    "prompt": row["prompt"],
                    "ground_truth": str((base / row["image_path"]).resolve()),
                    "gt_video_path": str((base / row["image_path"]).resolve()),
                    "aspect_ratio": row["aspect_ratio"],
                    "target_width": int(row["target_width"]),
                    "target_height": int(row["target_height"]),
                    "category": row.get("category", ""),
                })
                if n is not None and len(samples) >= n:
                    break
        return samples

    def build_model_input(self, sample, *, modality=None):
        from design_benchmarks.models.base import ModelInput

        return ModelInput(text=sample["prompt"])

    def evaluate(self, predictions, ground_truth):
        n_ok = sum(1 for p in predictions if p)
        return {
            "generation_success_rate": n_ok / max(len(predictions), 1),
        }
