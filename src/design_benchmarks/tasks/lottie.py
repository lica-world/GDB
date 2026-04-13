"""Lottie animation benchmarks (lottie-1, lottie-2).

Both tasks are implemented.

Data contract: each task reads ``{task-id}.json`` from the ``--data`` directory.
The JSON is an array of objects with ``question``, ``image``, and ``answer`` keys.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from design_benchmarks.base import BaseBenchmark, BenchmarkMeta, TaskType, benchmark
from design_benchmarks.utils.data_helpers import load_task_json
from design_benchmarks.utils.text_helpers import strip_thinking

logger = logging.getLogger(__name__)

# -- Lottie JSON helpers ----------------------------------------------------


def _parse_lottie_json(text: str) -> Optional[dict]:
    text = strip_thinking(text)
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return None


_LOTTIE_REQUIRED_KEYS = ("v", "fr", "ip", "op", "w", "h", "layers")


def _is_valid_lottie(data: Optional[dict]) -> bool:
    if not data:
        return False
    return all(k in data for k in _LOTTIE_REQUIRED_KEYS)


def _lottie_structural_similarity(pred: dict, gt: dict) -> float:
    scores: list = []
    pl = len(pred.get("layers", []))
    gl = len(gt.get("layers", []))
    scores.append(1.0 - abs(pl - gl) / max(pl, gl, 1))
    ptypes = {lay.get("ty") for lay in pred.get("layers", [])}
    gtypes = {lay.get("ty") for lay in gt.get("layers", [])}
    if ptypes or gtypes:
        scores.append(len(ptypes & gtypes) / len(ptypes | gtypes))
    pw, ph = pred.get("w", 1), pred.get("h", 1)
    gw, gh = gt.get("w", 1), gt.get("h", 1)
    scores.append(1.0 - abs(pw * ph - gw * gh) / max(pw * ph, gw * gh, 1))
    pfr, gfr = pred.get("fr", 30), gt.get("fr", 30)
    scores.append(1.0 - abs(pfr - gfr) / max(pfr, gfr, 1))
    return sum(scores) / len(scores) if scores else 0.0


def _render_lottie_frame(
    lottie_data: dict, frame_idx: int = 0, size: int = 256,
) -> Optional[Any]:
    """Render a single Lottie frame to a PIL RGB Image.

    Returns ``None`` when ``rlottie-python`` or ``Pillow`` is unavailable.
    """
    try:
        from PIL import Image
        from rlottie_python import LottieAnimation
    except ImportError:
        return None

    try:
        anim = LottieAnimation.from_data(data=json.dumps(lottie_data))
        total = anim.lottie_animation_get_totalframe()
        frame_idx = max(0, min(frame_idx, total - 1))
        buf = anim.lottie_animation_render(frame_num=frame_idx, width=size, height=size)
        img = Image.frombuffer("RGBA", (size, size), buf, "raw", "BGRA", 0, 1)
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg
    except Exception:
        return None


_NUM_SAMPLE_FRAMES = 5


def _frame_mse(pred_data: dict, gt_data: dict, size: int = 256) -> Optional[float]:
    """Mean-squared error averaged over sampled frames. Lower is better."""
    try:
        import numpy as np
        from rlottie_python import LottieAnimation
    except ImportError:
        return None

    try:
        pred_anim = LottieAnimation.from_data(data=json.dumps(pred_data))
        gt_anim = LottieAnimation.from_data(data=json.dumps(gt_data))
        pred_total = pred_anim.lottie_animation_get_totalframe()
        gt_total = gt_anim.lottie_animation_get_totalframe()
        total = max(pred_total, gt_total, 1)
        indices = [int(i * (total - 1) / max(_NUM_SAMPLE_FRAMES - 1, 1)) for i in range(_NUM_SAMPLE_FRAMES)]
        mse_sum = 0.0
        for idx in indices:
            pred_img = _render_lottie_frame(pred_data, min(idx, pred_total - 1), size)
            gt_img = _render_lottie_frame(gt_data, min(idx, gt_total - 1), size)
            if pred_img is None or gt_img is None:
                mse_sum += 1.0
                continue
            diff = np.array(pred_img, dtype=np.float64) - np.array(gt_img, dtype=np.float64)
            mse_sum += float(np.mean(diff ** 2)) / 65025.0
        return mse_sum / len(indices)
    except Exception:
        return None


def _frame_ssim(pred_data: dict, gt_data: dict, size: int = 256) -> Optional[float]:
    """SSIM averaged over sampled frames. Higher is better."""
    try:
        import numpy as np
        from rlottie_python import LottieAnimation
        from skimage.metrics import structural_similarity
    except ImportError:
        return None

    try:
        pred_anim = LottieAnimation.from_data(data=json.dumps(pred_data))
        gt_anim = LottieAnimation.from_data(data=json.dumps(gt_data))
        pred_total = pred_anim.lottie_animation_get_totalframe()
        gt_total = gt_anim.lottie_animation_get_totalframe()
        total = max(pred_total, gt_total, 1)
        indices = [int(i * (total - 1) / max(_NUM_SAMPLE_FRAMES - 1, 1)) for i in range(_NUM_SAMPLE_FRAMES)]
        ssim_sum = 0.0
        for idx in indices:
            pred_img = _render_lottie_frame(pred_data, min(idx, pred_total - 1), size)
            gt_img = _render_lottie_frame(gt_data, min(idx, gt_total - 1), size)
            if pred_img is None or gt_img is None:
                continue
            ssim_sum += float(structural_similarity(
                np.array(pred_img), np.array(gt_img), channel_axis=2, data_range=255,
            ))
        return ssim_sum / len(indices)
    except Exception:
        return None


def _evaluate_lottie(
    predictions: List[str], ground_truth: List[Dict[str, str]],
) -> Dict[str, float]:
    n = max(len(predictions), 1)
    val_s = struct_s = cl_s = 0.0
    mse_vals: List[Optional[float]] = []
    ssim_vals: List[Optional[float]] = []

    _warned_render = False

    for pred_text, gt_dict in zip(predictions, ground_truth):
        gt_text = gt_dict.get("lottie_json", "")
        cl_s += len(pred_text.encode("utf-8"))
        pred_data = _parse_lottie_json(pred_text)
        gt_data = _parse_lottie_json(gt_text)
        valid = _is_valid_lottie(pred_data)
        val_s += 1.0 if valid else 0.0
        if valid and gt_data:
            struct_s += _lottie_structural_similarity(pred_data, gt_data)
            mse_val = _frame_mse(pred_data, gt_data)
            ssim_val = _frame_ssim(pred_data, gt_data)
            if mse_val is None and not _warned_render:
                logger.warning(
                    "Lottie frame metrics unavailable — install "
                    "lica-bench[lottie-metrics] for frame_mse/frame_ssim."
                )
                _warned_render = True
            mse_vals.append(mse_val)
            ssim_vals.append(ssim_val)
        else:
            mse_vals.append(None)
            ssim_vals.append(None)

    real_mse = [v for v in mse_vals if v is not None]
    real_ssim = [v for v in ssim_vals if v is not None]

    scores: Dict[str, float] = {
        "lottie_validity": val_s / n,
        "structural_similarity": struct_s / n,
        "code_length": cl_s / n,
    }
    if real_mse:
        scores["frame_mse"] = sum(real_mse) / len(real_mse)
    if real_ssim:
        scores["frame_ssim"] = sum(real_ssim) / len(real_ssim)

    return scores


# ===========================================================================
# Implemented tasks
# ===========================================================================


@benchmark
class TextToLottieGeneration(BaseBenchmark):
    """lottie-1 — Generate Lottie animation JSON from a text description."""

    pipeline_implemented = True

    PROMPT = (
        "You are a Lottie animation generator. "
        "Given a description of an animation, output ONLY valid Lottie JSON "
        "(the standard Bodymovin format with keys: v, fr, ip, op, w, h, layers). "
        "Do not include any explanation, markdown fences, or extra text."
    )

    meta = BenchmarkMeta(
        id="lottie-1",
        name="Text-to-Lottie Generation",
        task_type=TaskType.GENERATION,
        domain="lottie",
        description="Generate Lottie animation JSON from a text description",
        input_spec="Natural-language description of animation",
        output_spec="Lottie JSON (Bodymovin format)",
        metrics=["lottie_validity", "structural_similarity", "frame_mse", "frame_ssim", "code_length"],
    )

    def load_data(self, data_dir, *, n=None, dataset_root: Union[str, Path]):
        data, data_root = load_task_json(data_dir, self.meta.id)
        samples = [
            {
                "sample_id": f"lottie_{i:03d}",
                "ground_truth": {
                    "lottie_json": item["answer"],
                    "description": item["question"][0] if item.get("question") else "",
                },
                "description": item["question"][0] if item.get("question") else "",
                "image_path": (
                    str(data_root / item["image"])
                    if item.get("image") and not Path(item["image"]).is_absolute()
                    else item.get("image", "")
                ),
            }
            for i, item in enumerate(data)
        ]
        return samples[:n] if n is not None else samples

    def build_model_input(self, sample, *, modality=None):
        from design_benchmarks.models.base import ModelInput

        return ModelInput(
            text=f"{self.PROMPT}\n\nDescription: {sample['description']}",
            images=[],
        )

    def parse_model_output(self, output):
        return strip_thinking(output.text.strip())

    def evaluate(self, predictions, ground_truth):
        return _evaluate_lottie(predictions, ground_truth)


@benchmark
class ImageTextToLottieGeneration(BaseBenchmark):
    """lottie-2 — Generate Lottie animation JSON from a keyframe image and description."""

    pipeline_implemented = True

    PROMPT = (
        "You are a Lottie animation generator. "
        "Given an animation keyframe image and its description, output ONLY valid "
        "Lottie JSON (the standard Bodymovin format with keys: v, fr, ip, op, w, h, layers). "
        "Do not include any explanation, markdown fences, or extra text."
    )

    meta = BenchmarkMeta(
        id="lottie-2",
        name="Image-Text-to-Lottie Generation",
        task_type=TaskType.GENERATION,
        domain="lottie",
        description="Generate Lottie animation JSON from a keyframe image and description",
        input_spec="Animation keyframe image + natural-language description",
        output_spec="Lottie JSON (Bodymovin format)",
        metrics=["lottie_validity", "structural_similarity", "frame_mse", "frame_ssim", "code_length"],
    )

    def load_data(self, data_dir, *, n=None, dataset_root: Union[str, Path]):
        data, data_root = load_task_json(data_dir, self.meta.id)
        samples = [
            {
                "sample_id": f"lottie_{i:03d}",
                "ground_truth": {
                    "lottie_json": item["answer"],
                    "description": item["question"][0] if item.get("question") else "",
                },
                "description": item["question"][0] if item.get("question") else "",
                "image_path": (
                    str(data_root / item["image"])
                    if item.get("image") and not Path(item["image"]).is_absolute()
                    else item.get("image", "")
                ),
            }
            for i, item in enumerate(data)
        ]
        return samples[:n] if n is not None else samples

    def build_model_input(self, sample, *, modality=None):
        from design_benchmarks.models.base import ModelInput

        images: list = []
        ip = sample.get("image_path", "")
        if ip and Path(ip).exists():
            images.append(ip)
        return ModelInput(
            text=f"{self.PROMPT}\n\nDescription: {sample['description']}",
            images=images,
        )

    def parse_model_output(self, output):
        return strip_thinking(output.text.strip())

    def evaluate(self, predictions, ground_truth):
        return _evaluate_lottie(predictions, ground_truth)
