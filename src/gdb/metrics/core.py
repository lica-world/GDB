"""Core metric primitives used by benchmark tasks."""

from typing import Any


def _missing_extra(package: str, extra: str) -> ImportError:
    return ImportError(
        f"{package} is required for this metric. "
        f'Install with: pip install -e ".[{extra}]"'
    )


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def iou(pred: Any, gt: Any) -> float:
    """Compute IoU between two axis-aligned bounding boxes.

    If the optional third-party ``evaluation`` package is installed, delegates
    to ``evaluation.iou``; otherwise uses a small local implementation.
    """
    try:
        from evaluation import iou as _iou

        return _iou(pred, gt)
    except ImportError:
        pass

    try:
        px, py, pw, ph = pred
        gx, gy, gw, gh = gt

        x1 = max(px, gx)
        y1 = max(py, gy)
        x2 = min(px + pw, gx + gw)
        y2 = min(py + ph, gy + gh)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        union = pw * ph + gw * gh - intersection
        return intersection / union if union > 0 else 0.0
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Text similarity
# ---------------------------------------------------------------------------


def edit_distance(source: str, target: str) -> float:
    """Character-level edit distance between two strings.

    Counts insert/delete/replace characters via ``difflib`` opcodes.
    """
    import re
    from difflib import SequenceMatcher

    a = re.sub(r"\s+", " ", source).strip()
    b = re.sub(r"\s+", " ", target).strip()
    ops = SequenceMatcher(None, a, b).get_opcodes()
    distance = 0
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            continue
        distance += max(i2 - i1, j2 - j1)
    return float(distance)


# ---------------------------------------------------------------------------
# Image quality
# ---------------------------------------------------------------------------


def ssim(pred: Any, gt: Any) -> float:
    """Structural similarity index.

    If the optional third-party ``evaluation.image`` module is available,
    delegates to it; otherwise uses ``scikit-image``.
    """
    try:
        from evaluation.image import ssim as _ssim

        return _ssim(pred, gt)
    except ImportError:
        pass

    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        raise _missing_extra("scikit-image", "metrics")

    return float(structural_similarity(gt, pred, channel_axis=-1))


def lpips_score(pred: Any, gt: Any) -> float:
    """Learned perceptual image patch similarity (lower is better).

    If the optional third-party ``evaluation.image.lpips`` is available,
    delegates to it; otherwise uses the ``lpips`` PyTorch package.
    """
    try:
        from evaluation.image import lpips as _lpips

        result = _lpips(pred, gt)
        if result is not None:
            return result
    except ImportError:
        pass

    try:
        import lpips
        import torch
    except ImportError:
        raise _missing_extra("lpips / torch", "svg-metrics")

    loss_fn = lpips.LPIPS(net="alex")
    return float(loss_fn(torch.tensor(pred), torch.tensor(gt)))


def fid(real_features: Any, generated_features: Any) -> float:
    """Fréchet Inception Distance between two sets of features.

    Expects NumPy arrays of shape ``(N, D)`` (e.g. Inception-v3 pool3
    activations).  Requires ``scipy``.
    """
    import numpy as np

    try:
        from scipy.linalg import sqrtm
    except ImportError:
        raise _missing_extra("scipy", "metrics")

    mu_r = np.mean(real_features, axis=0)
    mu_g = np.mean(generated_features, axis=0)
    sigma_r = np.cov(real_features, rowvar=False)
    sigma_g = np.cov(generated_features, rowvar=False)

    diff = mu_r - mu_g
    covmean, _ = sqrtm(sigma_r @ sigma_g, disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean))
