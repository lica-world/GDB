"""Shared image array / mask / OCR helpers for benchmark tasks."""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any, List, Optional, Tuple


def to_rgb_array(image_like: Any) -> Optional[Any]:
    """Convert paths, URLs, bytes, PIL images, or ndarray to RGB uint8 H×W×3."""
    import numpy as np

    if isinstance(image_like, np.ndarray):
        arr = image_like
    else:
        try:
            from PIL import Image
        except ImportError:
            return None

        pil = None
        if isinstance(image_like, Image.Image):
            pil = image_like
        elif isinstance(image_like, (bytes, bytearray)):
            try:
                pil = Image.open(io.BytesIO(image_like))
            except Exception:  # noqa: BLE001
                return None
        elif isinstance(image_like, (str, Path)):
            source = str(image_like)
            if source.startswith(("http://", "https://")):
                try:
                    import requests

                    resp = requests.get(source, timeout=20)
                    resp.raise_for_status()
                    pil = Image.open(io.BytesIO(resp.content))
                except Exception:  # noqa: BLE001
                    return None
            else:
                source = source.strip()
                if not source:
                    return None
                p = Path(source)
                if p.is_file():
                    try:
                        pil = Image.open(p)
                    except Exception:  # noqa: BLE001
                        return None
        if pil is None:
            return None
        try:
            arr = np.asarray(pil.convert("RGB"))
        except Exception:  # noqa: BLE001
            return None

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    elif arr.ndim != 3:
        return None
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def to_gray_mask(mask_like: Any, target_hw: Tuple[int, int]) -> Optional[Any]:
    """Load or normalize a mask to uint8 H×W matching *target_hw*."""
    import numpy as np

    if isinstance(mask_like, np.ndarray):
        mask = mask_like
    else:
        try:
            from PIL import Image
        except ImportError:
            return None

        pil = None
        if isinstance(mask_like, Image.Image):
            pil = mask_like
        elif isinstance(mask_like, (str, Path)):
            p = Path(mask_like)
            if not p.exists():
                return None
            try:
                pil = Image.open(p)
            except Exception:  # noqa: BLE001
                return None
        elif isinstance(mask_like, (bytes, bytearray)):
            try:
                pil = Image.open(io.BytesIO(mask_like))
            except Exception:  # noqa: BLE001
                return None
        if pil is None:
            return None
        mask = np.asarray(pil.convert("L"))

    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.shape[:2] != target_hw:
        try:
            from PIL import Image

            mask = np.asarray(
                Image.fromarray(mask.astype(np.uint8)).resize(
                    (target_hw[1], target_hw[0]),
                    Image.NEAREST,
                )
            )
        except Exception:  # noqa: BLE001
            return None
    return mask.astype(np.uint8)


def resize_to_match(image: Any, target_hw: Tuple[int, int]) -> Any:
    """Resize H×W×C ndarray to *target_hw* (height, width)."""
    import numpy as np

    if image.shape[:2] == target_hw:
        return image
    try:
        from PIL import Image

        resized = Image.fromarray(image).resize(
            (target_hw[1], target_hw[0]),
            Image.BILINEAR,
        )
        return np.asarray(resized)
    except Exception:  # noqa: BLE001
        return np.resize(image, (target_hw[0], target_hw[1], image.shape[2]))


def run_ocr(image: Any) -> Optional[str]:
    """Run Tesseract OCR if ``pytesseract`` is installed."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return None
    try:
        return str(pytesseract.image_to_string(Image.fromarray(image), config="--psm 6"))
    except Exception:  # noqa: BLE001
        return None


def append_if_finite(bucket: List[float], value: float) -> None:
    try:
        val = float(value)
    except Exception:  # noqa: BLE001
        return
    if math.isfinite(val):
        bucket.append(val)


def mean_or_nan(values: List[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))
