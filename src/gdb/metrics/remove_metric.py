"""ReMOVE metric helper for object/text erasure quality.

This adapts the public ReMOVE reference implementation into a reusable class
for gdb.
"""

from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity

logger = logging.getLogger(__name__)

DEFAULT_SAM_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
DEFAULT_SAM_CACHE_DIR = Path.home() / ".cache" / "gdb" / "checkpoints"
DEFAULT_SAM_CHECKPOINT_PATH = DEFAULT_SAM_CACHE_DIR / "sam_vit_h_4b8939.pth"


def ensure_sam_checkpoint(path: Optional[Union[str, Path]] = None) -> Path:
    """Resolve SAM checkpoint path and download it when missing."""

    if path is None:
        target = DEFAULT_SAM_CHECKPOINT_PATH
    else:
        target = Path(path).expanduser()
        if target.exists() and target.is_dir():
            target = target / DEFAULT_SAM_CHECKPOINT_PATH.name
        elif target.suffix == "":
            target = target / DEFAULT_SAM_CHECKPOINT_PATH.name

    if target.exists():
        return target.resolve()

    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading SAM checkpoint for ReMOVE: %s", target)
    urllib.request.urlretrieve(DEFAULT_SAM_CHECKPOINT_URL, str(target))
    return target.resolve()


def _find_smallest_bounding_square(binary_image: np.ndarray) -> Optional[Tuple[int, int, int]]:
    """Return (x, y, size) for the smallest square that encloses the mask."""

    if binary_image.ndim == 3:
        binary_image = np.asarray(Image.fromarray(binary_image).convert("L"))

    white_pixels = np.argwhere(binary_image == 255)
    if white_pixels.size == 0:
        return None

    min_row = int(np.min(white_pixels[:, 0]))
    max_row = int(np.max(white_pixels[:, 0]))
    min_col = int(np.min(white_pixels[:, 1]))
    max_col = int(np.max(white_pixels[:, 1]))

    width = max_col - min_col + 1
    height = max_row - min_row + 1
    size = max(width, height)

    margin = 16
    h, w = binary_image.shape[:2]
    pad = margin if (
        min_col - margin >= 0
        and min_row - margin >= 0
        and max_row + margin < h
        and max_col + margin < w
    ) else max(min(min_col, min_row, h - 1 - max_row, w - 1 - max_col), 0)

    x = max(min_col - pad, 0)
    y = max(min_row - pad, 0)
    size = size + 2 * pad

    # Clamp so the final crop remains in-bounds.
    size = min(size, h - y, w - x)
    return x, y, size


@dataclass
class RemoveMetricEvaluator:
    """Compute ReMOVE score for an inpainted image and removal mask."""

    sam_checkpoint: Union[str, Path]
    model_type: str = "vit_h"
    device: Optional[str] = None
    crop: bool = True

    def __post_init__(self) -> None:
        self._setup_predictor()

    def _setup_predictor(self) -> None:
        try:
            from segment_anything import sam_model_registry
            from segment_anything.predictor import SamPredictor
        except ImportError as exc:
            raise ImportError(
                "segment-anything is required for ReMOVE metric. "
                "Install with: pip install git+https://github.com/facebookresearch/segment-anything.git"
            ) from exc

        checkpoint = Path(self.sam_checkpoint).expanduser()
        if not checkpoint.exists():
            raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")

        resolved_device = str(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if resolved_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for ReMOVE but CUDA is not available.")

        sam = sam_model_registry[self.model_type](checkpoint=str(checkpoint))
        sam = sam.to(resolved_device)
        sam.eval()

        self.predictor = SamPredictor(sam)
        self.device = resolved_device

    def _get_mask_embeddings(self, image_np: np.ndarray, masks: list[np.ndarray]) -> list[torch.Tensor]:
        if hasattr(self.predictor, "get_aggregate_features"):
            embeddings = self.predictor.get_aggregate_features(image_np, masks)
            return [self._ensure_tensor(e) for e in embeddings]
        return self._aggregate_features(image_np, masks)

    @staticmethod
    def _ensure_tensor(value: torch.Tensor | np.ndarray) -> torch.Tensor:
        if torch.is_tensor(value):
            return value
        return torch.as_tensor(value)

    def _aggregate_features(self, image_np: np.ndarray, masks: list[np.ndarray]) -> list[torch.Tensor]:
        with torch.no_grad():
            self.predictor.set_image(image_np)
            features = self.predictor.get_image_embedding()

        embeddings: list[torch.Tensor] = []
        for mask in masks:
            mask_tensor = torch.as_tensor(mask)
            if mask_tensor.ndim == 2:
                mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
            elif mask_tensor.ndim == 3:
                mask_tensor = mask_tensor.unsqueeze(0)

            mask_tensor = mask_tensor.to(features.device)
            if mask_tensor.shape[-2:] != features.shape[-2:]:
                mask_tensor = F.interpolate(
                    mask_tensor.float(),
                    size=features.shape[-2:],
                    mode="nearest",
                )
            mask_bool = mask_tensor > 0
            expanded_mask = mask_bool.expand_as(features)

            if not expanded_mask.any():
                embedding = torch.zeros((1, features.shape[1]), device=features.device)
            else:
                masked_features = features[expanded_mask]
                embedding = masked_features.view(1, features.shape[1], -1).mean(dim=2)

            embeddings.append(embedding.detach())

        return embeddings

    @torch.no_grad()
    def score(self, image: Image.Image, mask: Image.Image) -> Optional[float]:
        """Return ReMOVE score in [-1, 1], or None for empty masks."""

        image_np = np.asarray(image.convert("RGB"))
        mask_np = np.asarray(mask.convert("L"))

        if mask_np.max() == 0:
            return None

        crop_info = _find_smallest_bounding_square(mask_np) if self.crop else None
        if crop_info is not None:
            x, y, size = crop_info
            x2 = min(x + size, image_np.shape[1])
            y2 = min(y + size, image_np.shape[0])
            image_np = image_np[y:y2, x:x2]
            mask_np = mask_np[y:y2, x:x2]

        # ReMOVE reference implementation computes foreground/background embeddings
        # with a 64x64 binary mask.
        mask_fg = (
            np.asarray(Image.fromarray(mask_np).resize((64, 64), Image.NEAREST))
            .reshape(1, 1, 64, 64)
            // 255
        ).astype(np.uint8)
        mask_bg = 1 - mask_fg

        embeddings = self._get_mask_embeddings(image_np, [mask_fg, mask_bg])
        if len(embeddings) != 2:
            raise RuntimeError("Unexpected embedding count returned by SAM predictor for ReMOVE.")

        fg = embeddings[0]
        bg = embeddings[1]
        if fg.device != bg.device:
            bg = bg.to(fg.device)

        score = cosine_similarity(fg, bg).item()
        return float(np.clip(score, -1.0, 1.0))

