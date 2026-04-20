"""Text-removal benchmark implementation (grouped under typography domain)."""

from __future__ import annotations

import csv
import io
import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from gdb.base import BaseBenchmark, BenchmarkMeta, TaskType, benchmark
from gdb.metrics.core import fid as fid_metric

logger = logging.getLogger(__name__)


@benchmark
class TextRemoval(BaseBenchmark):
    """image-6 -- Remove text and inpaint background cleanly (G16-style).

    Data format:
    - JSON manifest (array or ``{"samples": [...]}``)
    - CSV manifest (header row + one sample per row)

    Each sample needs ``input_image`` and ``ground_truth_image``; ``mask`` is
    strongly recommended, but can be inferred from common path conventions
    (e.g. ``text_removal/mask/<sample_id>.png`` or ``/input/`` -> ``/mask/``).
    Optional fields: ``forbidden_texts`` and ``prompt``. Image paths are
    resolved relative to the manifest file's directory.

    Accepts either a direct path to a manifest file or a directory
    containing ``text_removal_manifest.json`` or ``text_removal_manifest.csv``.
    """

    pipeline_implemented = True

    meta = BenchmarkMeta(
        id="image-6",
        name="Text Removal & Background Inpainting",
        task_type=TaskType.GENERATION,
        domain="typography",
        data_subpath="image/image-6-text-removal",
        description="Remove text and inpaint the underlying background cleanly",
        input_spec="Layout image with text components masked (+ optional text mask)",
        output_spec="Clean image with text removed and background reconstructed",
        metrics=[
            "psnr",
            "ssim",
            "lpips",
            "dino_score",
            "clip_score",
            "fid",
            "fid_coverage",
            "ocr_text_absence",
            "ocr_coverage",
            "bbox_text_absence",
            "bbox_coverage",
            "remove",
            "remove_coverage",
        ],
    )

    DEFAULT_PROMPT = "Remove all text and reconstruct the background naturally."
    PROMPT_PREFIX = (
        "You are an expert design retoucher specialized in text removal and "
        "background inpainting."
    )
    PROMPT_SIGNATURE = (
        "Task: remove all visible text while preserving non-text visual content."
    )
    BBOX_TEXT_ABSENCE_ENABLED_ENV = "GDB_IMAGE6_USE_BBOX_DETECTOR"
    BBOX_TEXT_ABSENCE_MAX_PHRASES_ENV = "GDB_IMAGE6_BBOX_MAX_PHRASES"
    BBOX_TEXT_ABSENCE_MAX_CHARS_ENV = "GDB_IMAGE6_BBOX_MAX_CHARS"
    _remove_evaluator_bundle: Any = None
    _fid_inception_bundle: Any = None

    @classmethod
    def _looks_like_composed_prompt(cls, text: str) -> bool:
        raw = str(text or "")
        return (
            cls.PROMPT_SIGNATURE in raw
            and "Hard constraints (must satisfy all):" in raw
        )

    @classmethod
    def compose_model_prompt(
        cls,
        *,
        user_prompt: str = "",
        forbidden_texts: Optional[List[str]] = None,
    ) -> str:
        objective = str(user_prompt or "").strip() or cls.DEFAULT_PROMPT
        forbidden = [str(t).strip() for t in (forbidden_texts or []) if str(t).strip()]

        lines = [
            cls.PROMPT_PREFIX,
            cls.PROMPT_SIGNATURE,
            "",
            f"Objective: {objective}",
            "",
            "Input semantics:",
            "- Image #1 is the original layout image.",
            "- A binary text mask is provided by the task runtime.",
            "- White mask pixels indicate where text is likely present.",
            "- Full-image regeneration is allowed; strict unmasked preservation is not required.",
        ]

        if forbidden:
            lines.extend(
                [
                    "",
                    "Texts that must be absent in the final output:",
                ]
            )
            for text in forbidden[:40]:
                lines.append(f'- "{text}"')

        lines.extend(
            [
                "",
                "Hard constraints (must satisfy all):",
                "- Remove all visible text traces, prioritizing masked regions.",
                "- Keep overall layout semantics, style, and composition coherent.",
                "- Reconstruct the background naturally with coherent texture/lighting.",
                "- Keep canvas size/aspect ratio consistent with the input image.",
                "- Output one final generated image only (no explanation text).",
            ]
        )
        return "\n".join(lines)

    @classmethod
    def _resolve_model_prompt(
        cls,
        *,
        user_prompt: str,
        forbidden_texts: Optional[List[str]] = None,
    ) -> str:
        raw = str(user_prompt or "").strip()
        if raw and cls._looks_like_composed_prompt(raw):
            return raw
        return cls.compose_model_prompt(
            user_prompt=raw or cls.DEFAULT_PROMPT,
            forbidden_texts=forbidden_texts,
        )

    def _resolve(self, base_dir: Path, value: str) -> str:
        """Resolve a path relative to the manifest's directory."""
        p = Path(value)
        if p.is_absolute():
            return str(p)
        return str((base_dir / p).resolve())

    def load_data(
        self,
        data_dir: Union[str, Path],
        *,
        n: Optional[int] = None,
        dataset_root: Union[str, Path],
    ) -> List[Dict[str, Any]]:
        path = Path(data_dir).resolve()
        if path.is_dir():
            json_manifest = path / "text_removal_manifest.json"
            csv_manifest = path / "text_removal_manifest.csv"
            if json_manifest.exists():
                path = json_manifest
            elif csv_manifest.exists():
                path = csv_manifest
            else:
                raise FileNotFoundError(
                    "Text removal manifest not found under directory: "
                    f"{path}. Expected text_removal_manifest.json or text_removal_manifest.csv"
                )
        if not path.exists():
            raise FileNotFoundError(f"Text removal manifest not found: {path}")

        rows = self._load_manifest_rows(path)

        base_dir = path.parent
        samples: List[Dict[str, Any]] = []
        used_sample_ids: set[str] = set()
        for i, row in enumerate(rows):
            if not isinstance(row, dict):
                logger.warning("Invalid sample at index %d (expected object/dict), skipping", i)
                continue

            input_image = self._first_nonempty_value(
                row,
                ("input_image", "masked_image", "image", "image_path"),
            )
            mask = self._first_nonempty_value(
                row,
                ("mask", "text_mask", "mask_path"),
            )
            gt_image = self._first_nonempty_value(
                row,
                ("ground_truth_image", "target_image", "ground_truth", "expected_output"),
            )

            raw_sample_id = (
                self._first_nonempty_value(
                    row,
                    ("sample_id", "id", "layout_id", "source_layout_id", "sampleId"),
                )
            )
            if not raw_sample_id and isinstance(input_image, str):
                raw_sample_id = Path(input_image).stem
            sample_id = str(raw_sample_id).strip() if raw_sample_id else f"text_removal_{i:03d}"
            if not sample_id:
                sample_id = f"text_removal_{i:03d}"

            if not mask:
                inferred_mask = self._infer_mask_path(
                    base_dir=base_dir,
                    sample_id=sample_id,
                    input_image=input_image,
                )
                if inferred_mask:
                    mask = inferred_mask

            if not input_image or not mask or not gt_image:
                logger.warning("Incomplete sample at index %d, skipping", i)
                continue

            forbidden_texts = self._parse_forbidden_texts(
                self._first_nonempty_value(
                    row,
                    ("forbidden_texts", "forbidden_text", "texts"),
                )
            )

            if sample_id in used_sample_ids:
                suffix = 2
                candidate = f"{sample_id}__{suffix}"
                while candidate in used_sample_ids:
                    suffix += 1
                    candidate = f"{sample_id}__{suffix}"
                sample_id = candidate
            used_sample_ids.add(sample_id)

            raw_prompt = str(self._first_nonempty_value(row, ("prompt",)) or "")
            prompt = self._resolve_model_prompt(
                user_prompt=self._decode_prompt_field(raw_prompt),
                forbidden_texts=forbidden_texts,
            )

            samples.append({
                "sample_id": sample_id,
                "ground_truth": {
                    "image": self._resolve(base_dir, gt_image),
                    "mask": self._resolve(base_dir, mask),
                    "forbidden_texts": [str(t) for t in forbidden_texts],
                    "prompt": prompt,
                },
                "input_image": self._resolve(base_dir, input_image),
                "mask": self._resolve(base_dir, mask),
                "forbidden_texts": [str(t) for t in forbidden_texts],
                "prompt": prompt,
            })

        if n is not None:
            samples = samples[:n]
        return samples

    @staticmethod
    def _load_manifest_rows(path: Path) -> List[Dict[str, Any]]:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            with open(path, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    raise ValueError(f"CSV manifest has no header row: {path}")
                rows: List[Dict[str, Any]] = []
                for row in reader:
                    if not isinstance(row, dict):
                        continue
                    cleaned: Dict[str, Any] = {}
                    for key, value in row.items():
                        header = str(key or "").strip()
                        if not header:
                            continue
                        cleaned[header] = value
                    rows.append(cleaned)
            return rows

        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        rows = payload.get("samples") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError(f"Manifest must be a list or dict with 'samples': {path}")
        return rows

    @staticmethod
    def _first_nonempty_value(row: Dict[str, Any], keys: Tuple[str, ...]) -> Any:
        for key in keys:
            if key not in row:
                continue
            value = row.get(key)
            if value is None:
                continue
            if isinstance(value, str):
                text = value.strip()
                if text:
                    return text
                continue
            if isinstance(value, (list, dict)):
                if value:
                    return value
                continue
            return value
        return None

    @staticmethod
    def _parse_forbidden_texts(raw: Any) -> List[str]:
        if raw is None:
            return []
        if isinstance(raw, list):
            return [str(t).strip() for t in raw if str(t).strip()]

        text = str(raw).strip()
        if not text:
            return []

        if text.startswith("["):
            try:
                decoded = json.loads(text)
                if isinstance(decoded, list):
                    return [str(t).strip() for t in decoded if str(t).strip()]
            except Exception:
                pass

        for sep in ("|||", "|", ";", "\n", ","):
            if sep in text:
                values = [part.strip() for part in text.split(sep) if part.strip()]
                if values:
                    return values
        return [text]

    @classmethod
    def _infer_mask_path(
        cls,
        *,
        base_dir: Path,
        sample_id: str,
        input_image: Any,
    ) -> str:
        sid = str(sample_id or "").strip()
        if sid:
            for rel in (
                f"text_removal/mask/{sid}.png",
                f"mask/{sid}.png",
                f"masks/{sid}.png",
            ):
                resolved = cls._resolve_existing_path(base_dir, rel)
                if resolved:
                    return resolved

        raw_input = str(input_image or "").strip()
        if not raw_input:
            return ""

        for src, dst in (
            ("/input/", "/mask/"),
            ("\\input\\", "\\mask\\"),
            ("/masked_layout/", "/mask/"),
            ("\\masked_layout\\", "\\mask\\"),
        ):
            if src in raw_input:
                candidate = raw_input.replace(src, dst)
                resolved = cls._resolve_existing_path(base_dir, candidate)
                if resolved:
                    return resolved
        return ""

    @staticmethod
    def _resolve_existing_path(base_dir: Path, raw: str) -> str:
        text = str(raw or "").strip()
        if not text:
            return ""
        as_path = Path(text)
        if as_path.is_file():
            return str(as_path.resolve())
        rel_path = (base_dir / text).resolve()
        if rel_path.is_file():
            return str(rel_path)
        return ""

    @staticmethod
    def _decode_prompt_field(raw: Any) -> str:
        """Decode one-line CSV escaped newlines into runtime prompt newlines."""
        text = str(raw or "")
        if not text.strip():
            return ""
        text = text.replace("\\r\\n", "\n").replace("\\n", "\n")
        return text.strip()

    def build_model_input(self, sample: Dict[str, Any], *, modality: Any = None) -> Any:
        from gdb.models.base import ModelInput

        prompt = self._resolve_model_prompt(
            user_prompt=str(sample.get("prompt") or ""),
            forbidden_texts=self._parse_forbidden_texts(sample.get("forbidden_texts")),
        )
        metadata: Dict[str, Any] = {
            "mask": sample["mask"],
            "task": "text_removal",
            "benchmark_id": self.meta.id,
            "sample_id": str(sample.get("sample_id") or ""),
        }
        width, height = self._read_image_size(sample.get("input_image"))
        if width > 0 and height > 0:
            metadata["target_width"] = width
            metadata["target_height"] = height

        return ModelInput(
            text=prompt,
            images=[sample["input_image"]],
            metadata=metadata,
        )

    @staticmethod
    def _read_image_size(image_like: Any) -> Tuple[int, int]:
        try:
            from PIL import Image
        except ImportError:
            return (0, 0)

        try:
            if isinstance(image_like, (str, Path)):
                p = Path(image_like)
                if p.exists():
                    with Image.open(p) as img:
                        return int(img.size[0]), int(img.size[1])
            elif isinstance(image_like, (bytes, bytearray)):
                with Image.open(io.BytesIO(image_like)) as img:
                    return int(img.size[0]), int(img.size[1])
            elif isinstance(image_like, Image.Image):
                return int(image_like.size[0]), int(image_like.size[1])
        except Exception:
            return (0, 0)
        return (0, 0)

    def parse_model_output(self, output: Any) -> Any:
        """Return the first generated image, path-like payload, or None."""
        if output is None:
            return None
        images = getattr(output, "images", None)
        if isinstance(images, list) and images:
            return images[0]
        if isinstance(output, dict):
            for key in ("image", "image_path", "prediction", "output_image"):
                if key in output:
                    return output[key]
        if isinstance(output, (str, Path, bytes, bytearray)):
            return output
        return None

    def evaluate(self, predictions: List[Any], ground_truth: List[Any]) -> Dict[str, float]:
        """Evaluate reconstruction quality + OCR-confirmed text absence."""
        from gdb.metrics.core import psnr as metric_psnr
        from gdb.metrics.core import ssim as metric_ssim
        from gdb.tasks.layout import LayerAwareObjectInsertion

        psnr_scores: List[float] = []
        ssim_scores: List[float] = []
        lpips_scores: List[float] = []
        dino_scores: List[float] = []
        clip_scores: List[float] = []
        fid_real_features: List[np.ndarray] = []
        fid_gen_features: List[np.ndarray] = []
        text_absence_scores: List[float] = []
        bbox_absence_scores: List[float] = []
        remove_scores: List[float] = []

        for pred_raw, gt_raw in zip(predictions, ground_truth):
            gt_bundle = self._normalise_gt_bundle(gt_raw)
            pred_image_like = self._extract_image_like(pred_raw)
            gt_image_like = self._extract_image_like(gt_bundle["image"])

            pred_img = self._to_rgb_array(pred_image_like)
            gt_img = self._to_rgb_array(gt_image_like)
            if pred_img is None or gt_img is None:
                continue

            pred_img_native = pred_img.copy()
            pred_img = self._resize_to_match(pred_img, gt_img.shape[:2])

            try:
                psnr_scores.append(float(metric_psnr(pred_img, gt_img)))
            except Exception:
                psnr_scores.append(self._fallback_psnr(pred_img, gt_img))

            try:
                ssim_scores.append(float(metric_ssim(pred_img, gt_img)))
            except Exception:
                ssim_scores.append(self._fallback_ssim(pred_img, gt_img))

            lpips = LayerAwareObjectInsertion._lpips_distance(pred_img, gt_img)
            if isinstance(lpips, float) and math.isfinite(lpips):
                lpips_scores.append(lpips)

            dino = LayerAwareObjectInsertion._dino_similarity(pred_img, gt_img)
            if isinstance(dino, float) and math.isfinite(dino):
                dino_scores.append(dino)

            # Image-6 clip_score is defined as image-image similarity
            # between generated output and ground-truth target.
            clip = LayerAwareObjectInsertion._clip_image_similarity(pred_img, gt_img)
            if isinstance(clip, float) and math.isfinite(clip):
                clip_scores.append(clip)

            real_feat = self._inception_feature(gt_img)
            gen_feat = self._inception_feature(pred_img)
            if real_feat is not None and gen_feat is not None:
                fid_real_features.append(real_feat)
                fid_gen_features.append(gen_feat)

            absence = self._ocr_text_absence_score(
                pred_img,
                gt_bundle["forbidden_texts"],
                gt_bundle["mask"],
            )
            if absence is not None:
                text_absence_scores.append(absence)

            bbox_absence = self._bbox_text_absence_score(
                prediction_image=pred_img,
                forbidden_texts=gt_bundle["forbidden_texts"],
                mask_like=gt_bundle["mask"],
                sample_id=str(gt_bundle.get("sample_id", "")),
            )
            if bbox_absence is not None:
                bbox_absence_scores.append(bbox_absence)

            remove_score = self._remove_score(pred_img_native, gt_bundle["mask"])
            if isinstance(remove_score, float) and math.isfinite(remove_score):
                remove_scores.append(remove_score)

        n = len(psnr_scores) or 1
        fid_score = float("nan")
        if len(fid_real_features) >= 2 and len(fid_gen_features) >= 2:
            try:
                fid_score = float(fid_metric(np.stack(fid_real_features), np.stack(fid_gen_features)))
                # Numerical noise from sqrtm can produce tiny negative values.
                if math.isfinite(fid_score):
                    fid_score = max(0.0, fid_score)
            except Exception:
                fid_score = float("nan")
        return {
            "psnr": sum(psnr_scores) / n,
            "ssim": sum(ssim_scores) / n,
            "lpips": (sum(lpips_scores) / len(lpips_scores) if lpips_scores else float("nan")),
            "lpips_coverage": len(lpips_scores) / n,
            "dino_score": (sum(dino_scores) / len(dino_scores) if dino_scores else float("nan")),
            "dino_coverage": len(dino_scores) / n,
            "clip_score": (sum(clip_scores) / len(clip_scores) if clip_scores else float("nan")),
            "clipscore": (sum(clip_scores) / len(clip_scores) if clip_scores else float("nan")),
            "clip_coverage": len(clip_scores) / n,
            "fid": fid_score,
            "fid_coverage": len(fid_real_features) / n,
            "ocr_text_absence": (
                sum(text_absence_scores) / len(text_absence_scores)
                if text_absence_scores
                else float("nan")
            ),
            "ocr_coverage": len(text_absence_scores) / n,
            "bbox_text_absence": (
                sum(bbox_absence_scores) / len(bbox_absence_scores)
                if bbox_absence_scores
                else float("nan")
            ),
            "bbox_coverage": len(bbox_absence_scores) / n,
            "remove": (sum(remove_scores) / len(remove_scores) if remove_scores else float("nan")),
            "remove_coverage": len(remove_scores) / n,
        }

    @staticmethod
    def _normalise_gt_bundle(raw: Any) -> Dict[str, Any]:
        if isinstance(raw, dict):
            image = raw.get("image", raw.get("ground_truth_image", raw))
            forbidden = raw.get("forbidden_texts") or raw.get("texts") or []
            if isinstance(forbidden, str):
                forbidden = [forbidden]
            mask = raw.get("mask") or raw.get("text_mask")
            prompt = str(raw.get("prompt", ""))
            sample_id = str(raw.get("sample_id", "")).strip()
            return {
                "image": image,
                "forbidden_texts": forbidden,
                "mask": mask,
                "prompt": prompt,
                "sample_id": sample_id,
            }

        return {"image": raw, "forbidden_texts": [], "mask": None, "prompt": "", "sample_id": ""}

    @staticmethod
    def _extract_image_like(value: Any) -> Any:
        if isinstance(value, dict):
            for key in ("image", "output_image", "predicted_image", "path"):
                if key in value:
                    return value[key]

        # Support ModelOutput-like objects without importing model modules.
        images = getattr(value, "images", None)
        if images:
            return images[0]

        return value

    @staticmethod
    def _to_rgb_array(image_like: Any) -> Optional[np.ndarray]:
        if isinstance(image_like, np.ndarray):
            arr = image_like
        else:
            try:
                from PIL import Image
            except ImportError:
                return None

            pil: Optional[Image.Image] = None
            if isinstance(image_like, Image.Image):
                pil = image_like
            elif isinstance(image_like, (str, Path)):
                path_text = str(image_like).strip()
                if not path_text:
                    return None
                path_obj = Path(path_text)
                if not path_obj.exists() or path_obj.is_dir():
                    return None
                pil = Image.open(path_obj)
            elif isinstance(image_like, (bytes, bytearray)):
                pil = Image.open(io.BytesIO(image_like))

            if pil is None:
                return None
            arr = np.asarray(pil.convert("RGB"))

        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        elif arr.ndim != 3:
            return None

        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        return arr

    @staticmethod
    def _resize_to_match(image: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
        if image.shape[:2] == target_hw:
            return image

        try:
            from PIL import Image

            resized = Image.fromarray(image).resize(
                (target_hw[1], target_hw[0]),
                Image.BILINEAR,
            )
            return np.asarray(resized)
        except ImportError:
            return np.resize(image, (target_hw[0], target_hw[1], image.shape[2]))

    @staticmethod
    def _fallback_psnr(pred: np.ndarray, gt: np.ndarray) -> float:
        pred_f = pred.astype(np.float32)
        gt_f = gt.astype(np.float32)
        mse = float(np.mean((pred_f - gt_f) ** 2))
        if mse == 0.0:
            return float("inf")
        return float(20.0 * math.log10(255.0) - 10.0 * math.log10(mse))

    @staticmethod
    def _fallback_ssim(pred: np.ndarray, gt: np.ndarray) -> float:
        # Lightweight fallback when skimage is unavailable.
        pred_f = pred.astype(np.float32)
        gt_f = gt.astype(np.float32)
        mse = float(np.mean((pred_f - gt_f) ** 2))
        return float(max(0.0, 1.0 - (mse / (255.0 ** 2))))

    @staticmethod
    def _normalise_text(raw: str) -> str:
        compact = re.sub(r"[^a-z0-9]+", " ", str(raw).lower())
        return re.sub(r"\s+", " ", compact).strip()

    @classmethod
    def _mask_to_region(cls, image: np.ndarray, mask_like: Any) -> np.ndarray:
        if mask_like is None:
            return image

        mask = cls._to_gray_mask(mask_like, image.shape[:2])
        if mask is None:
            return image

        ys, xs = np.where(mask > 127)
        if ys.size == 0 or xs.size == 0:
            return image

        y1, y2 = int(ys.min()), int(ys.max()) + 1
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        return image[y1:y2, x1:x2]

    @staticmethod
    def _to_gray_mask(mask_like: Any, target_hw: tuple[int, int]) -> Optional[np.ndarray]:
        if isinstance(mask_like, np.ndarray):
            mask = mask_like
        else:
            try:
                from PIL import Image
            except ImportError:
                return None

            pil: Optional[Image.Image] = None
            if isinstance(mask_like, Image.Image):
                pil = mask_like
            elif isinstance(mask_like, (str, Path)):
                if not Path(mask_like).exists():
                    return None
                pil = Image.open(mask_like)
            elif isinstance(mask_like, (bytes, bytearray)):
                pil = Image.open(io.BytesIO(mask_like))

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
            except ImportError:
                mask = np.resize(mask, target_hw)
        return mask.astype(np.uint8)

    @classmethod
    def _ocr_text_absence_score(
        cls,
        prediction_image: np.ndarray,
        forbidden_texts: List[str],
        mask_like: Any,
    ) -> Optional[float]:
        if not forbidden_texts:
            return 1.0

        ocr_text = cls._run_ocr(cls._mask_to_region(prediction_image, mask_like))
        if ocr_text is None:
            return None

        normalised_ocr = cls._normalise_text(ocr_text)
        if not normalised_ocr:
            return 1.0

        for phrase in forbidden_texts:
            needle = cls._normalise_text(phrase)
            if not needle:
                continue

            # Exact phrase check first.
            if needle in normalised_ocr:
                return 0.0

            # Token-level fallback for OCR spacing/punctuation variation.
            tokens = [tok for tok in needle.split(" ") if len(tok) >= 3]
            if tokens and all(tok in normalised_ocr for tok in tokens):
                return 0.0

        return 1.0

    @staticmethod
    def _env_flag_enabled(name: str, default: bool = False) -> bool:
        raw = str(os.environ.get(name, "")).strip().lower()
        if not raw:
            return default
        return raw in {"1", "true", "yes", "on"}

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        raw = str(os.environ.get(name, "")).strip()
        if not raw:
            return default
        try:
            return int(raw)
        except Exception:
            return default

    @staticmethod
    def _box_area(box: Tuple[int, int, int, int]) -> int:
        x1, y1, x2, y2 = box
        return max(0, x2 - x1) * max(0, y2 - y1)

    @classmethod
    def _box_iou(cls, a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
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
    def _prepare_bbox_forbidden_texts(cls, forbidden_texts: List[str]) -> List[str]:
        max_phrases = max(1, cls._env_int(cls.BBOX_TEXT_ABSENCE_MAX_PHRASES_ENV, 10))
        max_chars = max(80, cls._env_int(cls.BBOX_TEXT_ABSENCE_MAX_CHARS_ENV, 1200))
        out: List[str] = []
        seen: set[str] = set()
        char_budget = 0
        for raw in forbidden_texts:
            text = re.sub(r"\s+", " ", str(raw or "")).strip()
            if not text:
                continue
            key = cls._normalise_text(text)
            if not key or key in seen:
                continue
            if char_budget + len(text) > max_chars and out:
                break
            seen.add(key)
            out.append(text)
            char_budget += len(text)
            if len(out) >= max_phrases:
                break
        return out

    @classmethod
    def _bbox_query_text(cls, forbidden_texts: List[str]) -> str:
        lines = [
            "Detect any remaining visible text that matches one of the following forbidden texts.",
            "Return a bbox for one matching occurrence if found.",
            "",
            "Forbidden texts:",
        ]
        for idx, phrase in enumerate(forbidden_texts, start=1):
            lines.append(f"{idx}. {phrase}")
        return "\n".join(lines)

    @classmethod
    def _bbox_text_absence_score(
        cls,
        *,
        prediction_image: np.ndarray,
        forbidden_texts: List[str],
        mask_like: Any,
        sample_id: str = "",
    ) -> Optional[float]:
        if not cls._env_flag_enabled(cls.BBOX_TEXT_ABSENCE_ENABLED_ENV, default=False):
            return None
        prepared = cls._prepare_bbox_forbidden_texts(forbidden_texts)
        if not prepared:
            return 1.0

        try:
            from gdb.tasks.typography import StyledTextGeneration
        except Exception:
            return None

        # Reuse typography-7 bbox detector stack (gpt-5.4 by default).
        if StyledTextGeneration._get_bbox_detector_model() is None:
            return None

        mask = cls._to_gray_mask(mask_like, prediction_image.shape[:2])
        mask_bbox = StyledTextGeneration._mask_bbox(mask)
        query_text = cls._bbox_query_text(prepared)
        bbox = StyledTextGeneration._detect_text_bbox_llm(
            image=prediction_image,
            expected_text=query_text,
            mask_bbox=mask_bbox,
            sample_id=f"image-6|{sample_id}",
        )
        if bbox is None:
            return 1.0
        if mask_bbox is None:
            return 0.0
        # Ignore detections that do not overlap editable text area.
        return 0.0 if cls._box_iou(bbox, mask_bbox) > 0.0 else 1.0

    @staticmethod
    def _run_ocr(image: np.ndarray) -> Optional[str]:
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            return None

        try:
            return str(pytesseract.image_to_string(Image.fromarray(image), config="--psm 6"))
        except Exception:
            return None

    @classmethod
    def _inception_feature(cls, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract commonly used Inception-v3 pool3(2048) feature for FID."""
        if cls._fid_inception_bundle is None:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
                # Prefer pytorch-fid's Inception (common FID implementation).
                try:
                    from pytorch_fid.inception import InceptionV3

                    block = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
                    model = InceptionV3([block]).to(device).eval()
                    cls._fid_inception_bundle = ("pytorch_fid", model, torch, device)
                except Exception:
                    # Fallback: torchvision Inception-v3 pool3 feature.
                    from torchvision.models import Inception_V3_Weights, inception_v3

                    model = inception_v3(
                        weights=Inception_V3_Weights.IMAGENET1K_V1,
                        aux_logits=False,
                    )
                    model.fc = torch.nn.Identity()
                    model = model.to(device).eval()
                    cls._fid_inception_bundle = ("torchvision", model, torch, device)
            except Exception as exc:
                logger.info("Inception FID feature extractor unavailable: %s", exc)
                cls._fid_inception_bundle = False

        if not cls._fid_inception_bundle:
            return None

        mode, model, torch, device = cls._fid_inception_bundle
        try:
            img = image
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            img = np.array(img, copy=True)

            x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            x = torch.nn.functional.interpolate(
                x,
                size=(299, 299),
                mode="bilinear",
                align_corners=False,
            ).to(device)

            with torch.no_grad():
                if mode == "pytorch_fid":
                    feats = model(x)[0]
                    feats = feats.squeeze(-1).squeeze(-1)
                else:
                    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
                    x = (x - mean) / std
                    feats = model(x)

            vec = feats.detach().cpu().numpy().reshape(-1).astype(np.float64)
            if vec.size != 2048:
                return None
            if not np.all(np.isfinite(vec)):
                return None
            return vec
        except Exception:
            return None

    @classmethod
    def _remove_score(cls, prediction_image: np.ndarray, mask_like: Any) -> float:
        if mask_like is None:
            return float("nan")

        if cls._remove_evaluator_bundle is None:
            try:
                from gdb.metrics.remove_metric import (
                    DEFAULT_SAM_CHECKPOINT_PATH,
                    RemoveMetricEvaluator,
                    ensure_sam_checkpoint,
                )

                checkpoint_override = os.environ.get("GDB_REMOVE_SAM_CHECKPOINT")
                checkpoint = ensure_sam_checkpoint(checkpoint_override or DEFAULT_SAM_CHECKPOINT_PATH)
                disable_crop = os.environ.get("GDB_REMOVE_DISABLE_CROP", "")
                crop = str(disable_crop).strip().lower() not in {"1", "true", "yes", "on"}

                cls._remove_evaluator_bundle = RemoveMetricEvaluator(
                    sam_checkpoint=str(checkpoint),
                    model_type=os.environ.get("GDB_REMOVE_MODEL_TYPE", "vit_h"),
                    device=os.environ.get("GDB_REMOVE_DEVICE") or None,
                    crop=crop,
                )
            except Exception as exc:
                logger.info("ReMOVE metric unavailable, returning NaN: %s", exc)
                cls._remove_evaluator_bundle = False

        if not cls._remove_evaluator_bundle:
            return float("nan")

        try:
            from PIL import Image

            pred_u8 = prediction_image
            if pred_u8.dtype != np.uint8:
                pred_u8 = np.clip(pred_u8, 0, 255).astype(np.uint8)

            mask = cls._to_gray_mask(mask_like, pred_u8.shape[:2])
            if mask is None:
                return float("nan")

            pred_pil = Image.fromarray(pred_u8, mode="RGB")
            mask_pil = Image.fromarray(mask.astype(np.uint8), mode="L")
            score = cls._remove_evaluator_bundle.score(pred_pil, mask_pil)
            if score is None:
                return float("nan")
            return float(score)
        except Exception as exc:
            logger.debug("ReMOVE metric failed for a sample: %s", exc)
            return float("nan")
