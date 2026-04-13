"""Layout benchmarks: layout-1 … layout-8 (includes layer-aware object insertion)."""

from __future__ import annotations

import csv
import hashlib
import html
import importlib
import io
import json
import logging
import math
import os
import re
import site
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np

from design_benchmarks.base import BaseBenchmark, BenchmarkMeta, TaskType, benchmark
from design_benchmarks.metrics.core import fid as fid_metric
from design_benchmarks.utils.data_helpers import build_vision_input, load_csv_samples
from design_benchmarks.utils.text_helpers import extract_json_obj

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


@benchmark
class IntentToLayoutGeneration(BaseBenchmark):
    """layout-1 — Generate a flattened layout image from intent text."""

    pipeline_implemented = True

    meta = BenchmarkMeta(
        id="layout-1",
        name="Intent-to-Layout Generation",
        task_type=TaskType.GENERATION,
        domain="layout",
        data_subpath="layout/layout2-intention-to-layout-generation",
        description="Generate a flattened design image from user intent",
        input_spec="Intent prompt + required on-canvas texts",
        output_spec="Rendered layout image",
        metrics=[
            "nima_score",
            "hpsv3",
            "clip_score",
            "pickscore",
            "imagereward",
            "mjudge_win_rate",
            "fid",
            "ocr_readability",
            "color_harmony_index",
        ],
    )

    _clip_bundle: Any = None
    _pickscore_bundle: Any = None
    _imagereward_bundle: Any = None
    _nima_bundle: Any = None
    _hpsv3_bundle: Any = None
    _hpsv2_bundle: Any = None
    _mjudge_bundle: Any = None
    _hps_assets_prepared: bool = False
    LAYOUT2_MANIFEST_ENV = "DESIGN_BENCHMARKS_LAYOUT2_MANIFEST"
    MANIFEST_JSON_FILENAMES = (
        "layout2_manifest.json",
        "intent_to_layout_manifest.json",
    )
    MANIFEST_CSV_FILENAMES = (
        "layout2_manifest.csv",
        "intent_to_layout_manifest.csv",
    )

    def load_data(
        self,
        data_dir: Union[str, Path],
        *,
        n: Optional[int] = None,
        dataset_root: Union[str, Path],
    ) -> List[Dict[str, Any]]:
        root = Path(data_dir).resolve()
        layouts_dir, images_dir, component_renders_dir = self._resolve_data_dirs(data_dir)
        manifest_path = self._resolve_layout2_manifest_path(root)
        if not layouts_dir.is_dir() and manifest_path is None:
            raise FileNotFoundError(f"Layouts directory not found: {layouts_dir}")

        image_index = self._index_images(images_dir)

        if manifest_path is not None:
            samples = self._load_layout2_manifest_samples(
                manifest_path=manifest_path,
                image_index=image_index,
                component_renders_dir=component_renders_dir,
                n=n,
            )
            if samples:
                return samples
            raise ValueError(
                "No valid layout-1 samples found in manifest. "
                f"Check manifest file: {manifest_path}"
            )

        samples: List[Dict[str, Any]] = []
        for layout_file in sorted(layouts_dir.glob("*.json")):
            try:
                row = json.loads(layout_file.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Failed to parse %s: %s", layout_file.name, exc)
                continue

            sample = self._make_sample(
                layout_file=layout_file,
                row=row,
                image_index=image_index,
                component_renders_dir=component_renders_dir,
            )
            if sample is None:
                continue

            samples.append(sample)
            if n is not None and len(samples) >= n:
                break

        if not samples:
            raise ValueError(
                "No valid layout-1 samples found. "
                "Check semantic descriptions and reference images."
            )
        return samples

    def build_model_input(self, sample: Dict[str, Any], *, modality: Any = None) -> Any:
        from design_benchmarks.models.base import ModelInput

        prompt = str(sample.get("prompt") or "").strip() or self._compose_prompt(sample)
        metadata = {
            "benchmark_id": self.meta.id,
            "sub_category": sample.get("sub_category", ""),
            "target_width": sample.get("width", 0),
            "target_height": sample.get("height", 0),
            "target_aspect_ratio": sample.get("aspect_ratio", 1.0),
            "component_render_dir": sample.get("component_render_dir", ""),
        }
        return ModelInput(text=prompt, metadata=metadata)

    def parse_model_output(self, output: Any) -> Any:
        if output is None:
            return None

        images = getattr(output, "images", None)
        if images:
            return images[0]

        text = getattr(output, "text", "")
        if isinstance(text, str):
            cleaned = text.strip()
            cleaned = re.sub(r"^```(?:txt|text)?\s*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```\s*$", "", cleaned).strip()
            if cleaned.startswith(("http://", "https://")):
                return cleaned
            as_path = Path(cleaned)
            if as_path.exists():
                return str(as_path)
        return None

    def evaluate(self, predictions: List[Any], ground_truth: List[Any]) -> Dict[str, float]:
        clip_scores: List[float] = []
        pick_scores: List[float] = []
        imagereward_scores: List[float] = []
        mjudge_scores: List[float] = []
        hpsv3_scores: List[float] = []
        hpsv2_scores: List[float] = []
        nima_scores: List[float] = []
        ocr_scores: List[float] = []
        harmony_scores: List[float] = []
        real_features: List[np.ndarray] = []
        gen_features: List[np.ndarray] = []
        mjudge_attempted = 0

        evaluated = 0
        for pred_raw, gt_raw in zip(predictions, ground_truth):
            gt = self._normalize_gt_bundle(gt_raw)
            prompt = gt["prompt"]

            pred_img = self._to_rgb_array(pred_raw)
            if pred_img is None:
                continue
            evaluated += 1

            gt_img = self._to_rgb_array(gt["image"])
            if gt_img is not None:
                pred_img = self._resize_to_match(pred_img, gt_img.shape[:2])
                real_features.append(self._feature_vector(gt_img))
                gen_features.append(self._feature_vector(pred_img))

            clip = self._clip_score(prompt, pred_img)
            pick = self._pick_score(prompt, pred_img)
            imagereward = self._imagereward_score(prompt, pred_img)
            mjudge = float("nan")
            if gt_img is not None:
                mjudge_attempted += 1
                mjudge = self._mjudge_pairwise_win_rate(
                    prompt=prompt,
                    pred_image=pred_img,
                    gt_image=gt_img,
                    sample_id=str(gt.get("sample_id", "")),
                )
            hpsv3 = self._hpsv3_score(prompt, pred_img, clip_fallback=clip)
            hpsv2 = self._hpsv2_score(prompt, pred_img, clip_fallback=clip)
            nima = self._nima_score(pred_img)
            ocr = self._ocr_readability_score(pred_img, gt["expected_texts"])
            harmony = self._color_harmony_index(pred_img)

            self._append_if_finite(clip_scores, clip)
            self._append_if_finite(pick_scores, pick)
            self._append_if_finite(imagereward_scores, imagereward)
            self._append_if_finite(mjudge_scores, mjudge)
            self._append_if_finite(hpsv3_scores, hpsv3)
            self._append_if_finite(hpsv2_scores, hpsv2)
            self._append_if_finite(nima_scores, nima)
            self._append_if_finite(ocr_scores, ocr)
            self._append_if_finite(harmony_scores, harmony)

        fid_score = float("nan")
        if len(real_features) >= 2:
            try:
                fid_score = float(fid_metric(np.stack(real_features), np.stack(gen_features)))
            except Exception:
                fid_score = float("nan")

        denom = max(evaluated, 1)
        return {
            "nima_score": self._mean_or_nan(nima_scores),
            "hpsv3": self._mean_or_nan(hpsv3_scores),
            "hpsv2": self._mean_or_nan(hpsv2_scores),
            "clip_score": self._mean_or_nan(clip_scores),
            "pickscore": self._mean_or_nan(pick_scores),
            "imagereward": self._mean_or_nan(imagereward_scores),
            "mjudge_win_rate": self._mean_or_nan(mjudge_scores),
            "fid": fid_score,
            "ocr_readability": self._mean_or_nan(ocr_scores),
            "color_harmony_index": self._mean_or_nan(harmony_scores),
            "evaluated_samples": float(evaluated),
            "clip_coverage": len(clip_scores) / denom,
            "ocr_coverage": len(ocr_scores) / denom,
            "mjudge_coverage": len(mjudge_scores) / max(mjudge_attempted, 1),
            "fid_pair_count": float(len(real_features)),
        }

    def _make_sample(
        self,
        *,
        layout_file: Path,
        row: Dict[str, Any],
        image_index: Dict[str, str],
        component_renders_dir: Path,
    ) -> Optional[Dict[str, Any]]:
        sid = layout_file.stem

        semantic = row.get("layout_semantic_description", {})
        if not isinstance(semantic, dict):
            semantic = {}

        intent = self._first_nonempty(
            semantic.get("user_intent"),
            row.get("user_intent"),
        )
        if not intent:
            return None

        image_description = self._first_nonempty(
            semantic.get("description"),
            row.get("description"),
            row.get("image_description"),
        )
        aesthetics = self._first_nonempty(
            semantic.get("aesthetics"),
            row.get("aesthetics"),
        )
        tags = semantic.get("tags")
        if isinstance(tags, list):
            tags = ", ".join(str(v).strip() for v in tags if str(v).strip())
        else:
            tags = str(tags or "").strip()

        sub_category = str(row.get("sub_category", "") or "")

        width = self._safe_int(row.get("layout_metadata", {}).get("width"), 0)
        height = self._safe_int(row.get("layout_metadata", {}).get("height"), 0)
        aspect_ratio = float(width) / float(height) if width > 0 and height > 0 else 1.0

        reference_image = image_index.get(sid) or str(row.get("layout_remotion_image_url", ""))
        if not reference_image:
            return None

        render_dir = component_renders_dir / sid
        component_render_dir = str(render_dir) if render_dir.exists() else ""

        expected_texts = self._extract_texts(row.get("layout_config", {}))
        return {
            "sample_id": sid,
            "intent": intent,
            "image_description": image_description,
            "aesthetics": aesthetics,
            "tags": tags,
            "expected_texts": expected_texts,
            "sub_category": sub_category,
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio,
            "reference_image": reference_image,
            "component_render_dir": component_render_dir,
            "ground_truth": {
                "image": reference_image,
                "prompt": intent,
                "expected_texts": expected_texts,
            },
        }

    def _load_layout2_manifest_samples(
        self,
        *,
        manifest_path: Path,
        image_index: Dict[str, str],
        component_renders_dir: Path,
        n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        rows = self._read_layout2_manifest_rows(manifest_path)
        samples: List[Dict[str, Any]] = []
        for row in rows:
            sample = self._make_manifest_sample(
                row=row,
                root=manifest_path.parent,
                image_index=image_index,
                component_renders_dir=component_renders_dir,
            )
            if sample is None:
                continue
            samples.append(sample)
            if n is not None and len(samples) >= n:
                break
        return samples

    def _make_manifest_sample(
        self,
        *,
        row: Dict[str, Any],
        root: Path,
        image_index: Dict[str, str],
        component_renders_dir: Path,
    ) -> Optional[Dict[str, Any]]:
        sid = self._first_nonempty(row.get("sample_id"))
        if not sid:
            return None

        semantic = row.get("layout_semantic_description", {})
        if not isinstance(semantic, dict):
            semantic = {}

        intent = self._first_nonempty(
            row.get("intent"),
            row.get("prompt_user"),
            semantic.get("user_intent"),
            row.get("user_intent"),
        )
        if not intent:
            return None

        image_description = self._first_nonempty(
            row.get("image_description"),
            semantic.get("description"),
            row.get("description"),
        )
        aesthetics = self._first_nonempty(
            row.get("aesthetics"),
            semantic.get("aesthetics"),
        )
        tags_raw = row.get("tags", semantic.get("tags"))
        if isinstance(tags_raw, list):
            tags = ", ".join(str(v).strip() for v in tags_raw if str(v).strip())
        else:
            tags = str(tags_raw or "").strip()

        sub_category = self._first_nonempty(row.get("sub_category"))

        layout_metadata = row.get("layout_metadata")
        if not isinstance(layout_metadata, dict):
            layout_metadata = {}
        width = self._safe_int(
            row.get("width"),
            self._safe_int(layout_metadata.get("width"), 0),
        )
        height = self._safe_int(
            row.get("height"),
            self._safe_int(layout_metadata.get("height"), 0),
        )
        aspect_ratio = self._safe_float(row.get("aspect_ratio"), 0.0)
        if aspect_ratio <= 0.0:
            aspect_ratio = float(width) / float(height) if width > 0 and height > 0 else 1.0

        reference_image = self._first_nonempty(
            self._resolve_manifest_file_path(root, row.get("reference_image")),
            self._resolve_manifest_file_path(root, row.get("ground_truth_image")),
            image_index.get(sid),
            str(row.get("layout_remotion_image_url", "")),
        )
        if not reference_image:
            return None

        component_render_dir = self._first_nonempty(
            self._resolve_manifest_dir_path(root, row.get("component_render_dir")),
        )
        if not component_render_dir:
            render_dir = component_renders_dir / sid
            if render_dir.exists():
                component_render_dir = str(render_dir.resolve())

        expected_texts = self._normalize_expected_texts(row.get("expected_texts"))
        if not expected_texts:
            expected_texts = self._extract_texts(row.get("layout_config", {}))

        sample = {
            "sample_id": sid,
            "intent": intent,
            "image_description": image_description,
            "aesthetics": aesthetics,
            "tags": tags,
            "expected_texts": expected_texts,
            "sub_category": sub_category,
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio,
            "reference_image": reference_image,
            "component_render_dir": component_render_dir,
            "ground_truth": {
                "image": reference_image,
                "prompt": intent,
                "expected_texts": expected_texts,
            },
        }

        prompt_override = self._first_nonempty(row.get("prompt"))
        if prompt_override:
            sample["prompt"] = prompt_override
        return sample

    @classmethod
    def _read_layout2_manifest_rows(cls, manifest_path: Path) -> List[Dict[str, Any]]:
        suffix = manifest_path.suffix.lower()
        if suffix == ".csv":
            try:
                with open(manifest_path, "r", encoding="utf-8-sig", newline="") as f:
                    reader = csv.DictReader(f)
                    if not reader.fieldnames:
                        raise ValueError(f"CSV manifest has no header row: {manifest_path}")
                    rows = [
                        cls._normalize_layout2_manifest_csv_row(row)
                        for row in reader
                        if isinstance(row, dict)
                    ]
                    return rows
            except Exception as exc:
                raise ValueError(f"Failed to parse CSV manifest {manifest_path}: {exc}") from exc

        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Failed to parse manifest {manifest_path}: {exc}") from exc

        rows = payload.get("samples", []) if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError(
                f"Manifest must be a list (or dict with samples): {manifest_path}"
            )
        return [row for row in rows if isinstance(row, dict)]

    @classmethod
    def _normalize_layout2_manifest_csv_row(cls, row: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(row)

        for key in (
            "sample_id",
            "sub_category",
            "intent",
            "prompt_user",
            "prompt",
            "image_description",
            "aesthetics",
            "tags",
            "reference_image",
            "ground_truth_image",
            "component_render_dir",
        ):
            value = row.get(key)
            if isinstance(value, str):
                out[key] = value.replace("\\r\\n", "\n").replace("\\n", "\n").strip()

        width = cls._safe_int(row.get("width"), cls._safe_int(row.get("canvas_width"), 0))
        height = cls._safe_int(row.get("height"), cls._safe_int(row.get("canvas_height"), 0))
        out["width"] = width
        out["height"] = height

        aspect_ratio = cls._safe_float(row.get("aspect_ratio"), 0.0)
        if aspect_ratio <= 0.0 and width > 0 and height > 0:
            aspect_ratio = float(width) / float(height)
        out["aspect_ratio"] = aspect_ratio if aspect_ratio > 0.0 else 1.0

        expected_raw = row.get("expected_texts")
        parsed_expected = cls._parse_json_cell(expected_raw)
        if isinstance(parsed_expected, list):
            out["expected_texts"] = [str(v).strip() for v in parsed_expected if str(v).strip()]
        elif isinstance(parsed_expected, str):
            text = parsed_expected.strip()
            out["expected_texts"] = [text] if text else []
        elif isinstance(expected_raw, str):
            text = expected_raw.strip()
            out["expected_texts"] = [text] if text else []
        else:
            out["expected_texts"] = []

        for key in ("layout_config", "layout_metadata", "layout_semantic_description"):
            parsed = cls._parse_json_cell(row.get(key))
            if isinstance(parsed, dict):
                out[key] = parsed
        return out

    def _normalize_expected_texts(self, value: Any) -> List[str]:
        if isinstance(value, list):
            return [
                self._normalize_text_constraint(v)
                for v in value
                if self._normalize_text_constraint(v)
            ][:20]

        if isinstance(value, str):
            parsed = self._parse_json_cell(value)
            if isinstance(parsed, list):
                return [
                    self._normalize_text_constraint(v)
                    for v in parsed
                    if self._normalize_text_constraint(v)
                ][:20]
            normalized = self._normalize_text_constraint(value)
            return [normalized] if normalized else []

        return []

    def _resolve_layout2_manifest_path(self, data_root: Path) -> Optional[Path]:
        env_value = str(os.environ.get(self.LAYOUT2_MANIFEST_ENV, "")).strip()
        if env_value:
            override = Path(env_value).expanduser()
            if not override.is_absolute():
                override = (data_root / override).resolve()
            if override.is_file():
                return override
            raise FileNotFoundError(
                f"{self.LAYOUT2_MANIFEST_ENV} points to missing file: {override}"
            )

        for filename in self.MANIFEST_JSON_FILENAMES:
            candidate = data_root / filename
            if candidate.is_file():
                return candidate
        for filename in self.MANIFEST_CSV_FILENAMES:
            candidate = data_root / filename
            if candidate.is_file():
                return candidate
        return None

    @staticmethod
    def _resolve_manifest_file_path(root: Path, value: Any) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""
        if raw.startswith(("http://", "https://")):
            return raw
        as_path = Path(raw)
        if as_path.is_file():
            return str(as_path.resolve())
        candidate = (root / raw).resolve()
        if candidate.is_file():
            return str(candidate)
        return ""

    @staticmethod
    def _resolve_manifest_dir_path(root: Path, value: Any) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""
        as_path = Path(raw)
        if as_path.is_dir():
            return str(as_path.resolve())
        candidate = (root / raw).resolve()
        if candidate.is_dir():
            return str(candidate)
        return ""

    def _compose_prompt(self, sample: Dict[str, Any]) -> str:
        lines = [
            "You are an expert end-to-end layout designer.",
            f"User intent: {sample['intent']}",
        ]
        image_description = str(sample.get("image_description") or "").strip()
        if image_description:
            lines.append(f"Image description: {image_description}")

        aesthetics = str(sample.get("aesthetics") or "").strip()
        if aesthetics:
            lines.append(f"Aesthetic/style cues: {aesthetics}")

        required_texts = sample.get("expected_texts", [])
        if isinstance(required_texts, list) and required_texts:
            lines.append("Required texts to include in the layout (verbatim, legible):")
            for text in required_texts:
                normalized = self._normalize_text_constraint(text)
                if normalized:
                    lines.append(f'- "{normalized}"')
        if sample.get("width") and sample.get("height"):
            lines.append(
                f"Target ratio: {sample['width']}:{sample['height']} "
                f"(~{sample['aspect_ratio']:.3f})."
            )
        lines.extend(
            [
                "",
                "Requirements:",
                "- Produce one cohesive layout image.",
                "- Keep typography readable and hierarchy clear.",
                "- Use a consistent visual and color system.",
                "- Include all required texts with exact spelling.",
                "- Avoid gibberish text artifacts.",
            ]
        )
        return "\n".join(lines)

    def _resolve_data_dirs(self, data_dir: Union[str, Path]) -> Tuple[Path, Path, Path]:
        root = Path(data_dir).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Data path not found: {root}")
        return root / "layouts", root / "images", root / "component_renders"

    @staticmethod
    def _index_images(images_dir: Path) -> Dict[str, str]:
        index: Dict[str, str] = {}
        if not images_dir.is_dir():
            return index
        for path in images_dir.iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                index.setdefault(path.stem, str(path.resolve()))
        return index

    def _extract_texts(self, layout_config: Any) -> List[str]:
        if not isinstance(layout_config, dict):
            return []
        texts: List[str] = []

        def walk(component: Any) -> None:
            if not isinstance(component, dict):
                return
            text = component.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
            for child in component.get("components", []):
                walk(child)

        for top in layout_config.get("components", []):
            walk(top)
        return texts[:20]

    @staticmethod
    def _normalize_gt_bundle(raw: Any) -> Dict[str, Any]:
        if isinstance(raw, dict):
            expected = raw.get("expected_texts", [])
            if isinstance(expected, str):
                expected = [expected]
            return {
                "image": raw.get("image", raw),
                "prompt": str(raw.get("prompt", "")),
                "expected_texts": [str(t) for t in expected],
                "sample_id": str(raw.get("sample_id", raw.get("_sample_id", "")) or ""),
            }
        return {"image": raw, "prompt": "", "expected_texts": [], "sample_id": ""}

    @classmethod
    def _to_rgb_array(cls, image_like: Any) -> Optional[np.ndarray]:
        if isinstance(image_like, np.ndarray):
            arr = image_like
        else:
            try:
                import requests
                from PIL import Image
            except ImportError:
                return None

            pil = None
            if isinstance(image_like, Image.Image):
                pil = image_like
            elif isinstance(image_like, (bytes, bytearray)):
                pil = Image.open(io.BytesIO(image_like))
            elif isinstance(image_like, (str, Path)):
                source = str(image_like)
                if source.startswith(("http://", "https://")):
                    try:
                        resp = requests.get(source, timeout=20)
                        resp.raise_for_status()
                        pil = Image.open(io.BytesIO(resp.content))
                    except Exception:
                        return None
                else:
                    source = source.strip()
                    if not source:
                        return None
                    p = Path(source)
                    if p.is_file():
                        try:
                            pil = Image.open(p)
                        except Exception:
                            return None
            if pil is None:
                return None
            arr = np.asarray(pil.convert("RGB"))

        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        elif arr.ndim != 3:
            return None
        return np.clip(arr, 0, 255).astype(np.uint8)

    @staticmethod
    def _resize_to_match(image: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
        if image.shape[:2] == target_hw:
            return image
        try:
            from PIL import Image

            resized = Image.fromarray(image).resize((target_hw[1], target_hw[0]), Image.BILINEAR)
            return np.asarray(resized)
        except ImportError:
            return np.resize(image, (target_hw[0], target_hw[1], image.shape[2]))

    @staticmethod
    def _truthy_env(value: Optional[str], default: bool = False) -> bool:
        if value is None:
            return default
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

    @classmethod
    def _mjudge_enabled(cls) -> bool:
        flag = os.environ.get("DESIGN_BENCHMARKS_ENABLE_MJUDGE")
        configured_model = str(os.environ.get("DESIGN_BENCHMARKS_MJUDGE_MODEL", "")).strip()
        if flag is None:
            return bool(configured_model)
        return cls._truthy_env(flag, default=False)

    @classmethod
    def _load_mjudge_bundle(cls) -> Any:
        if cls._mjudge_bundle is not None:
            return cls._mjudge_bundle

        if not cls._mjudge_enabled():
            cls._mjudge_bundle = False
            return cls._mjudge_bundle

        model_spec = str(
            os.environ.get(
                "DESIGN_BENCHMARKS_MJUDGE_MODEL",
                "gemini:gemini-3.1-flash-lite-preview",
            )
        ).strip()
        if ":" not in model_spec:
            logger.info("Invalid DESIGN_BENCHMARKS_MJUDGE_MODEL=%r", model_spec)
            cls._mjudge_bundle = False
            return cls._mjudge_bundle

        provider_raw, model_id = model_spec.split(":", 1)
        provider = provider_raw.strip().lower()
        model_id = model_id.strip()
        provider_aliases = {"gemini": "google", "google": "google"}
        provider = provider_aliases.get(provider, provider)
        if not provider or not model_id:
            cls._mjudge_bundle = False
            return cls._mjudge_bundle

        kwargs: Dict[str, Any] = {
            "model_id": model_id,
            "temperature": 0.0,
            "max_tokens": max(32, cls._safe_int(os.environ.get("DESIGN_BENCHMARKS_MJUDGE_MAX_TOKENS"), 128)),
        }
        credentials_path = str(os.environ.get("DESIGN_BENCHMARKS_MJUDGE_CREDENTIALS", "")).strip()
        if credentials_path and provider == "google":
            kwargs["credentials_path"] = credentials_path
        elif credentials_path and provider == "openai":
            try:
                creds = json.loads(Path(credentials_path).read_text(encoding="utf-8"))
                api_key = str(creds.get("api_key", "")).strip()
                if api_key:
                    kwargs["api_key"] = api_key
            except Exception:
                pass

        try:
            from design_benchmarks.models import load_model

            judge_model = load_model(provider, **kwargs)
            cls._mjudge_bundle = (judge_model, model_spec)
        except Exception as exc:
            logger.info("mjudge model unavailable, metric will be NaN: %s", exc)
            cls._mjudge_bundle = False
        return cls._mjudge_bundle

    @staticmethod
    def _mjudge_prompt(intent: str) -> str:
        lines = [
            "You are a visual language model designed to evaluate and rate visual templates.",
            "You are presented with 2 visual templates.",
            "The first attached image is image_1 and the second attached image is image_2.",
            "Choose the better template using these criteria:",
            "- Aesthetics: visual appeal and balance.",
            "- Clarity: readability and communication clarity.",
            "- Usability: practical and user-friendly arrangement.",
            "- Creativity: uniqueness and design originality.",
            "- Consistency: coherence with design principles and standards.",
        ]
        clean_intent = re.sub(r"\s+", " ", str(intent or "")).strip()
        if clean_intent:
            lines.extend(
                [
                    "",
                    f"Context intent: {clean_intent}",
                ]
            )
        lines.extend(
            [
                "",
                "Return ONLY strict JSON with no explanation:",
                '{"better_layout": "image_1"}',
                "or",
                '{"better_layout": "image_2"}',
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _mjudge_choice_normalize(value: Any) -> Optional[str]:
        text = str(value or "").strip().lower()
        text = text.replace("-", "_").replace(" ", "_")
        if text in {"image_1", "image1", "1", "first", "left"}:
            return "image_1"
        if text in {"image_2", "image2", "2", "second", "right"}:
            return "image_2"
        return None

    @classmethod
    def _parse_mjudge_choice(cls, raw_text: str) -> Optional[str]:
        text = str(raw_text or "").strip()
        if not text:
            return None
        text = re.sub(r"^```(?:json|JSON)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text).strip()

        parsed: Optional[Any] = None
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None
        if not isinstance(parsed, dict):
            decoder = json.JSONDecoder()
            for idx, ch in enumerate(text):
                if ch not in "{[":
                    continue
                try:
                    parsed, _ = decoder.raw_decode(text[idx:])
                    break
                except Exception:
                    continue
        if isinstance(parsed, dict):
            choice = cls._mjudge_choice_normalize(parsed.get("better_layout", ""))
            if choice is not None:
                return choice

        m = re.search(r"better_layout[^a-z0-9]*(image[_\\s-]*[12]|[12])", text, re.IGNORECASE)
        if m:
            choice = cls._mjudge_choice_normalize(m.group(1))
            if choice is not None:
                return choice

        m = re.search(r"\bimage[_\\s-]*([12])\b", text, re.IGNORECASE)
        if m:
            return cls._mjudge_choice_normalize(f"image_{m.group(1)}")
        return None

    @staticmethod
    def _png_bytes_from_array(image: Optional[np.ndarray]) -> Optional[bytes]:
        if image is None:
            return None
        try:
            from PIL import Image

            arr = np.clip(image, 0, 255).astype(np.uint8)
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="PNG")
            return buf.getvalue()
        except Exception:
            return None

    @classmethod
    def _mjudge_pairwise_win_rate(
        cls,
        *,
        prompt: str,
        pred_image: Optional[np.ndarray],
        gt_image: Optional[np.ndarray],
        sample_id: str = "",
    ) -> float:
        bundle = cls._load_mjudge_bundle()
        if not bundle:
            return float("nan")

        pred_png = cls._png_bytes_from_array(pred_image)
        gt_png = cls._png_bytes_from_array(gt_image)
        if pred_png is None or gt_png is None:
            return float("nan")

        # Deterministic image-order randomization reduces position bias.
        hash_src = f"{sample_id}|{prompt}|{pred_image.shape[:2] if pred_image is not None else ''}"
        flip = (hashlib.sha256(hash_src.encode("utf-8")).digest()[0] % 2) == 1
        if flip:
            images = [gt_png, pred_png]
            pred_slot = "image_2"
        else:
            images = [pred_png, gt_png]
            pred_slot = "image_1"

        judge_model = bundle[0]
        try:
            from design_benchmarks.models.base import ModelInput

            model_output = judge_model.predict(
                ModelInput(
                    text=cls._mjudge_prompt(prompt),
                    images=images,
                    metadata={
                        "benchmark_id": "layout-mjudge",
                        "task": "pairwise_aesthetic_judgment",
                    },
                )
            )
        except Exception:
            return float("nan")

        choice = cls._parse_mjudge_choice(getattr(model_output, "text", ""))
        if choice is None:
            return float("nan")
        return 1.0 if choice == pred_slot else 0.0

    @classmethod
    def _clip_score(cls, prompt: str, image: np.ndarray) -> float:
        if not prompt:
            return float("nan")

        if cls._clip_bundle is None:
            try:
                import torch
                from transformers import CLIPModel, CLIPProcessor

                device = "cuda" if torch.cuda.is_available() else "cpu"
                processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
                cls._clip_bundle = (model, processor, torch, device)
            except Exception as exc:
                logger.info("CLIP unavailable, metric will be NaN: %s", exc)
                cls._clip_bundle = False

        if not cls._clip_bundle:
            return float("nan")

        model, processor, torch, device = cls._clip_bundle
        try:
            from PIL import Image

            pil = Image.fromarray(image)
            inputs = processor(text=[prompt], images=[pil], return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs)
                txt = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
                img = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
                return float((txt * img).sum().item())
        except Exception:
            return float("nan")

    @classmethod
    def _pick_score(cls, prompt: str, image: np.ndarray) -> float:
        if not prompt:
            return float("nan")

        if cls._pickscore_bundle is None:
            try:
                import torch
                from transformers import AutoModel, AutoProcessor

                device = "cuda" if torch.cuda.is_available() else "cpu"
                # Official PickScore usage uses CLIP-H processor + PickScore checkpoint.
                processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
                model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").to(device).eval()
                cls._pickscore_bundle = (model, processor, torch, device)
            except Exception as exc:
                logger.info("PickScore unavailable, metric will be NaN: %s", exc)
                cls._pickscore_bundle = False

        if not cls._pickscore_bundle:
            return float("nan")

        model, processor, torch, device = cls._pickscore_bundle
        try:
            from PIL import Image

            pil = Image.fromarray(image)
            image_inputs = processor(
                images=[pil],
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(device)
            text_inputs = processor(
                text=[prompt],
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                image_embs = model.get_image_features(**image_inputs)
                image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

                text_embs = model.get_text_features(**text_inputs)
                text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

                scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
                return float(scores[0].item())
        except Exception:
            return float("nan")

    @classmethod
    def _imagereward_score(cls, prompt: str, image: np.ndarray) -> float:
        if not prompt:
            return float("nan")

        if cls._imagereward_bundle is None:
            try:
                import torch

                model = cls._load_imagereward_model(torch=torch)
                cls._imagereward_bundle = (model,)
            except Exception as exc:
                logger.info("ImageReward unavailable, metric will be NaN: %s", exc)
                cls._imagereward_bundle = False

        if not cls._imagereward_bundle:
            return float("nan")

        model = cls._imagereward_bundle[0]
        try:
            from PIL import Image

            pil = Image.fromarray(image)
            score = model.score(prompt, pil)
            if isinstance(score, (list, tuple)):
                if not score:
                    return float("nan")
                score = score[0]
            if hasattr(score, "item"):
                score = score.item()
            return float(score)
        except Exception:
            return float("nan")

    @classmethod
    def _load_imagereward_model(cls, torch: Any) -> Any:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Preferred path when ImageReward package import works as-is.
        try:
            import ImageReward as RM  # type: ignore[reportMissingImports]

            return RM.load("ImageReward-v1.0", device=device)
        except Exception:
            pass

        # Fallback: bypass ImageReward __init__ side effects (ReFL/diffusers import),
        # then import ImageReward.utils directly and call its official loader.
        pkg_dir = cls._find_imagereward_pkg_dir()
        if pkg_dir is None:
            raise ImportError("ImageReward package directory not found in site-packages.")

        sys.modules.pop("ImageReward", None)
        fake_pkg = types.ModuleType("ImageReward")
        fake_pkg.__path__ = [str(pkg_dir)]
        sys.modules["ImageReward"] = fake_pkg

        utils_mod = importlib.import_module("ImageReward.utils")
        if not hasattr(utils_mod, "load"):
            raise ImportError("ImageReward.utils.load is unavailable.")
        return utils_mod.load("ImageReward-v1.0", device=device)

    @staticmethod
    def _find_imagereward_pkg_dir() -> Optional[Path]:
        search_roots: List[str] = []
        try:
            search_roots.extend(site.getsitepackages())
        except Exception:
            pass
        try:
            user_site = site.getusersitepackages()
            if isinstance(user_site, str):
                search_roots.append(user_site)
            else:
                search_roots.extend(user_site)
        except Exception:
            pass

        for root in search_roots:
            cand = Path(root) / "ImageReward"
            if (cand / "utils.py").is_file():
                return cand
        return None

    @classmethod
    def _hpsv3_score(cls, prompt: str, image: np.ndarray, clip_fallback: float) -> float:
        if cls._hpsv3_bundle is None:
            try:
                import hpsv3  # type: ignore[reportMissingImports]
                import torch

                preferred_device = os.environ.get("DESIGN_BENCHMARKS_HPS_DEVICE")
                if not preferred_device:
                    preferred_device = "cuda" if torch.cuda.is_available() else "cpu"

                device_candidates = [preferred_device]
                if preferred_device == "cuda":
                    device_candidates.append("cpu")

                inferencer = None
                last_error: Optional[Exception] = None
                used_device = preferred_device
                for device in device_candidates:
                    try:
                        inferencer = hpsv3.HPSv3RewardInferencer(device=device)
                        used_device = device
                        break
                    except Exception as exc_device:
                        last_error = exc_device
                        continue

                if inferencer is None:
                    raise RuntimeError(
                        f"HPSv3 inferencer initialization failed on devices={device_candidates}: {last_error}"
                    )

                cls._hpsv3_bundle = ("hpsv3", inferencer, used_device)
            except Exception as exc_v3:
                logger.info("HPSv3 unavailable, fallback to CLIP: %s", exc_v3)
                cls._hpsv3_bundle = False

        if not cls._hpsv3_bundle:
            return clip_fallback

        backend = cls._hpsv3_bundle[0]
        scorer = cls._hpsv3_bundle[1]
        try:
            from PIL import Image

            pil = Image.fromarray(image)
            if backend != "hpsv3":
                return clip_fallback

            rewards = scorer.reward([pil], [str(prompt or "")])
            first: Any = None
            if hasattr(rewards, "detach"):
                arr = rewards.detach().cpu().numpy().reshape(-1)
                if arr.size:
                    first = float(arr[0])
            elif isinstance(rewards, (list, tuple)) and rewards:
                first = rewards[0]
                try:
                    first = float(first[0].item())
                except Exception:
                    arr = np.asarray(first).reshape(-1)
                    if arr.size:
                        first = float(arr[0])
            elif rewards is not None:
                arr = np.asarray(rewards).reshape(-1)
                if arr.size:
                    first = float(arr[0])

            if first is not None and math.isfinite(float(first)):
                return float(first)
            return clip_fallback
        except Exception:
            return clip_fallback

    @classmethod
    def _hpsv2_score(cls, prompt: str, image: np.ndarray, clip_fallback: float) -> float:
        if cls._hpsv2_bundle is None:
            try:
                import hpsv2  # type: ignore[reportMissingImports]

                cls._ensure_hpsv2_assets(hpsv2)
                cls._hpsv2_bundle = ("hpsv2", hpsv2)
            except Exception as exc_v2:
                logger.info("HPSv2 unavailable, fallback to CLIP: %s", exc_v2)
                cls._hpsv2_bundle = False

        if not cls._hpsv2_bundle:
            return clip_fallback

        scorer = cls._hpsv2_bundle[1]
        try:
            from PIL import Image

            pil = Image.fromarray(image)
            score = scorer.score(pil, prompt, hps_version="v2.1")
            if isinstance(score, (list, tuple)) and score:
                score = score[0]
            score_f = float(score)
            return score_f if math.isfinite(score_f) else clip_fallback
        except Exception:
            return clip_fallback

    @classmethod
    def _ensure_hpsv2_assets(cls, hpsv2_module: Any) -> None:
        if cls._hps_assets_prepared:
            return
        cls._hps_assets_prepared = True

        module_file = getattr(hpsv2_module, "__file__", "")
        if not module_file:
            return

        target = Path(module_file).resolve().parent / "src" / "open_clip" / "bpe_simple_vocab_16e6.txt.gz"
        if target.is_file():
            return

        url = "https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            req = Request(url, headers={"User-Agent": "lica-bench"})
            with urlopen(req, timeout=30) as resp:
                data = resp.read()
            if not data:
                raise RuntimeError("Downloaded empty HPSv2 tokenizer vocab.")
            target.write_bytes(data)
            logger.info("Prepared HPSv2 tokenizer vocab at %s", target)
        except Exception as exc:
            logger.info("Failed to prepare HPSv2 assets: %s", exc)

    @classmethod
    def _nima_score(cls, image: np.ndarray) -> float:
        if cls._nima_bundle is None:
            try:
                import pyiqa  # type: ignore[reportMissingImports]
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
                metric = pyiqa.create_metric("nima", device=device)
                cls._nima_bundle = (metric, torch, device)
            except Exception as exc:
                logger.info("NIMA backend unavailable, using proxy: %s", exc)
                cls._nima_bundle = False

        if not cls._nima_bundle:
            return cls._aesthetic_proxy(image)

        metric, torch, device = cls._nima_bundle
        try:
            tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            tensor = tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                return float(metric(tensor).item())
        except Exception:
            return cls._aesthetic_proxy(image)

    @staticmethod
    def _feature_vector(image: np.ndarray) -> np.ndarray:
        # Lightweight RGB histogram feature for robust FID fallback.
        bins = 16
        arr = image.astype(np.float32)
        feats: List[np.ndarray] = []
        for c in range(3):
            h, _ = np.histogram(arr[:, :, c], bins=bins, range=(0, 255))
            h = h.astype(np.float64)
            h /= max(h.sum(), 1.0)
            feats.append(h)
        return np.concatenate(feats, axis=0)

    @staticmethod
    def _aesthetic_proxy(image: np.ndarray) -> float:
        x = image.astype(np.float32) / 255.0
        gray = (0.299 * x[:, :, 0]) + (0.587 * x[:, :, 1]) + (0.114 * x[:, :, 2])
        contrast = float(np.clip(np.std(gray) / 0.25, 0.0, 1.0))
        gx = np.abs(np.diff(gray, axis=1))
        gy = np.abs(np.diff(gray, axis=0))
        sharpness = float(np.clip((gx.mean() + gy.mean()) / 0.2, 0.0, 1.0))
        sat = float(np.clip(np.mean(np.max(x, axis=2) - np.min(x, axis=2)) / 0.5, 0.0, 1.0))
        return 1.0 + (9.0 * ((0.35 * contrast) + (0.35 * sharpness) + (0.30 * sat)))

    @classmethod
    def _ocr_readability_score(cls, image: np.ndarray, expected_texts: List[str]) -> float:
        if not expected_texts:
            return float("nan")
        try:
            import pytesseract  # type: ignore[reportMissingImports]
            from PIL import Image

            text = str(pytesseract.image_to_string(Image.fromarray(image), config="--psm 6"))
        except Exception:
            return float("nan")

        gt_tokens = cls._tokenize(" ".join(expected_texts))
        if not gt_tokens:
            return float("nan")
        pred_tokens = cls._tokenize(text)
        if not pred_tokens:
            return 0.0
        return len(gt_tokens & pred_tokens) / len(gt_tokens)

    @classmethod
    def _color_harmony_index(cls, image: np.ndarray) -> float:
        try:
            from PIL import Image
        except ImportError:
            return float("nan")

        hsv = np.asarray(Image.fromarray(image).convert("HSV")).astype(np.float32)
        hue = hsv[:, :, 0] * (360.0 / 255.0)
        sat = hsv[:, :, 1] / 255.0
        val = hsv[:, :, 2] / 255.0
        mask = (sat > 0.20) & (val > 0.15)
        if mask.sum() < 100:
            return float("nan")

        hist, _ = np.histogram(hue[mask], bins=36, range=(0.0, 360.0))
        top = sorted([(b + 0.5) * 10.0 for b in np.argsort(hist)[-3:]])
        if len(top) < 2:
            return float("nan")

        def d(a: float, b: float) -> float:
            x = abs(a - b) % 360.0
            return min(x, 360.0 - x)

        analogous = math.exp(-((d(top[-1], top[-2]) / 35.0) ** 2))
        complementary = 0.0
        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                complementary = max(complementary, math.exp(-(((d(top[i], top[j]) - 180) / 30) ** 2)))
        return float(max(analogous, complementary))

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        cleaned = re.sub(r"[^a-z0-9]+", " ", text.lower())
        return {tok for tok in cleaned.split() if len(tok) >= 3}

    @staticmethod
    def _normalize_text_constraint(value: Any) -> str:
        text = str(value).strip()
        if not text:
            return ""
        return re.sub(r"\s+", " ", text)

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(float(value))
        except Exception:
            return default

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
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

    @staticmethod
    def _first_nonempty(*values: Any) -> str:
        for value in values:
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    @staticmethod
    def _append_if_finite(bucket: List[float], value: float) -> None:
        if isinstance(value, float) and math.isfinite(value):
            bucket.append(value)

    @staticmethod
    def _mean_or_nan(values: List[float]) -> float:
        if not values:
            return float("nan")
        return float(sum(values) / len(values))


@benchmark
class PartialLayoutCompletion(IntentToLayoutGeneration):
    """layout-2 — Top-layer single-element placement from visual components."""

    pipeline_implemented = True

    meta = BenchmarkMeta(
        id="layout-2",
        name="Visual-Element-Contents-Aware Layout Generation (Single)",
        task_type=TaskType.GENERATION,
        domain="layout",
        data_subpath="layout/layout-3-partial-layout-completion",
        description=(
            "Predict placement of one random top-layer element from provided component content"
        ),
        input_spec="Selected top-layer component asset + intent + canvas size",
        output_spec="Layout config JSON (single component bbox) + composited layout quality",
        metrics=[
            "miou",
            "component_coverage",
            "mjudge_win_rate",
            "clip_score",
            "dino_score",
            "lpips",
            "dreamsim_distance",
            "nima_score",
            "hpsv3",
            "imagereward",
            "fid",
        ],
    )

    PLACEMENT_MODE = "single"
    TOP_LAYER_SCAN = 0  # 0 means all top-level components
    MIN_COMPONENT_AREA_RATIO = 0.0
    BACKGROUND_AREA_RATIO = 0.85
    JSON_ALPHA_IOU_AGREE = 0.90
    MIN_ALPHA_PIXELS = 9
    _translate_re = re.compile(
        r"translate\(\s*([-+]?\d*\.?\d+(?:e[-+]?\d+)?)px(?:\s*,\s*([-+]?\d*\.?\d+(?:e[-+]?\d+)?)px)?",
        re.IGNORECASE,
    )
    _translate_x_re = re.compile(
        r"translateX\(\s*([-+]?\d*\.?\d+(?:e[-+]?\d+)?)px",
        re.IGNORECASE,
    )
    _translate_y_re = re.compile(
        r"translateY\(\s*([-+]?\d*\.?\d+(?:e[-+]?\d+)?)px",
        re.IGNORECASE,
    )
    _num_re = re.compile(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", re.IGNORECASE)
    _component_key_fuzzy_re = re.compile(
        r"[\"']component\s*_?\s*key[\"']\s*:\s*[\"']([^\"']+)[\"']",
        re.IGNORECASE,
    )
    _bbox_fuzzy_re = re.compile(
        r"[\"']bbox[\"']\s*:\s*\[([^\]]+)\]",
        re.IGNORECASE | re.DOTALL,
    )
    _html_tag_re = re.compile(r"<[^>]+>")
    _placeholder_desc_re = re.compile(r"^top-layer component\s+\d+$", re.IGNORECASE)
    _json_decoder = json.JSONDecoder()
    DEFAULT_PROMPT = (
        "Create a coherent layout by arranging the provided visual components "
        "within the target canvas."
    )
    KNOWN_MANIFEST_ID_PREFIXES = ("G3_",)

    def load_data(
        self,
        data_dir: Union[str, Path],
        *,
        n: Optional[int] = None,
        dataset_root: Union[str, Path],
    ) -> List[Dict[str, Any]]:
        root = Path(data_dir).resolve()
        single_manifest_json = root / "layout_single_manifest.json"
        single_manifest_csv = root / "layout_single_manifest.csv"
        multiple_manifest_json = root / "layout_multiple_manifest.json"
        multiple_manifest_csv = root / "layout_multiple_manifest.csv"
        has_single = single_manifest_json.is_file() or single_manifest_csv.is_file()
        has_multiple = multiple_manifest_json.is_file() or multiple_manifest_csv.is_file()
        if has_single or has_multiple:
            mode_env = os.environ.get("DESIGN_BENCHMARKS_PARTIAL_MODE", "all").strip().lower()
            if mode_env not in {"all", "single", "multiple"}:
                logger.warning(
                    "Invalid DESIGN_BENCHMARKS_PARTIAL_MODE=%r. Falling back to 'all'.",
                    mode_env,
                )
                mode_env = "all"

            selected: List[Tuple[Path, str]] = []
            if mode_env in {"all", "single"} and has_single:
                single_manifest = single_manifest_json if single_manifest_json.is_file() else single_manifest_csv
                selected.append((single_manifest, "single"))
            if mode_env in {"all", "multiple"} and has_multiple:
                multiple_manifest = (
                    multiple_manifest_json if multiple_manifest_json.is_file() else multiple_manifest_csv
                )
                selected.append((multiple_manifest, "multiple"))
            if not selected:
                raise ValueError(
                    "Top-layer manifests exist, but none match requested mode. "
                    f"mode={mode_env}, single={has_single}, multiple={has_multiple}"
                )

            pools: List[List[Dict[str, Any]]] = []
            for manifest_path, mode_name in selected:
                pool = self._load_top_layer_manifest(
                    root=root,
                    manifest_path=manifest_path,
                    n=n,
                    default_mode=mode_name,
                )
                pools.append(pool)

            merged: List[Dict[str, Any]] = []
            # Interleave pools so small n still captures both modes.
            while any(pools):
                for pool in pools:
                    if not pool:
                        continue
                    merged.append(pool.pop(0))
                    if n is not None and len(merged) >= n:
                        return merged
            return merged

        layouts_dir, images_dir, component_renders_dir = self._resolve_data_dirs(data_dir)
        if not layouts_dir.is_dir():
            raise FileNotFoundError(f"Layouts directory not found: {layouts_dir}")

        image_index = self._index_images(images_dir)
        samples: List[Dict[str, Any]] = []

        for layout_file in sorted(layouts_dir.glob("*.json")):
            try:
                row = json.loads(layout_file.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Failed to parse %s: %s", layout_file.name, exc)
                continue

            sample = self._make_visual_sample(
                layout_file=layout_file,
                row=row,
                image_index=image_index,
                component_renders_dir=component_renders_dir,
            )
            if sample is None:
                continue

            samples.append(sample)
            if n is not None and len(samples) >= n:
                break

        if not samples:
            raise ValueError(
                "No valid layout-2 samples found. "
                "Check component renders and layout JSON structure."
            )
        return samples

    def build_model_input(self, sample: Dict[str, Any], *, modality: Any = None) -> Any:
        from design_benchmarks.models.base import Modality, ModelInput

        images: List[str] = []
        if modality != Modality.TEXT:
            base_image = str(sample.get("input_composite") or "").strip()
            if base_image:
                images.append(base_image)
            images.extend(c["image_path"] for c in sample.get("components", []))

        prompt = self._compose_component_layout_prompt(sample, with_images=bool(images))
        sample_mode = str(sample.get("placement_mode", self.PLACEMENT_MODE)).lower().strip()
        metadata = {
            "benchmark_id": self.meta.id,
            "target_width": sample.get("canvas_width", 0),
            "target_height": sample.get("canvas_height", 0),
            "component_keys": [c["component_key"] for c in sample.get("components", [])],
            "placement_mode": sample_mode,
            "has_input_composite": bool(sample.get("input_composite")),
            "task": "visual_element_contents_aware_layout_generation",
        }
        return ModelInput(text=prompt, images=images, metadata=metadata)

    def parse_model_output(self, output: Any) -> Any:
        if output is None:
            return {"components": []}

        if isinstance(output, dict):
            components = self._extract_predicted_components(output)
            return {"components": components}

        text = getattr(output, "text", "")
        if not isinstance(text, str):
            text = str(text or "")
        cleaned = self._strip_code_fence(text)

        payload = self._decode_json_like(cleaned)
        if payload is not None:
            components = self._extract_predicted_components(payload)
            if components:
                return {"components": components}

        # Truncated responses (e.g., MAX_TOKENS) may still include valid early
        # component bbox entries. Recover them instead of returning empty.
        recovered = self._salvage_components_from_fragment(cleaned)
        if recovered:
            return {"components": recovered}
            return {"components": []}

    def evaluate(self, predictions: List[Any], ground_truth: List[Any]) -> Dict[str, float]:
        from design_benchmarks.metrics.core import iou as iou_metric

        miou_scores: List[float] = []
        coverage_scores: List[float] = []
        clip_scores: List[float] = []
        dino_scores: List[float] = []
        lpips_scores: List[float] = []
        dreamsim_scores: List[float] = []
        nima_scores: List[float] = []
        mjudge_scores: List[float] = []
        hpsv3_scores: List[float] = []
        hpsv2_scores: List[float] = []
        imagereward_scores: List[float] = []
        real_features: List[np.ndarray] = []
        gen_features: List[np.ndarray] = []
        mjudge_attempted = 0

        evaluated = 0
        for pred_raw, gt_raw in zip(predictions, ground_truth):
            gt = self._normalize_visual_gt(gt_raw)
            gt_components = gt.get("components", [])
            if not gt_components:
                continue

            pred_map, pred_order = self._normalize_prediction_map(
                pred_raw,
                canvas_width=gt["canvas_width"],
                canvas_height=gt["canvas_height"],
            )

            ious: List[float] = []
            matched = 0
            rendered_boxes: Dict[str, List[float]] = {}
            for idx, comp in enumerate(gt_components):
                pbox = (
                    pred_map.get(comp["component_key"])
                    or pred_map.get(comp["component_id"])
                    or (pred_order[idx] if idx < len(pred_order) else None)
                )
                if pbox is None:
                    ious.append(0.0)
                    continue
                matched += 1
                ious.append(float(iou_metric(pbox, comp["bbox"])))
                rendered_boxes[comp["component_key"]] = pbox

            if ious:
                miou_scores.append(float(sum(ious) / len(ious)))
                coverage_scores.append(float(matched / len(gt_components)))
                evaluated += 1

            pred_render = self._render_layout_from_boxes(
                gt=gt,
                component_boxes=rendered_boxes,
                fallback_order=pred_order,
            )
            gt_render = self._to_rgb_array(gt.get("ground_truth_image", ""))
            if gt_render is None:
                gt_render = self._render_layout_from_boxes(
                    gt=gt,
                    component_boxes={c["component_key"]: c["bbox"] for c in gt_components},
                    fallback_order=[],
                )
            if pred_render is None or gt_render is None:
                continue

            pred_render = self._resize_to_match(pred_render, gt_render.shape[:2])
            self._maybe_save_layout3_renders(
                gt=gt,
                pred_raw=pred_raw,
                pred_map=pred_map,
                pred_order=pred_order,
                rendered_boxes=rendered_boxes,
                pred_render=pred_render,
                gt_render=gt_render,
            )
            real_features.append(self._feature_vector(gt_render))
            gen_features.append(self._feature_vector(pred_render))

            prompt = str(gt.get("prompt", ""))
            clip = self._clip_score(prompt, pred_render)
            dino = LayerAwareObjectInsertion._dino_similarity(pred_render, gt_render)
            lpips = LayerAwareObjectInsertion._lpips_distance(pred_render, gt_render)
            dreamsim = LayerAwareObjectInsertion._dreamsim_distance(pred_render, gt_render)
            nima = self._nima_score(pred_render)
            mjudge_attempted += 1
            mjudge = self._mjudge_pairwise_win_rate(
                prompt=prompt,
                pred_image=pred_render,
                gt_image=gt_render,
                sample_id=str(gt.get("sample_id", "")),
            )
            hpsv3 = self._hpsv3_score(prompt, pred_render, clip_fallback=clip)
            hpsv2 = self._hpsv2_score(prompt, pred_render, clip_fallback=clip)
            imagereward = self._imagereward_score(prompt, pred_render)

            self._append_if_finite(clip_scores, clip)
            self._append_if_finite(dino_scores, dino)
            self._append_if_finite(lpips_scores, lpips)
            self._append_if_finite(dreamsim_scores, dreamsim)
            self._append_if_finite(nima_scores, nima)
            self._append_if_finite(mjudge_scores, mjudge)
            self._append_if_finite(hpsv3_scores, hpsv3)
            self._append_if_finite(hpsv2_scores, hpsv2)
            self._append_if_finite(imagereward_scores, imagereward)

        fid_score = float("nan")
        if len(real_features) >= 2:
            try:
                fid_score = float(fid_metric(np.stack(real_features), np.stack(gen_features)))
            except Exception:
                fid_score = float("nan")

        return {
            "miou": self._mean_or_nan(miou_scores),
            "component_coverage": self._mean_or_nan(coverage_scores),
            "mjudge_win_rate": self._mean_or_nan(mjudge_scores),
            "clip_score": self._mean_or_nan(clip_scores),
            "dino_score": self._mean_or_nan(dino_scores),
            "lpips": self._mean_or_nan(lpips_scores),
            "dreamsim_distance": self._mean_or_nan(dreamsim_scores),
            "nima_score": self._mean_or_nan(nima_scores),
            "hpsv3": self._mean_or_nan(hpsv3_scores),
            "hpsv2": self._mean_or_nan(hpsv2_scores),
            "imagereward": self._mean_or_nan(imagereward_scores),
            "fid": fid_score,
            "evaluated_samples": float(evaluated),
            "mjudge_coverage": len(mjudge_scores) / max(mjudge_attempted, 1),
            "fid_pair_count": float(len(real_features)),
        }

    def _make_visual_sample(
        self,
        *,
        layout_file: Path,
        row: Dict[str, Any],
        image_index: Dict[str, str],
        component_renders_dir: Path,
    ) -> Optional[Dict[str, Any]]:
        sid = layout_file.stem
        width = self._safe_int((row.get("layout_metadata") or {}).get("width"), 0)
        height = self._safe_int((row.get("layout_metadata") or {}).get("height"), 0)
        if width <= 0 or height <= 0:
            return None

        semantic = row.get("layout_semantic_description", {})
        if not isinstance(semantic, dict):
            semantic = {}
        prompt = self._first_nonempty(
            semantic.get("user_intent"),
            row.get("user_intent"),
            self.DEFAULT_PROMPT,
        )

        layout_components = ((row.get("layout_config") or {}).get("components") or [])
        component_refs = row.get("component_renders") or []
        if not isinstance(layout_components, list) or not isinstance(component_refs, list):
            return None
        if len(component_refs) < 2:
            return None

        usable = min(len(layout_components), len(component_refs) - 1)
        visual_components: List[Dict[str, Any]] = []
        for idx in range(usable):
            cfg = layout_components[idx]
            if not isinstance(cfg, dict) or not self._is_visual_component(cfg):
                continue

            image_path = self._resolve_component_asset(
                sample_id=sid,
                value=component_refs[idx + 1],
                component_renders_dir=component_renders_dir,
            )
            if not image_path:
                continue

            json_bbox = self._extract_bbox_from_component(
                cfg,
                canvas_width=width,
                canvas_height=height,
            )
            alpha_bbox = self._extract_bbox_from_alpha(
                image_path=image_path,
                canvas_width=width,
                canvas_height=height,
            )
            bbox, bbox_source, bbox_iou = self._resolve_component_bbox(
                json_bbox=json_bbox,
                alpha_bbox=alpha_bbox,
                canvas_width=width,
                canvas_height=height,
            )
            if bbox is None:
                continue

            area_ratio = float(bbox[2] * bbox[3]) / float(width * height)
            if area_ratio < self.MIN_COMPONENT_AREA_RATIO:
                continue

            prepared_asset = self._prepare_component_asset(
                sample_id=sid,
                source_index=idx,
                image_path=image_path,
                alpha_bbox=alpha_bbox,
                canvas_width=width,
                canvas_height=height,
                component_renders_dir=component_renders_dir,
            )
            if not prepared_asset:
                continue

            visual_components.append(
                {
                    "component_id": str(cfg.get("id") or f"{sid}_component_{idx:03d}"),
                    "source_index": int(idx),
                    "z_index": int(idx),
                    "bbox": bbox,
                    "bbox_source": bbox_source,
                    "bbox_agreement_iou": bbox_iou,
                    "component_type": str(cfg.get("type") or "").upper(),
                    "canvas_width": int(width),
                    "canvas_height": int(height),
                    "source_image_path": image_path,
                    "image_path": prepared_asset,
                    "description": self._extract_component_description(
                        cfg,
                        fallback=f"Visual component {idx + 1}",
                    ),
                }
            )

        if not visual_components:
            return None

        top_components = self._select_top_layer_components(visual_components)
        if not top_components:
            return None

        selected_components = self._select_components_for_mode(
            sample_id=sid,
            top_components=top_components,
        )
        if not selected_components:
            return None

        components: List[Dict[str, Any]] = []
        for i, comp in enumerate(selected_components, start=1):
            item = dict(comp)
            item["component_key"] = f"C{i}"
            components.append(item)

        background_image = self._resolve_component_asset(
            sample_id=sid,
            value=component_refs[0],
            component_renders_dir=component_renders_dir,
        )
        reference_image = image_index.get(sid) or str(row.get("layout_remotion_image_url") or "").strip()
        if not reference_image and not background_image:
            return None

        gt_components = [
            {
                "component_key": c["component_key"],
                "component_id": c["component_id"],
                "bbox": [float(v) for v in c["bbox"]],
                "z_index": int(c["z_index"]),
                "component_type": str(c.get("component_type") or ""),
                "source_image_path": str(c.get("source_image_path") or ""),
                "image_path": c["image_path"],
                "description": c["description"],
                "bbox_source": str(c.get("bbox_source") or ""),
                "bbox_agreement_iou": float(c.get("bbox_agreement_iou"))
                if isinstance(c.get("bbox_agreement_iou"), (int, float))
                else float("nan"),
            }
            for c in components
        ]

        return {
            "sample_id": sid,
            "prompt": prompt,
            "sub_category": str(row.get("sub_category") or ""),
            "placement_mode": self.PLACEMENT_MODE,
            "canvas_width": int(width),
            "canvas_height": int(height),
            "components": gt_components,
            "ground_truth": {
                "prompt": prompt,
                "canvas_width": int(width),
                "canvas_height": int(height),
                "background_image": background_image,
                "reference_image": reference_image,
                "placement_mode": self.PLACEMENT_MODE,
                "components": gt_components,
            },
        }

    def _load_top_layer_manifest(
        self,
        *,
        root: Path,
        manifest_path: Path,
        n: Optional[int],
        default_mode: str,
    ) -> List[Dict[str, Any]]:
        rows = self._read_top_layer_manifest_rows(manifest_path)

        samples: List[Dict[str, Any]] = []
        for row in rows:
            sample = self._make_manifest_sample(
                row=row,
                root=root,
                default_mode=default_mode,
            )
            if sample is None:
                continue
            samples.append(sample)
            if n is not None and len(samples) >= n:
                break

        if not samples:
            raise ValueError(
                f"No valid samples in top-layer manifest: {manifest_path}. "
                "Check paths (input_composite / tight_crop_asset / ground_truth_image)."
            )
        return samples

    @classmethod
    def _read_top_layer_manifest_rows(cls, manifest_path: Path) -> List[Dict[str, Any]]:
        suffix = manifest_path.suffix.lower()
        if suffix == ".csv":
            try:
                with open(manifest_path, "r", encoding="utf-8-sig", newline="") as f:
                    reader = csv.DictReader(f)
                    if not reader.fieldnames:
                        raise ValueError(f"CSV manifest has no header row: {manifest_path}")
                    rows = [cls._normalize_manifest_csv_row(row) for row in reader if isinstance(row, dict)]
                return rows
            except Exception as exc:
                raise ValueError(f"Failed to parse CSV manifest {manifest_path}: {exc}") from exc

        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Failed to parse manifest {manifest_path}: {exc}") from exc

        rows = payload.get("samples", []) if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError(
                f"Manifest must be a list (or dict with samples): {manifest_path}"
            )
        return rows

    @classmethod
    def _normalize_manifest_csv_row(cls, row: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, value in row.items():
            k = str(key or "").strip()
            if not k:
                continue
            out[k] = value

        prompt = str(out.get("prompt") or "").strip()
        if prompt:
            prompt = prompt.replace("\\r\\n", "\n").replace("\\n", "\n")
            out["prompt"] = prompt

        for key in ("canvas_width", "canvas_height"):
            if key in out:
                out[key] = cls._safe_int(out.get(key), 0)

        for key in ("components", "top_layer_candidate_indices", "removed_indices", "remaining_indices"):
            raw = out.get(key)
            if raw is None:
                continue
            if isinstance(raw, (list, dict)):
                continue
            text = str(raw).strip()
            if not text:
                out[key] = [] if key != "components" else []
                continue
            try:
                decoded = json.loads(text)
                out[key] = decoded
            except Exception:
                if key == "components":
                    out[key] = []
                else:
                    vals = [v.strip() for v in text.split(",") if v.strip()]
                    out[key] = [cls._safe_int(v, 0) for v in vals]

        return out

    def _make_manifest_sample(
        self,
        *,
        row: Any,
        root: Path,
        default_mode: str,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(row, dict):
            return None

        sample_id = self._first_nonempty(row.get("sample_id"))
        if not sample_id:
            return None
        sample_mode = self._first_nonempty(row.get("mode"), default_mode).lower()
        if sample_mode not in {"single", "multiple"}:
            sample_mode = default_mode
        prompt = self._first_nonempty(row.get("prompt"), self.DEFAULT_PROMPT)
        width = self._safe_int(row.get("canvas_width"), 0)
        height = self._safe_int(row.get("canvas_height"), 0)
        if width <= 0 or height <= 0:
            return None

        base_sample_id = self._first_nonempty(
            row.get("base_sample_id"),
            sample_id.split("_component_")[0],
            sample_id.replace("_toplayer", ""),
            sample_id,
        )
        source_layout_components = self._load_layout_components_for_manifest_sample(
            root=root,
            base_sample_id=base_sample_id,
        )

        input_composite = self._resolve_manifest_path(root, row.get("input_composite"))
        gt_image = self._resolve_manifest_path(root, row.get("ground_truth_image"))
        if not input_composite or not gt_image:
            return None

        raw_components = row.get("components") or []
        if not isinstance(raw_components, list):
            return None

        components: List[Dict[str, Any]] = []
        for idx, item in enumerate(raw_components):
            if not isinstance(item, dict):
                continue

            bbox_raw = item.get("bbox")
            if not isinstance(bbox_raw, list) or len(bbox_raw) < 4:
                continue
            bbox = self._clip_box(
                [self._parse_number(v, 0.0) for v in bbox_raw[:4]],
                width,
                height,
            )
            if bbox is None:
                continue

            image_path = self._resolve_manifest_path(
                root,
                item.get("tight_crop_asset") or item.get("image_path"),
            )
            if not image_path:
                continue

            component_key = self._first_nonempty(item.get("component_key"), f"C{idx + 1}")
            component_id = self._first_nonempty(
                item.get("component_id"),
                f"{sample_id}_component_{idx:03d}",
            )
            source_index = self._safe_int(
                item.get("source_index", item.get("z_index", idx)),
                idx,
            )
            source_render = str(item.get("source_render") or "").strip()
            source_render_path = self._resolve_manifest_path(root, source_render)
            source_cfg: Optional[Dict[str, Any]] = None
            if 0 <= source_index < len(source_layout_components):
                cfg = source_layout_components[source_index]
                if isinstance(cfg, dict):
                    source_cfg = cfg
            component_type = self._first_nonempty(
                item.get("component_type"),
                source_cfg.get("type") if isinstance(source_cfg, dict) else "",
                source_cfg.get("data0_element_type") if isinstance(source_cfg, dict) else "",
            )
            components.append(
                {
                    "component_key": component_key,
                    "component_id": component_id,
                    "bbox": [float(v) for v in bbox],
                    "z_index": int(source_index),
                    "component_type": str(component_type or "").upper(),
                    "source_image_path": source_render_path or source_render,
                    "image_path": image_path,
                    "description": self._build_manifest_component_description(
                        item=item,
                        source_cfg=source_cfg,
                        image_path=image_path,
                        fallback=f"Top-layer component {idx + 1}",
                    ),
                    "bbox_source": "mask_bbox",
                    "bbox_agreement_iou": float("nan"),
                }
            )

        if not components:
            return None

        components.sort(key=lambda c: c.get("z_index", 0))
        # Normalize component keys to deterministic C1..Cn ordering for prompting/parsing.
        for i, comp in enumerate(components, start=1):
            comp["component_key"] = f"C{i}"

        gt_components = [
            {
                "component_key": c["component_key"],
                "component_id": c["component_id"],
                "bbox": [float(v) for v in c["bbox"]],
                "z_index": int(c["z_index"]),
                "component_type": str(c.get("component_type") or ""),
                "source_image_path": str(c.get("source_image_path") or ""),
                "image_path": c["image_path"],
                "description": c["description"],
                "bbox_source": str(c.get("bbox_source") or ""),
                "bbox_agreement_iou": float(c.get("bbox_agreement_iou"))
                if isinstance(c.get("bbox_agreement_iou"), (int, float))
                else float("nan"),
            }
            for c in components
        ]

        return {
            "sample_id": sample_id,
            "prompt": prompt,
            "sub_category": str(row.get("sub_category") or ""),
            "placement_mode": sample_mode,
            "canvas_width": int(width),
            "canvas_height": int(height),
            "input_composite": input_composite,
            "components": gt_components,
            "ground_truth": {
                "prompt": prompt,
                "canvas_width": int(width),
                "canvas_height": int(height),
                "base_image": input_composite,
                "ground_truth_image": gt_image,
                "placement_mode": sample_mode,
                "components": gt_components,
            },
        }

    def _load_layout_components_for_manifest_sample(
        self,
        *,
        root: Path,
        base_sample_id: str,
    ) -> List[Dict[str, Any]]:
        if not base_sample_id:
            return []

        cache = getattr(self, "_layout_component_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            setattr(self, "_layout_component_cache", cache)
        if base_sample_id in cache:
            cached = cache[base_sample_id]
            return cached if isinstance(cached, list) else []

        for candidate_id in self._candidate_base_sample_ids(base_sample_id):
            if candidate_id in cache:
                cached = cache[candidate_id]
                out = cached if isinstance(cached, list) else []
                cache[base_sample_id] = out
                return out

            layout_path = self._resolve_layout_json_path(root=root, sample_id=candidate_id)
            if layout_path is None:
                continue

            try:
                payload = json.loads(layout_path.read_text(encoding="utf-8"))
            except Exception:
                continue

            layout_cfg = payload.get("layout_config") if isinstance(payload, dict) else None
            components = layout_cfg.get("components") if isinstance(layout_cfg, dict) else None
            if not isinstance(components, list):
                continue

            out = [c for c in components if isinstance(c, dict)]
            cache[candidate_id] = out
            cache[base_sample_id] = out
            return out

        cache[base_sample_id] = []
        return []

    @classmethod
    def _candidate_base_sample_ids(cls, sample_id: str) -> List[str]:
        raw = str(sample_id or "").strip()
        if not raw:
            return []
        candidates = [raw]
        stripped = cls._strip_known_manifest_id_prefix(raw)
        if stripped and stripped not in candidates:
            candidates.append(stripped)
        return candidates

    @classmethod
    def _strip_known_manifest_id_prefix(cls, value: str) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""
        for prefix in cls.KNOWN_MANIFEST_ID_PREFIXES:
            p = str(prefix or "").strip()
            if p and raw.startswith(p):
                return raw[len(p):]
        return raw

    @staticmethod
    def _resolve_layout_json_path(*, root: Path, sample_id: str) -> Optional[Path]:
        if not sample_id:
            return None
        candidates = [
            root / "layouts" / f"{sample_id}.json",
            root.parent / "layouts" / f"{sample_id}.json",
            root.parent.parent / "layouts" / f"{sample_id}.json",
            root / f"{sample_id}.json",
        ]
        for path in candidates:
            if path.is_file():
                return path
        return None

    def _build_manifest_component_description(
        self,
        *,
        item: Dict[str, Any],
        source_cfg: Optional[Dict[str, Any]],
        image_path: str,
        fallback: str,
    ) -> str:
        raw_desc = self._normalize_component_description_text(item.get("description"))
        if not raw_desc or self._placeholder_desc_re.match(raw_desc):
            if isinstance(source_cfg, dict):
                raw_desc = self._extract_component_description(source_cfg, fallback=fallback)
            else:
                raw_desc = fallback
        raw_desc = self._normalize_component_description_text(raw_desc)
        if not raw_desc:
            raw_desc = fallback

        shape_hint = self._asset_shape_hint(image_path)
        if shape_hint:
            return f"{raw_desc} Visual cue: {shape_hint}."
        return raw_desc

    @classmethod
    def _normalize_component_description_text(cls, value: Any, *, max_chars: int = 240) -> str:
        text = html.unescape(str(value or ""))
        text = cls._html_tag_re.sub(" ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return ""
        if len(text) > max_chars:
            text = text[: max_chars - 3].rstrip() + "..."
        return text

    def _asset_geometry_stats(self, image_path: str) -> Dict[str, Any]:
        src = str(image_path or "").strip()
        if not src:
            return {
                "width": 0,
                "height": 0,
                "aspect_ratio": 0.0,
                "alpha_ratio": 0.0,
                "shape_hint": "",
            }

        cache = getattr(self, "_asset_geometry_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            setattr(self, "_asset_geometry_cache", cache)
        if src in cache:
            cached = cache[src]
            return dict(cached) if isinstance(cached, dict) else {
                "width": 0,
                "height": 0,
                "aspect_ratio": 0.0,
                "alpha_ratio": 0.0,
                "shape_hint": "",
            }

        try:
            from PIL import Image
        except ImportError:
            out = {
                "width": 0,
                "height": 0,
                "aspect_ratio": 0.0,
                "alpha_ratio": 0.0,
                "shape_hint": "",
            }
            cache[src] = out
            return dict(out)

        path = Path(src)
        if not path.is_file():
            out = {
                "width": 0,
                "height": 0,
                "aspect_ratio": 0.0,
                "alpha_ratio": 0.0,
                "shape_hint": "",
            }
            cache[src] = out
            return dict(out)

        try:
            with Image.open(path) as pil:
                rgba = pil.convert("RGBA")
                width, height = rgba.size
                alpha = np.asarray(rgba.getchannel("A"), dtype=np.uint8)
        except Exception:
            out = {
                "width": 0,
                "height": 0,
                "aspect_ratio": 0.0,
                "alpha_ratio": 0.0,
                "shape_hint": "",
            }
            cache[src] = out
            return dict(out)

        if width <= 0 or height <= 0:
            out = {
                "width": 0,
                "height": 0,
                "aspect_ratio": 0.0,
                "alpha_ratio": 0.0,
                "shape_hint": "",
            }
            cache[src] = out
            return dict(out)

        aspect = float(width) / float(height)
        max_side = max(width, height)
        alpha_ratio = float((alpha > 0).mean()) if alpha.size > 0 else 1.0

        if max_side <= 96:
            size_bucket = "small"
        elif max_side <= 320:
            size_bucket = "medium"
        else:
            size_bucket = "large"

        if aspect >= 2.8:
            shape_bucket = "very wide"
        elif aspect >= 1.5:
            shape_bucket = "wide"
        elif aspect <= 0.45:
            shape_bucket = "very tall"
        elif aspect <= 0.75:
            shape_bucket = "tall"
        else:
            shape_bucket = "roughly square"

        if alpha_ratio >= 0.85:
            alpha_bucket = "mostly opaque"
        elif alpha_ratio >= 0.45:
            alpha_bucket = "partially transparent"
        else:
            alpha_bucket = "sparse on transparent background"

        out = {
            "width": int(width),
            "height": int(height),
            "aspect_ratio": float(aspect),
            "alpha_ratio": float(alpha_ratio),
            "shape_hint": f"{size_bucket}, {shape_bucket}, {alpha_bucket}",
        }
        cache[src] = out
        return dict(out)

    def _asset_shape_hint(self, image_path: str) -> str:
        stats = self._asset_geometry_stats(image_path)
        return str(stats.get("shape_hint") or "")

    def _compose_component_layout_prompt(self, sample: Dict[str, Any], *, with_images: bool) -> str:
        width = int(sample.get("canvas_width", 0))
        height = int(sample.get("canvas_height", 0))
        prompt = str(sample.get("prompt", self.DEFAULT_PROMPT))
        components = sample.get("components", [])
        mode = str(sample.get("placement_mode", self.PLACEMENT_MODE)).lower().strip()
        is_single = mode == "single"
        has_input_composite = bool(str(sample.get("input_composite") or "").strip())
        mode_label = mode if mode in {"single", "multiple"} else ("single" if is_single else "multiple")
        sample_id = str(sample.get("sample_id") or "").strip()

        lines = [
            "You are an expert layout planner focused on high-fidelity placement.",
        ]
        if sample_id:
            lines.append(f"Sample ID: {sample_id}.")
        lines.extend(
            [
            f"User intent: {prompt}",
            f"Canvas size: {width}x{height} pixels.",
            f"Placement mode: {mode_label}.",
            "",
            "Task objective:",
            "- Predict axis-aligned bounding boxes [x, y, w, h] for the listed component keys.",
            "- Infer coordinates from available evidence only; exact original coordinates are intentionally hidden.",
            "",
            "Evidence available in this task:",
            "- A base composite image with target component(s) removed.",
            "- One asset image per target component, preserving native crop size and transparency.",
            "- Semantic descriptions and structural cues for each component.",
            "",
            "Dataset prior:",
            "- Listed components are top-layer elements removed from the same layout context.",
            "- Non-listed content in the base composite should remain undisturbed.",
            "",
            "You are given visual element components.",
            ]
        )
        if with_images and has_input_composite:
            lines.extend(
                [
                    "Input mapping:",
                    "- Input image #1 is the base composite with target component(s) removed.",
                    "- Input images #2..#(N+1) are component assets in the same order as the list below.",
                    "- Use the base composite to infer anchors (alignment lines, spacing rhythm, visual groups).",
                ]
            )
        elif with_images:
            lines.extend(
                [
                    "Input mapping:",
                    "- Each input image corresponds to one component in the same order as the list below.",
                    "- Infer placement from component asset content and the user intent.",
                ]
            )
        else:
            lines.append("Use the component descriptions below (text-only mode).")
        lines.extend(
            [
                "- Preserve each component's visual identity and style in placement.",
                "",
                "Components (output must follow these keys):",
            ]
        )

        required_keys: List[str] = []
        canvas_area = float(max(1, width * height))
        for idx, comp in enumerate(components, start=1):
            key = str(comp.get("component_key") or f"C{idx}")
            required_keys.append(key)
            comp_type = str(comp.get("component_type") or "UNKNOWN")
            z_index = self._safe_int(comp.get("z_index"), idx - 1)
            desc = self._normalize_component_description_text(comp.get("description")) or f"Component {idx}"
            stats = self._asset_geometry_stats(str(comp.get("image_path") or ""))
            asset_w = self._safe_int(stats.get("width"), 0)
            asset_h = self._safe_int(stats.get("height"), 0)
            asset_aspect = float(stats.get("aspect_ratio") or 0.0)
            alpha_ratio = float(stats.get("alpha_ratio") or 0.0)
            shape_hint = str(stats.get("shape_hint") or "")
            native_area_pct = (
                (100.0 * float(asset_w * asset_h) / canvas_area)
                if asset_w > 0 and asset_h > 0
                else float("nan")
            )

            if with_images and has_input_composite:
                image_ref = idx + 1
                lines.append(
                    f"- {key} (input image #{image_ref}, type={comp_type}, z_index={z_index}): {desc}"
                )
            elif with_images:
                lines.append(f"- {key} (input image #{idx}, type={comp_type}, z_index={z_index}): {desc}")
            else:
                lines.append(f"- {key} (type={comp_type}, z_index={z_index}): {desc}")

            if asset_w > 0 and asset_h > 0:
                lines.append(
                    "  - Native asset geometry: "
                    f"{asset_w}x{asset_h}px, aspect={asset_aspect:.3f}, "
                    f"native_canvas_area={native_area_pct:.2f}%, alpha_coverage={alpha_ratio * 100.0:.2f}%."
                )
            if shape_hint:
                lines.append(f"  - Shape prior: {shape_hint}.")

        mode_task = (
            "- Predict exactly one bounding box for the single listed component."
            if is_single
            else "- Predict one bounding box for every listed component."
        )
        mode_schema_note = (
            "- Return exactly one component object in the output array."
            if is_single
            else "- Return all listed components in the output array, each exactly once."
        )
        lines.extend(
            [
                "",
                "Task:",
                mode_task,
                mode_schema_note,
                f"- Required output component keys: {', '.join(required_keys) if required_keys else '(none)'}",
                "",
                "Quality constraints (strict):",
                "- Keep each component's native aspect ratio from its asset; do not stretch or squash.",
                "- Prefer near-native asset scale unless scene context clearly requires resizing.",
                "- Do not expand foreground components to near full-canvas unless they are obvious full-bleed backgrounds.",
                "- Place components to align naturally with nearby spacing, edges, and reading flow in the base composite.",
                "- In multiple mode, keep a coherent hierarchy and avoid unnecessary overlap.",
                "- In multiple mode, avoid duplicate placement of semantically similar assets in the same location.",
                "- When uncertain, preserve relative ordering and spacing consistency from surrounding context.",
                "- Keep all boxes within canvas bounds.",
                "- Return JSON only (no markdown/code fences/explanations).",
                "",
                "Output format requirements:",
                "- Use numeric pixel coordinates.",
                '- Preferred component format: {"component_key": "C1", "bbox": [x, y, w, h]}.',
                "- If you use style instead of bbox, include left/top/width/height as pixel values.",
                f"- layout_config.width must be {width}; layout_config.height must be {height}.",
                "- Each required component key must appear exactly once.",
                "- All bbox values must be finite numbers with w>1 and h>1.",
                "",
                "JSON schema:",
                "{",
                '  "layout_config": {',
                '    "width": <int>,',
                '    "height": <int>,',
                '    "components": [',
                "      {",
                '        "component_key": "C1",',
                '        "bbox": [<x>, <y>, <w>, <h>]',
                "      }",
                "    ]",
                "  }",
                "}",
            ]
        )
        return "\n".join(lines)

    def _select_top_layer_components(self, components: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not components:
            return []

        by_z_asc = sorted(components, key=lambda c: c.get("z_index", 0))
        background = None
        for comp in by_z_asc:
            if str(comp.get("component_type", "")).upper() == "TEXT":
                continue
            bbox = comp.get("bbox") or [0.0, 0.0, 0.0, 0.0]
            area = float(bbox[2] * bbox[3])
            canvas_area = float(max(1.0, comp.get("canvas_width", 0) * comp.get("canvas_height", 0)))
            ratio = area / canvas_area if canvas_area > 0 else 0.0
            if ratio >= self.BACKGROUND_AREA_RATIO:
                background = comp
                break

        top_to_bottom = sorted(components, key=lambda c: c.get("z_index", 0), reverse=True)
        if self.TOP_LAYER_SCAN > 0:
            top_to_bottom = top_to_bottom[: self.TOP_LAYER_SCAN]

        if background is not None:
            bg_id = str(background.get("component_id", ""))
            filtered = [c for c in top_to_bottom if str(c.get("component_id", "")) != bg_id]
            if filtered:
                top_to_bottom = filtered

        return sorted(top_to_bottom, key=lambda c: c.get("z_index", 0))

    def _select_components_for_mode(
        self,
        *,
        sample_id: str,
        top_components: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not top_components:
            return []

        mode = str(self.PLACEMENT_MODE).lower().strip()
        if mode == "single":
            digest = hashlib.sha1(f"{sample_id}|single".encode("utf-8")).hexdigest()
            idx = int(digest, 16) % len(top_components)
            return [top_components[idx]]
        return top_components

    def _resolve_component_bbox(
        self,
        *,
        json_bbox: Optional[List[float]],
        alpha_bbox: Optional[List[float]],
        canvas_width: int,
        canvas_height: int,
    ) -> Tuple[Optional[List[float]], str, float]:
        from design_benchmarks.metrics.core import iou as iou_metric

        if json_bbox is not None:
            json_bbox = self._clip_box(json_bbox, canvas_width, canvas_height)
        if alpha_bbox is not None:
            alpha_bbox = self._clip_box(alpha_bbox, canvas_width, canvas_height)

        if json_bbox is None and alpha_bbox is None:
            return None, "none", float("nan")

        if json_bbox is None:
            return alpha_bbox, "alpha_only", float("nan")
        if alpha_bbox is None:
            return json_bbox, "json_only", float("nan")

        agree_iou = float(iou_metric(json_bbox, alpha_bbox))
        if agree_iou >= self.JSON_ALPHA_IOU_AGREE:
            return json_bbox, "json", agree_iou
        return alpha_bbox, "alpha_fallback", agree_iou

    def _extract_bbox_from_alpha(
        self,
        *,
        image_path: str,
        canvas_width: int,
        canvas_height: int,
    ) -> Optional[List[float]]:
        try:
            from PIL import Image
        except ImportError:
            return None

        p = Path(image_path)
        if not p.is_file():
            return None

        try:
            with Image.open(p) as img:
                rgba = np.asarray(img.convert("RGBA"))
        except Exception:
            return None

        if rgba.ndim != 3 or rgba.shape[2] < 4:
            return None
        if rgba.shape[1] != canvas_width or rgba.shape[0] != canvas_height:
            # Without canvas-sized render, absolute bbox is ambiguous.
            return None

        alpha = rgba[:, :, 3]
        ys, xs = np.where(alpha > 1)
        if ys.size < self.MIN_ALPHA_PIXELS or xs.size < self.MIN_ALPHA_PIXELS:
            return None

        x1 = float(xs.min())
        y1 = float(ys.min())
        x2 = float(xs.max()) + 1.0
        y2 = float(ys.max()) + 1.0
        return self._clip_box([x1, y1, x2 - x1, y2 - y1], canvas_width, canvas_height)

    def _prepare_component_asset(
        self,
        *,
        sample_id: str,
        source_index: int,
        image_path: str,
        alpha_bbox: Optional[List[float]],
        canvas_width: int,
        canvas_height: int,
        component_renders_dir: Path,
    ) -> str:
        p = Path(str(image_path or "").strip())
        if not p.is_file():
            return str(image_path or "")
        if alpha_bbox is None:
            return str(p.resolve())

        try:
            from PIL import Image
        except ImportError:
            return str(p.resolve())

        try:
            with Image.open(p) as src:
                rgba = src.convert("RGBA")
                if rgba.size != (canvas_width, canvas_height):
                    return str(p.resolve())
                x, y, w, h = [int(round(v)) for v in alpha_bbox]
                if w <= 1 or h <= 1:
                    return str(p.resolve())
                x = max(0, min(canvas_width - 1, x))
                y = max(0, min(canvas_height - 1, y))
                w = max(1, min(canvas_width - x, w))
                h = max(1, min(canvas_height - y, h))
                cropped = rgba.crop((x, y, x + w, y + h))
        except Exception:
            return str(p.resolve())

        out_dir = component_renders_dir / "_layout_component_crops" / sample_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"component_{source_index:03d}.png"
        try:
            cropped.save(out_path)
            return str(out_path.resolve())
        except Exception:
            return str(p.resolve())

    def _extract_bbox_from_component(
        self,
        cfg: Dict[str, Any],
        *,
        canvas_width: int,
        canvas_height: int,
    ) -> Optional[List[float]]:
        style = cfg.get("style") or {}
        if not isinstance(style, dict):
            return None

        width = self._parse_number(style.get("width"), 0.0)
        height = self._parse_number(style.get("height"), 0.0)
        if width <= 1.0 or height <= 1.0:
            return None

        left = self._parse_number(style.get("left"), 0.0)
        top = self._parse_number(style.get("top"), 0.0)
        tx, ty = self._extract_translate(style.get("transform"))

        x = left + tx
        y = top + ty
        clipped = self._clip_box([x, y, width, height], canvas_width, canvas_height)
        if clipped is None:
            return None
        return [float(v) for v in clipped]

    @classmethod
    def _is_visual_component(cls, cfg: Dict[str, Any]) -> bool:
        ctype = str(cfg.get("type") or "").upper()
        if ctype == "IMAGE":
            return True
        if ctype == "TEXT":
            return True
        if ctype != "GROUP":
            return False

        subtype = str(cfg.get("data0_element_type") or "").lower()
        if subtype in {"standard_img", "lottie_svg", "graph_svg", "svg_stroke", "frame_grid"}:
            return True

        children = cfg.get("components") or []
        return any(
            isinstance(ch, dict) and str(ch.get("type") or "").upper() in {"IMAGE", "TEXT"}
            for ch in children
        )

    @classmethod
    def _extract_component_description(cls, cfg: Dict[str, Any], *, fallback: str) -> str:
        element = cfg.get("element")
        if not isinstance(element, dict):
            element = {}

        ctype = str(cfg.get("type") or "").upper()
        if ctype == "TEXT":
            text_value = cls._normalize_component_description_text(
                cls._first_nonempty(cfg.get("text"), element.get("text"))
            )
            if text_value:
                return f'Text element containing "{text_value}".'

        if ctype == "GROUP":
            text_children = []
            for child in cfg.get("components") or []:
                if not isinstance(child, dict):
                    continue
                if str(child.get("type") or "").upper() != "TEXT":
                    continue
                child_text = cls._normalize_component_description_text(
                    cls._first_nonempty(child.get("text"), (child.get("element") or {}).get("text"))
                )
                if child_text:
                    text_children.append(child_text)
            if text_children:
                preview = ", ".join(f'"{t}"' for t in text_children[:3])
                return f"Grouped text element containing: {preview}."

        raw = cls._normalize_component_description_text(
            cls._first_nonempty(
                cfg.get("alt"),
                element.get("alt"),
                element.get("description"),
                fallback,
            )
        )
        return raw or cls._normalize_component_description_text(fallback)

    @staticmethod
    def _parse_number(value: Any, default: float = 0.0) -> float:
        if value is None:
            return float(default)
        if isinstance(value, (int, float)):
            return float(value)
        m = PartialLayoutCompletion._num_re.search(str(value))
        if not m:
            return float(default)
        try:
            return float(m.group(0))
        except Exception:
            return float(default)

    @classmethod
    def _extract_translate(cls, transform_value: Any) -> Tuple[float, float]:
        text = str(transform_value or "")
        if not text:
            return 0.0, 0.0

        m = cls._translate_re.search(text)
        if m:
            tx = cls._parse_number(m.group(1), 0.0)
            ty = cls._parse_number(m.group(2), 0.0)
            return tx, ty

        mx = cls._translate_x_re.search(text)
        my = cls._translate_y_re.search(text)
        tx = cls._parse_number(mx.group(1), 0.0) if mx else 0.0
        ty = cls._parse_number(my.group(1), 0.0) if my else 0.0
        return tx, ty

    @staticmethod
    def _clip_box(
        box: List[float],
        canvas_width: int,
        canvas_height: int,
    ) -> Optional[List[float]]:
        if canvas_width <= 0 or canvas_height <= 0:
            return None
        x, y, w, h = [float(v) for v in box]
        if w <= 1.0 or h <= 1.0:
            return None

        x1 = max(0.0, min(float(canvas_width), x))
        y1 = max(0.0, min(float(canvas_height), y))
        x2 = max(0.0, min(float(canvas_width), x + w))
        y2 = max(0.0, min(float(canvas_height), y + h))
        if x2 - x1 <= 1.0 or y2 - y1 <= 1.0:
            return None
        return [x1, y1, x2 - x1, y2 - y1]

    def _resolve_component_asset(
        self,
        *,
        sample_id: str,
        value: Any,
        component_renders_dir: Path,
    ) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""

        as_path = Path(raw)
        if as_path.is_file():
            return str(as_path.resolve())

        sample_dir = component_renders_dir / sample_id
        if raw.startswith(("http://", "https://")):
            filename = Path(urlparse(raw).path).name
            if filename:
                cached = sample_dir / filename
                if cached.is_file():
                    return str(cached.resolve())
            return raw

        if sample_dir.is_dir():
            candidate = sample_dir / raw
            if candidate.is_file():
                return str(candidate.resolve())
        return ""

    @staticmethod
    def _resolve_manifest_path(root: Path, value: Any) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""
        as_path = Path(raw)
        if as_path.is_file():
            return str(as_path.resolve())
        candidate = (root / raw).resolve()
        if candidate.is_file():
            return str(candidate)
        return ""

    @classmethod
    def _strip_code_fence(cls, text: str) -> str:
        cleaned = str(text or "").strip()
        cleaned = re.sub(r"^```(?:json|JSON|txt|text)?\s*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)
        return cleaned.strip()

    @classmethod
    def _decode_json_like(cls, text: str) -> Optional[Any]:
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            pass

        for i, ch in enumerate(text):
            if ch not in "{[":
                continue
            try:
                obj, _ = cls._json_decoder.raw_decode(text[i:])
                return obj
            except Exception:
                continue
        return None

    def _salvage_components_from_fragment(self, text: str) -> List[Dict[str, Any]]:
        src = str(text or "")
        if not src:
            return []

        out: List[Dict[str, Any]] = []
        seen_keys: set[str] = set()

        for match in self._bbox_fuzzy_re.finditer(src):
            values = [float(n) for n in self._num_re.findall(match.group(1))]
            if len(values) < 4:
                continue
            box = [float(values[0]), float(values[1]), float(values[2]), float(values[3])]
            if box[2] <= 1.0 or box[3] <= 1.0:
                continue

            prefix = src[max(0, match.start() - 260):match.start()]
            key_matches = list(self._component_key_fuzzy_re.finditer(prefix))
            key = key_matches[-1].group(1).strip() if key_matches else f"@{len(out)}"
            if not key:
                key = f"@{len(out)}"
            if key in seen_keys:
                continue

            seen_keys.add(key)
            out.append(
                {
                    "component_key": key,
                    "component_id": "",
                    "bbox": box,
                    "z_index": len(out),
                    "order_index": len(out),
                }
            )
        return out

    def _extract_predicted_components(self, payload: Any) -> List[Dict[str, Any]]:
        candidates: Any = None
        if isinstance(payload, list):
            candidates = payload
        elif isinstance(payload, dict):
            if isinstance(payload.get("components"), list):
                candidates = payload.get("components")
            elif isinstance(payload.get("placements"), list):
                candidates = payload.get("placements")
            else:
                layout_cfg = payload.get("layout_config")
                if isinstance(layout_cfg, dict):
                    if isinstance(layout_cfg.get("components"), list):
                        candidates = layout_cfg.get("components")
                    elif isinstance(layout_cfg.get("placements"), list):
                        candidates = layout_cfg.get("placements")
        if not isinstance(candidates, list):
            return []

        out: List[Dict[str, Any]] = []
        for idx, item in enumerate(candidates):
            if not isinstance(item, dict):
                continue

            key = self._first_nonempty(
                item.get("component_key"),
                item.get("key"),
                item.get("component_id"),
                item.get("id"),
            )
            if not key and item.get("index") is not None:
                key = f"C{self._safe_int(item.get('index'), idx) + 1}"
            if not key:
                key = f"@{idx}"

            bbox = self._extract_pred_bbox(item)
            if bbox is None:
                continue

            z_index = self._safe_int(
                item.get("z_index", item.get("layer", item.get("order"))),
                idx,
            )
            out.append(
                {
                    "component_key": key,
                    "component_id": str(item.get("component_id", "")),
                    "bbox": bbox,
                    "z_index": z_index,
                    "order_index": idx,
                }
            )
        return out

    def _extract_pred_bbox(self, item: Dict[str, Any]) -> Optional[List[float]]:
        bbox = item.get("bbox")
        if isinstance(bbox, list) and len(bbox) >= 4:
            box = [self._parse_number(v, 0.0) for v in bbox[:4]]
            return box if box[2] > 1.0 and box[3] > 1.0 else None

        if isinstance(bbox, dict):
            box = [
                self._parse_number(bbox.get("x", bbox.get("left")), 0.0),
                self._parse_number(bbox.get("y", bbox.get("top")), 0.0),
                self._parse_number(bbox.get("width", bbox.get("w")), 0.0),
                self._parse_number(bbox.get("height", bbox.get("h")), 0.0),
            ]
            return box if box[2] > 1.0 and box[3] > 1.0 else None

        style = item.get("style")
        if isinstance(style, dict):
            box = [
                self._parse_number(style.get("left", item.get("x")), 0.0),
                self._parse_number(style.get("top", item.get("y")), 0.0),
                self._parse_number(style.get("width", item.get("width")), 0.0),
                self._parse_number(style.get("height", item.get("height")), 0.0),
            ]
            if box[2] > 1.0 and box[3] > 1.0:
                return box

        box = [
            self._parse_number(item.get("x", item.get("left")), 0.0),
            self._parse_number(item.get("y", item.get("top")), 0.0),
            self._parse_number(item.get("width", item.get("w")), 0.0),
            self._parse_number(item.get("height", item.get("h")), 0.0),
        ]
        if box[2] > 1.0 and box[3] > 1.0:
            return box
        return None

    def _normalize_visual_gt(self, raw: Any) -> Dict[str, Any]:
        if not isinstance(raw, dict):
            return {
                "sample_id": "",
                "_sample_id": "",
                "_model_name": "",
                "_benchmark_id": "",
                "placement_mode": "",
                "prompt": "",
                "canvas_width": 0,
                "canvas_height": 0,
                "base_image": "",
                "background_image": "",
                "ground_truth_image": "",
                "components": [],
            }

        width = self._safe_int(raw.get("canvas_width", raw.get("width")), 0)
        height = self._safe_int(raw.get("canvas_height", raw.get("height")), 0)
        components: List[Dict[str, Any]] = []
        for idx, comp in enumerate(raw.get("components") or []):
            if not isinstance(comp, dict):
                continue
            bbox = comp.get("bbox")
            if not isinstance(bbox, list) or len(bbox) < 4:
                continue
            clipped = self._clip_box(
                [self._parse_number(v, 0.0) for v in bbox[:4]],
                width,
                height,
            )
            if clipped is None:
                continue
            components.append(
                {
                    "component_key": str(comp.get("component_key") or f"C{idx + 1}"),
                    "component_id": str(comp.get("component_id") or ""),
                    "bbox": clipped,
                    "z_index": self._safe_int(comp.get("z_index"), idx),
                    "component_type": str(comp.get("component_type") or ""),
                    "source_image_path": str(comp.get("source_image_path") or ""),
                    "image_path": str(comp.get("image_path") or ""),
                    "description": str(comp.get("description") or ""),
                    "bbox_source": str(comp.get("bbox_source") or ""),
                }
            )

        return {
            "sample_id": str(raw.get("sample_id", raw.get("_sample_id", "")) or ""),
            "_sample_id": str(raw.get("_sample_id", raw.get("sample_id", "")) or ""),
            "_model_name": str(raw.get("_model_name") or ""),
            "_benchmark_id": str(raw.get("_benchmark_id") or ""),
            "prompt": str(raw.get("prompt") or ""),
            "canvas_width": width,
            "canvas_height": height,
            "base_image": str(raw.get("base_image", raw.get("input_composite", "")) or ""),
            "background_image": str(raw.get("background_image") or ""),
            "ground_truth_image": str(raw.get("ground_truth_image") or ""),
            "reference_image": str(raw.get("reference_image") or ""),
            "placement_mode": str(raw.get("placement_mode") or ""),
            "components": components,
        }

    def _normalize_prediction_map(
        self,
        pred_raw: Any,
        *,
        canvas_width: int,
        canvas_height: int,
    ) -> Tuple[Dict[str, List[float]], List[List[float]]]:
        payload = pred_raw
        if isinstance(pred_raw, dict) and isinstance(pred_raw.get("components"), list):
            payload = pred_raw.get("components")

        if isinstance(payload, list):
            comps = payload
        elif isinstance(payload, dict):
            comps = payload.get("components") or payload.get("placements") or []
        else:
            comps = []

        mapping: Dict[str, List[float]] = {}
        ordered: List[List[float]] = []
        for idx, comp in enumerate(comps):
            if not isinstance(comp, dict):
                continue
            bbox = comp.get("bbox")
            if not isinstance(bbox, list) or len(bbox) < 4:
                continue
            clipped = self._clip_box(
                [self._parse_number(v, 0.0) for v in bbox[:4]],
                canvas_width,
                canvas_height,
            )
            if clipped is None:
                continue

            key = self._first_nonempty(
                comp.get("component_key"),
                comp.get("component_id"),
                comp.get("id"),
                f"@{idx}",
            )
            mapping[key] = clipped
            ordered.append(clipped)
        return mapping, ordered

    def _render_layout_from_boxes(
        self,
        *,
        gt: Dict[str, Any],
        component_boxes: Dict[str, List[float]],
        fallback_order: List[List[float]],
    ) -> Optional[np.ndarray]:
        width = int(gt.get("canvas_width", 0))
        height = int(gt.get("canvas_height", 0))
        if width <= 0 or height <= 0:
            return None

        try:
            from PIL import Image
        except ImportError:
            return None

        base = self._load_rgba_image(
            gt.get("base_image", "") or gt.get("background_image", ""),
            size=(width, height),
        )
        if base is None:
            base = Image.new("RGBA", (width, height), (255, 255, 255, 255))

        components = sorted(gt.get("components", []), key=lambda c: c.get("z_index", 0))
        for idx, comp in enumerate(components):
            box = component_boxes.get(comp["component_key"])
            if box is None and idx < len(fallback_order):
                box = fallback_order[idx]
            if box is None:
                continue
            x, y, w, h = [int(round(v)) for v in box]
            if w <= 1 or h <= 1:
                continue

            asset = self._load_rgba_image(comp.get("image_path", ""), size=(w, h))
            if asset is None:
                continue

            layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            layer.paste(asset, (x, y), asset)
            base = Image.alpha_composite(base, layer)

        return np.asarray(base.convert("RGB"), dtype=np.uint8)

    @staticmethod
    def _safe_fs_name(value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))
        cleaned = cleaned.strip("._")
        return cleaned or "item"

    @classmethod
    def _maybe_save_layout3_renders(
        cls,
        *,
        gt: Dict[str, Any],
        pred_raw: Any,
        pred_map: Dict[str, List[float]],
        pred_order: List[List[float]],
        rendered_boxes: Dict[str, List[float]],
        pred_render: np.ndarray,
        gt_render: np.ndarray,
    ) -> None:
        out_root_raw = str(os.environ.get("DESIGN_BENCHMARKS_LAYOUT3_SAVE_RENDERS_DIR", "")).strip()
        if not out_root_raw:
            return

        try:
            from PIL import Image
        except ImportError:
            return

        out_root = Path(out_root_raw).resolve()
        benchmark_id = cls._safe_fs_name(str(gt.get("_benchmark_id") or "layout-2"))
        model_name = cls._safe_fs_name(str(gt.get("_model_name") or "model"))
        mode = cls._safe_fs_name(str(gt.get("placement_mode") or "unknown"))
        sample_id = cls._safe_fs_name(
            str(gt.get("sample_id") or gt.get("_sample_id") or "sample")
        )
        out_dir = out_root / benchmark_id / model_name / mode
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            pred_u8 = pred_render.astype(np.uint8, copy=False)
            gt_u8 = gt_render.astype(np.uint8, copy=False)
            Image.fromarray(pred_u8, mode="RGB").save(out_dir / f"{sample_id}__pred.png")
            Image.fromarray(gt_u8, mode="RGB").save(out_dir / f"{sample_id}__gt.png")
            side = np.concatenate([pred_u8, gt_u8], axis=1)
            Image.fromarray(side, mode="RGB").save(out_dir / f"{sample_id}__pred_vs_gt.png")
        except Exception:
            # Keep evaluation robust even when debug-asset saving fails.
            return

        debug = {
            "sample_id": str(gt.get("sample_id") or gt.get("_sample_id") or ""),
            "benchmark_id": str(gt.get("_benchmark_id") or "layout-2"),
            "model_name": str(gt.get("_model_name") or ""),
            "placement_mode": str(gt.get("placement_mode") or ""),
            "canvas_width": int(gt.get("canvas_width") or 0),
            "canvas_height": int(gt.get("canvas_height") or 0),
            "pred_raw": pred_raw,
            "pred_map": pred_map,
            "pred_order": pred_order,
            "rendered_boxes": rendered_boxes,
            "gt_components": gt.get("components") or [],
        }
        try:
            (out_dir / f"{sample_id}__meta.json").write_text(
                json.dumps(debug, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception:
            return

    @classmethod
    def _load_rgba_image(
        cls,
        source: str,
        *,
        size: Optional[Tuple[int, int]] = None,
    ) -> Optional[Any]:
        if not source:
            return None
        try:
            import requests
            from PIL import Image
        except ImportError:
            return None

        pil = None
        src = str(source).strip()
        if not src:
            return None
        if src.startswith(("http://", "https://")):
            try:
                resp = requests.get(src, timeout=20)
                resp.raise_for_status()
                pil = Image.open(io.BytesIO(resp.content))
            except Exception:
                return None
        else:
            p = Path(src)
            if not p.is_file():
                return None
            try:
                pil = Image.open(p)
            except Exception:
                return None
        if pil is None:
            return None
        pil = pil.convert("RGBA")
        if size is not None and pil.size != size:
            pil = pil.resize(size, Image.BILINEAR)
        return pil


# ===========================================================================
# Helpers for layout-4 through layout-7
# ===========================================================================


def _macro_f1_precision_recall(
    predictions: List[Any], ground_truth: List[Any],
) -> Dict[str, float]:
    labels = sorted({str(g) for g in ground_truth} | {str(p) for p in predictions})
    if not labels:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}
    preds_s = [str(p) for p in predictions]
    gts_s = [str(g) for g in ground_truth]
    per_f1: List[float] = []
    per_p: List[float] = []
    per_r: List[float] = []
    for lab in labels:
        tp = sum(1 for p, g in zip(preds_s, gts_s) if p == lab and g == lab)
        fp = sum(1 for p, g in zip(preds_s, gts_s) if p == lab and g != lab)
        fn = sum(1 for p, g in zip(preds_s, gts_s) if p != lab and g == lab)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_f1.append(f1)
        per_p.append(prec)
        per_r.append(rec)
    nl = max(len(per_f1), 1)
    return {
        "f1": sum(per_f1) / nl,
        "precision": sum(per_p) / nl,
        "recall": sum(per_r) / nl,
    }


def _bbox_iou(pred: List[float], gt: List[float]) -> float:
    px, py, pw, ph = pred
    gx, gy, gw, gh = gt
    x1, y1 = max(px, gx), max(py, gy)
    x2, y2 = min(px + pw, gx + gw), min(py + ph, gy + gh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    union = pw * ph + gw * gh - inter
    return inter / union if union > 0 else 0.0


def _detection_class_ap(
    predictions: List[List[Dict]],
    ground_truth: List[List[Dict]],
    label: str,
    iou_threshold: float,
) -> float:
    all_preds: list = []
    total_gt = 0
    for img_i, (preds, gts) in enumerate(zip(predictions, ground_truth)):
        for p in preds:
            if p.get("label") == label:
                all_preds.append((img_i, p["bbox"], float(p.get("score", 0.0))))
        total_gt += sum(1 for g in gts if g.get("label") == label)
    if total_gt == 0:
        return 0.0
    all_preds.sort(key=lambda x: x[2], reverse=True)
    matched: Dict[int, set] = {}
    tp_cum = fp_cum = 0
    prec_rec: list = []
    for img_i, bbox, _score in all_preds:
        if img_i not in matched:
            matched[img_i] = set()
        img_gts = [
            (j, g) for j, g in enumerate(ground_truth[img_i]) if g.get("label") == label
        ]
        best_val, best_j = 0.0, -1
        for j, g in img_gts:
            v = _bbox_iou(bbox, g["bbox"])
            if v > best_val:
                best_val, best_j = v, j
        if best_val >= iou_threshold and best_j not in matched[img_i]:
            tp_cum += 1
            matched[img_i].add(best_j)
        else:
            fp_cum += 1
        prec_rec.append((tp_cum / (tp_cum + fp_cum), tp_cum / total_gt))
    if not prec_rec:
        return 0.0
    ap = 0.0
    for t in (i / 10.0 for i in range(11)):
        ap += max((p for p, r in prec_rec if r >= t), default=0.0) / 11.0
    return ap


def _detection_map(
    predictions: List[List[Dict]], ground_truth: List[List[Dict]],
) -> Dict[str, float]:
    iou_thresholds = [0.5 + 0.05 * i for i in range(10)]
    all_labels: set = set()
    for gts in ground_truth:
        for g in gts:
            all_labels.add(g["label"])
    if not all_labels:
        return {"mAP@0.5": 0.0, "mAP@0.5:0.95": 0.0}
    ap_at_05: Dict[str, float] = {}
    per_thresh: List[float] = []
    for threshold in iou_thresholds:
        class_aps: List[float] = []
        for label in sorted(all_labels):
            ap = _detection_class_ap(predictions, ground_truth, label, threshold)
            class_aps.append(ap)
            if abs(threshold - 0.5) < 1e-6:
                ap_at_05[label] = ap
        per_thresh.append(sum(class_aps) / max(len(class_aps), 1))
    result: Dict[str, float] = {
        "mAP@0.5": per_thresh[0] if per_thresh else 0.0,
        "mAP@0.5:0.95": sum(per_thresh) / max(len(per_thresh), 1),
    }
    for label, ap_val in ap_at_05.items():
        result[f"AP@0.5_{label}"] = ap_val
    return result


@benchmark
class AspectRatioAdaptation(PartialLayoutCompletion):
    """layout-3 -- Adapt a layout to a different aspect ratio."""

    meta = BenchmarkMeta(
        id="layout-3",
        name="Aspect-Ratio Adaptation",
        task_type=TaskType.GENERATION,
        domain="layout",
        data_subpath="layout/layout4-multi-aspect-ratio",
        description="Adapt a layout from long canvas to square canvas",
        input_spec=(
            "Source composite image (long ratio) + separated component assets + "
            "target square canvas size"
        ),
        output_spec=(
            "Either layout JSON (component bboxes) or directly generated adapted image"
        ),
        metrics=[
            "miou",
            "component_coverage",
            "reverse_iou",
            "element_recall",
            "hallucination_rate",
            "mjudge_win_rate",
            "clip_score",
            "dino_score",
            "lpips",
            "dreamsim_distance",
            "nima_score",
            "hpsv3",
            "hpsv2",
            "imagereward",
            "pickscore",
            "fid",
        ],
    )

    PAIR_MANIFEST_ENV = "DESIGN_BENCHMARKS_LAYOUT4_PAIR_MANIFEST"
    DIRECTION_ENV = "DESIGN_BENCHMARKS_LAYOUT4_DIRECTION"
    DIRECT_EVAL_SIZE_ENV = "DESIGN_BENCHMARKS_LAYOUT4_DIRECT_EVAL_SIZE"
    DEFAULT_DIRECTION = "long_to_short"
    DEFAULT_DIRECT_EVAL_SIZE = (1024, 1024)
    DIRECT_MATCH_THRESHOLD = 0.35
    DIRECT_IOU_MATCH_THRESHOLD = 0.10
    DIRECT_TEMPLATE_SCALES = (0.80, 1.00, 1.20)

    def load_data(
        self,
        data_dir: Union[str, Path],
        *,
        n: Optional[int] = None,
        dataset_root: Optional[Union[str, Path]] = None,
    ) -> List[Dict[str, Any]]:
        _ = dataset_root
        root = Path(data_dir).resolve()
        manifest_path = self._resolve_pair_manifest_path(root)
        if manifest_path is None:
            raise FileNotFoundError(
                "layout-3 pair manifest not found. "
                "Expected g4_firestore_image_gen_pairs_manifest*.json under "
                f"{root}/manifests or set {self.PAIR_MANIFEST_ENV}."
            )

        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Failed to parse pair manifest {manifest_path}: {exc}") from exc

        pair_rows = payload.get("pairs", payload) if isinstance(payload, dict) else payload
        if not isinstance(pair_rows, list):
            raise ValueError(f"Invalid pair manifest format: {manifest_path}")

        direction = str(os.environ.get(self.DIRECTION_ENV, self.DEFAULT_DIRECTION)).strip().lower()
        if direction not in {"long_to_short", "short_to_long"}:
            direction = self.DEFAULT_DIRECTION

        samples: List[Dict[str, Any]] = []
        for row in pair_rows:
            sample = self._make_adaptation_sample(row=row, root=root, direction=direction)
            if sample is None:
                continue
            samples.append(sample)
            if n is not None and len(samples) >= n:
                break

        if not samples:
            raise ValueError(
                "No valid layout-3 samples found. "
                "Check pair manifest entries and component_renders directories."
            )
        return samples

    def build_model_input(self, sample: Dict[str, Any], *, modality: Any = None) -> Any:
        from design_benchmarks.models.base import ModelInput, Modality

        source_w = int(sample.get("source_canvas_width", 0))
        source_h = int(sample.get("source_canvas_height", 0))
        target_w = int(sample.get("canvas_width", 0))
        target_h = int(sample.get("canvas_height", 0))
        source_image = str(sample.get("source_image") or "").strip()

        if modality == Modality.IMAGE_GENERATION:
            images: List[str] = []
            if source_image:
                images.append(source_image)
            eval_w, eval_h = self._layout4_direct_eval_size()
            # Direct image adaptation is image-only for visual inputs:
            # - source composite image only (no component references)
            # Text instruction remains detailed, but must not include
            # JSON-derived explicit text snippets.
            prompt_sample = dict(sample)
            prompt_sample["prompt"] = self.DEFAULT_PROMPT
            prompt_sample["expected_texts"] = []
            prompt = self._compose_direct_adaptation_prompt(
                prompt_sample,
                with_source_image=bool(source_image),
                direct_components=[],
                eval_size=(eval_w, eval_h),
            )
            # Force image-edit style conditioning using full editable mask.
            # This ensures source image guidance is actually consumed by edit-capable APIs.
            mask_h = max(8, source_h)
            mask_w = max(8, source_w)
            full_edit_mask: Any = None
            try:
                from PIL import Image

                full_edit_mask = Image.fromarray(
                    np.full((mask_h, mask_w), 255, dtype=np.uint8),
                    mode="L",
                )
            except Exception:
                full_edit_mask = None
            metadata: Dict[str, Any] = {
                "benchmark_id": self.meta.id,
                "task": "layout_aspect_ratio_adaptation_image",
                "sample_id": str(sample.get("sample_id") or ""),
                "source_width": source_w,
                "source_height": source_h,
                "dataset_target_width": target_w,
                "dataset_target_height": target_h,
                "target_width": eval_w,
                "target_height": eval_h,
                "width": eval_w,
                "height": eval_h,
                "layout4_source_only_input": True,
            }
            if full_edit_mask is not None:
                metadata["mask"] = full_edit_mask
            return ModelInput(
                text=prompt,
                images=images,
                metadata=metadata,
            )

        images = []
        if modality != Modality.TEXT:
            if source_image:
                images.append(source_image)
            images.extend(
                str(c.get("image_path") or "")
                for c in sample.get("components", [])
                if str(c.get("image_path") or "").strip()
            )

        prompt = self._compose_coordinate_adaptation_prompt(sample, with_images=bool(images))
        metadata = {
            "benchmark_id": self.meta.id,
            "task": "layout_aspect_ratio_adaptation_coordinates",
            "source_width": source_w,
            "source_height": source_h,
            "target_width": target_w,
            "target_height": target_h,
            "component_keys": [str(c.get("component_key") or "") for c in sample.get("components", [])],
            "placement_mode": "multiple",
        }
        return ModelInput(text=prompt, images=images, metadata=metadata)

    def parse_model_output(self, output: Any) -> Any:
        if output is None:
            return {"components": []}

        if isinstance(output, (str, Path)):
            raw = str(output).strip()
            p = Path(raw)
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                return str(p.resolve())

        images = getattr(output, "images", None)
        if isinstance(images, list) and images:
            arr = self._to_rgb_array(images[0])
            if arr is not None:
                eval_w, eval_h = self._layout4_direct_eval_size()
                return self._resize_to_match(arr, (eval_h, eval_w))

        return super().parse_model_output(output)

    def evaluate(self, predictions: List[Any], ground_truth: List[Any]) -> Dict[str, float]:
        from design_benchmarks.metrics.core import iou as iou_metric

        miou_scores: List[float] = []
        coverage_scores: List[float] = []
        reverse_iou_scores: List[float] = []
        recall_scores: List[float] = []
        hallucination_scores: List[float] = []
        clip_scores: List[float] = []
        dino_scores: List[float] = []
        lpips_scores: List[float] = []
        dreamsim_scores: List[float] = []
        nima_scores: List[float] = []
        mjudge_scores: List[float] = []
        hpsv3_scores: List[float] = []
        hpsv2_scores: List[float] = []
        imagereward_scores: List[float] = []
        pickscore_scores: List[float] = []
        real_features: List[np.ndarray] = []
        gen_features: List[np.ndarray] = []

        mjudge_attempted = 0
        coord_attempted = 0
        coord_evaluated = 0
        reverse_attempted = 0
        reverse_evaluated = 0
        evaluated = 0
        direct_eval_w, direct_eval_h = self._layout4_direct_eval_size()

        for pred_raw, gt_raw in zip(predictions, ground_truth):
            gt = self._normalize_adaptation_gt(gt_raw)
            gt_components = gt.get("components", [])
            if not gt_components:
                continue

            gt_render = self._to_rgb_array(gt.get("ground_truth_image", ""))
            if gt_render is None:
                gt_render = self._render_layout_from_boxes(
                    gt=gt,
                    component_boxes={c["component_key"]: c["bbox"] for c in gt_components},
                    fallback_order=[],
                )
            if gt_render is None:
                continue

            pred_render: Optional[np.ndarray] = None
            metrics_gt_render: np.ndarray = gt_render
            local_miou = float("nan")
            local_cov = float("nan")
            local_reverse_iou = float("nan")
            local_recall = float("nan")
            local_hallucination = float("nan")

            if self._looks_like_layout_prediction(pred_raw):
                coord_attempted += 1
                pred_map, pred_order = self._normalize_prediction_map(
                    pred_raw,
                    canvas_width=gt["canvas_width"],
                    canvas_height=gt["canvas_height"],
                )

                ious: List[float] = []
                matched = 0
                rendered_boxes: Dict[str, List[float]] = {}
                for idx, comp in enumerate(gt_components):
                    pbox = (
                        pred_map.get(comp["component_key"])
                        or pred_map.get(comp["component_id"])
                        or (pred_order[idx] if idx < len(pred_order) else None)
                    )
                    if pbox is None:
                        ious.append(0.0)
                        continue
                    matched += 1
                    ious.append(float(iou_metric(pbox, comp["bbox"])))
                    rendered_boxes[comp["component_key"]] = pbox

                if ious:
                    local_miou = float(sum(ious) / len(ious))
                    local_cov = float(matched / len(gt_components))
                    local_reverse_iou = local_miou
                    local_recall = local_cov
                    pred_count = max(len(pred_order), len(pred_map))
                    local_hallucination = float(
                        max(pred_count - matched, 0) / max(pred_count, 1)
                    )
                    coord_evaluated += 1
                    evaluated += 1

                pred_render = self._render_layout_from_boxes(
                    gt=gt,
                    component_boxes=rendered_boxes,
                    fallback_order=pred_order,
                )
            else:
                reverse_attempted += 1
                pred_render = self._to_rgb_array(pred_raw)
                metrics_gt_render = self._resize_to_match(
                    gt_render,
                    (direct_eval_h, direct_eval_w),
                )
                if pred_render is not None:
                    pred_render = self._resize_to_match(
                        pred_render,
                        (direct_eval_h, direct_eval_w),
                    )

                if pred_render is not None:
                    scaled_gt_components = self._rescale_components_for_canvas(
                        gt_components=gt_components,
                        from_canvas=(gt["canvas_width"], gt["canvas_height"]),
                        to_canvas=(direct_eval_w, direct_eval_h),
                    )
                    recovered_boxes, _ = self._recover_boxes_from_templates(
                        image=pred_render,
                        gt_components=scaled_gt_components,
                        canvas_width=direct_eval_w,
                        canvas_height=direct_eval_h,
                    )

                    ious: List[float] = []
                    detected = 0
                    matched = 0
                    for comp in scaled_gt_components:
                        rbox = recovered_boxes.get(comp["component_key"])
                        if rbox is None:
                            ious.append(0.0)
                            continue
                        detected += 1
                        iou_val = float(iou_metric(rbox, comp["bbox"]))
                        ious.append(iou_val)
                        if iou_val >= self.DIRECT_IOU_MATCH_THRESHOLD:
                            matched += 1

                    if ious:
                        local_reverse_iou = float(sum(ious) / len(ious))
                        local_recall = float(detected / len(gt_components))
                        local_hallucination = float(
                            max(detected - matched, 0) / max(detected, 1)
                        )
                        local_miou = local_reverse_iou
                        local_cov = local_recall
                        reverse_evaluated += 1
                        evaluated += 1

            self._append_if_finite(miou_scores, local_miou)
            self._append_if_finite(coverage_scores, local_cov)
            self._append_if_finite(reverse_iou_scores, local_reverse_iou)
            self._append_if_finite(recall_scores, local_recall)
            self._append_if_finite(hallucination_scores, local_hallucination)

            if pred_render is None:
                continue

            pred_render = self._resize_to_match(pred_render, metrics_gt_render.shape[:2])
            real_features.append(self._feature_vector(metrics_gt_render))
            gen_features.append(self._feature_vector(pred_render))

            prompt = str(gt.get("aesthetic_prompt") or gt.get("prompt") or "")
            clip = self._clip_score(prompt, pred_render)
            dino = LayerAwareObjectInsertion._dino_similarity(pred_render, metrics_gt_render)
            lpips = LayerAwareObjectInsertion._lpips_distance(pred_render, metrics_gt_render)
            dreamsim = LayerAwareObjectInsertion._dreamsim_distance(pred_render, metrics_gt_render)
            nima = self._nima_score(pred_render)
            mjudge_attempted += 1
            mjudge = self._mjudge_pairwise_win_rate(
                prompt=prompt,
                pred_image=pred_render,
                gt_image=metrics_gt_render,
                sample_id=str(gt.get("sample_id", "")),
            )
            hpsv3 = self._hpsv3_score(prompt, pred_render, clip_fallback=clip)
            hpsv2 = self._hpsv2_score(prompt, pred_render, clip_fallback=clip)
            imagereward = self._imagereward_score(prompt, pred_render)
            pickscore = self._pick_score(prompt, pred_render)

            self._append_if_finite(clip_scores, clip)
            self._append_if_finite(dino_scores, dino)
            self._append_if_finite(lpips_scores, lpips)
            self._append_if_finite(dreamsim_scores, dreamsim)
            self._append_if_finite(nima_scores, nima)
            self._append_if_finite(mjudge_scores, mjudge)
            self._append_if_finite(hpsv3_scores, hpsv3)
            self._append_if_finite(hpsv2_scores, hpsv2)
            self._append_if_finite(imagereward_scores, imagereward)
            self._append_if_finite(pickscore_scores, pickscore)

        fid_score = float("nan")
        if len(real_features) >= 2:
            try:
                fid_score = float(fid_metric(np.stack(real_features), np.stack(gen_features)))
            except Exception:
                fid_score = float("nan")

        return {
            "miou": self._mean_or_nan(miou_scores),
            "component_coverage": self._mean_or_nan(coverage_scores),
            "reverse_iou": self._mean_or_nan(reverse_iou_scores),
            "element_recall": self._mean_or_nan(recall_scores),
            "hallucination_rate": self._mean_or_nan(hallucination_scores),
            "mjudge_win_rate": self._mean_or_nan(mjudge_scores),
            "clip_score": self._mean_or_nan(clip_scores),
            "dino_score": self._mean_or_nan(dino_scores),
            "lpips": self._mean_or_nan(lpips_scores),
            "dreamsim_distance": self._mean_or_nan(dreamsim_scores),
            "nima_score": self._mean_or_nan(nima_scores),
            "hpsv3": self._mean_or_nan(hpsv3_scores),
            "hpsv2": self._mean_or_nan(hpsv2_scores),
            "imagereward": self._mean_or_nan(imagereward_scores),
            "pickscore": self._mean_or_nan(pickscore_scores),
            "fid": fid_score,
            "evaluated_samples": float(evaluated),
            "coord_coverage": float(coord_evaluated / max(coord_attempted, 1)),
            "reverse_coverage": float(reverse_evaluated / max(reverse_attempted, 1)),
            "mjudge_coverage": len(mjudge_scores) / max(mjudge_attempted, 1),
            "fid_pair_count": float(len(real_features)),
        }

    def _normalize_adaptation_gt(self, raw: Any) -> Dict[str, Any]:
        out = self._normalize_visual_gt(raw)
        if isinstance(raw, dict):
            out["source_image"] = str(raw.get("source_image", raw.get("base_image", "")) or "")
            out["aesthetic_prompt"] = str(raw.get("aesthetic_prompt", raw.get("prompt", "")) or "")
            expected = raw.get("expected_texts", [])
            if isinstance(expected, str):
                expected = [expected]
            if not isinstance(expected, list):
                expected = []
            out["expected_texts"] = [str(v) for v in expected if str(v).strip()]
        else:
            out["source_image"] = ""
            out["aesthetic_prompt"] = str(out.get("prompt") or "")
            out["expected_texts"] = []
        return out

    @staticmethod
    def _looks_like_layout_prediction(pred_raw: Any) -> bool:
        if isinstance(pred_raw, dict):
            if isinstance(pred_raw.get("components"), list):
                return True
            if isinstance(pred_raw.get("placements"), list):
                return True
            layout_cfg = pred_raw.get("layout_config")
            if isinstance(layout_cfg, dict) and isinstance(layout_cfg.get("components"), list):
                return True
            return False
        if isinstance(pred_raw, list):
            return all(isinstance(x, dict) for x in pred_raw[:4])
        return False

    def _recover_boxes_from_templates(
        self,
        *,
        image: np.ndarray,
        gt_components: List[Dict[str, Any]],
        canvas_width: int,
        canvas_height: int,
    ) -> Tuple[Dict[str, List[float]], Dict[str, float]]:
        recovered: Dict[str, List[float]] = {}
        scores: Dict[str, float] = {}
        for comp in gt_components:
            key = str(comp.get("component_key") or "")
            expected = comp.get("bbox")
            if not key or not isinstance(expected, list) or len(expected) < 4:
                continue
            template_path = str(
                comp.get("source_image_path")
                or comp.get("image_path")
                or ""
            ).strip()
            box, score = self._template_match_bbox(
                image=image,
                template_path=template_path,
                expected_box=[float(v) for v in expected[:4]],
                canvas_width=canvas_width,
                canvas_height=canvas_height,
            )
            if box is None or not math.isfinite(score):
                continue
            if score < self.DIRECT_MATCH_THRESHOLD:
                continue
            recovered[key] = box
            scores[key] = score
        return recovered, scores

    def _template_match_bbox(
        self,
        *,
        image: np.ndarray,
        template_path: str,
        expected_box: List[float],
        canvas_width: int,
        canvas_height: int,
    ) -> Tuple[Optional[List[float]], float]:
        try:
            import cv2  # type: ignore[reportMissingImports]
        except Exception:
            return None, float("nan")

        template_rgb = self._load_template_rgb(template_path)
        if template_rgb is None:
            return None, float("nan")
        if image.ndim != 3 or image.shape[2] < 3:
            return None, float("nan")

        exp = self._clip_box(expected_box, canvas_width, canvas_height)
        if exp is None:
            return None, float("nan")
        ex, ey, ew, eh = [int(round(v)) for v in exp]
        ew = max(2, ew)
        eh = max(2, eh)

        H, W = image.shape[:2]
        margin_x = max(32, int(ew * 1.25))
        margin_y = max(32, int(eh * 1.25))
        x0 = max(0, ex - margin_x)
        y0 = max(0, ey - margin_y)
        x1 = min(W, ex + ew + margin_x)
        y1 = min(H, ey + eh + margin_y)
        search = image[y0:y1, x0:x1]
        if search.size == 0:
            return None, float("nan")

        search_gray = cv2.cvtColor(search.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        best_score = float("-inf")
        best_box: Optional[List[float]] = None

        for scale in self.DIRECT_TEMPLATE_SCALES:
            tw = max(2, int(round(ew * float(scale))))
            th = max(2, int(round(eh * float(scale))))
            if search_gray.shape[1] < tw or search_gray.shape[0] < th:
                continue
            resized = cv2.resize(template_rgb, (tw, th), interpolation=cv2.INTER_LINEAR)
            templ_gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
            try:
                result = cv2.matchTemplate(search_gray, templ_gray, cv2.TM_CCOEFF_NORMED)
            except Exception:
                continue
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val > best_score:
                bx = float(x0 + max_loc[0])
                by = float(y0 + max_loc[1])
                clipped = self._clip_box([bx, by, float(tw), float(th)], canvas_width, canvas_height)
                if clipped is not None:
                    best_score = float(max_val)
                    best_box = clipped

        if best_box is None:
            return None, float("nan")
        return best_box, float(best_score)

    def _load_template_rgb(self, template_path: str) -> Optional[np.ndarray]:
        src = str(template_path or "").strip()
        if not src:
            return None
        cache = getattr(self, "_layout4_template_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            setattr(self, "_layout4_template_cache", cache)
        if src in cache:
            cached = cache[src]
            return cached.copy() if isinstance(cached, np.ndarray) else None

        try:
            from PIL import Image
        except ImportError:
            cache[src] = None
            return None

        path = Path(src)
        if not path.is_file():
            cache[src] = None
            return None
        try:
            with Image.open(path) as pil:
                rgba = np.asarray(pil.convert("RGBA"), dtype=np.uint8)
        except Exception:
            cache[src] = None
            return None

        if rgba.ndim != 3 or rgba.shape[2] < 4:
            rgb = rgba[:, :, :3] if rgba.ndim == 3 else None
            if rgb is None:
                cache[src] = None
                return None
            cache[src] = rgb
            return rgb.copy()

        alpha = rgba[:, :, 3]
        ys, xs = np.where(alpha > 8)
        if ys.size == 0 or xs.size == 0:
            rgb = rgba[:, :, :3]
            cache[src] = rgb
            return rgb.copy()
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        cropped = rgba[y1:y2, x1:x2, :3]
        cache[src] = cropped
        return cropped.copy()

    def _resolve_pair_manifest_path(self, root: Path) -> Optional[Path]:
        env_path = str(os.environ.get(self.PAIR_MANIFEST_ENV, "")).strip()
        if env_path:
            p = Path(env_path).expanduser().resolve()
            if p.is_file():
                return p

        if root.is_file():
            return root

        candidates = [
            root / "manifests" / "g4_firestore_image_gen_pairs_manifest.filtered_component_renders.json",
            root / "manifests" / "g4_firestore_image_gen_pairs_manifest.json",
            root / "g4_firestore_image_gen_pairs_manifest.filtered_component_renders.json",
            root / "g4_firestore_image_gen_pairs_manifest.json",
            root.parent / "manifests" / "g4_firestore_image_gen_pairs_manifest.filtered_component_renders.json",
        ]
        for path in candidates:
            if path.is_file():
                return path
        return None

    def _make_adaptation_sample(
        self,
        *,
        row: Any,
        root: Path,
        direction: str,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(row, dict):
            return None

        side_a = self._load_pair_side(row.get("a"), root=root)
        side_b = self._load_pair_side(row.get("b"), root=root)
        if side_a is None or side_b is None:
            return None

        selected = self._select_directional_pair(side_a, side_b, direction=direction)
        if selected is None:
            return None
        source, target = selected

        source_assets = self._collect_component_assets(source["component_dir"], source["id"])
        target_assets = self._collect_component_assets(target["component_dir"], target["id"])
        if not source_assets or not target_assets:
            return None

        source_components = source["components"]
        target_components = target["components"]
        usable = min(
            len(source_assets),
            len(target_assets),
            len(source_components),
            len(target_components),
        )
        if usable <= 0:
            return None

        gt_components: List[Dict[str, Any]] = []
        for idx in range(usable):
            src_cfg = source_components[idx]
            tgt_cfg = target_components[idx]
            if not isinstance(src_cfg, dict) or not isinstance(tgt_cfg, dict):
                continue
            if not self._is_visual_component(src_cfg) and not self._is_visual_component(tgt_cfg):
                continue

            source_bbox = self._extract_bbox_from_component(
                src_cfg,
                canvas_width=source["width"],
                canvas_height=source["height"],
            )
            bbox = self._extract_bbox_from_component(
                tgt_cfg,
                canvas_width=target["width"],
                canvas_height=target["height"],
            )
            if bbox is None:
                continue

            source_asset = source_assets[idx]
            target_asset = target_assets[idx]
            comp_type = self._first_nonempty(
                tgt_cfg.get("type"),
                src_cfg.get("type"),
            ).upper()
            desc_cfg = tgt_cfg if self._is_visual_component(tgt_cfg) else src_cfg
            description = self._extract_component_description(
                desc_cfg,
                fallback=f"Adapted component {idx + 1}",
            )
            shape_hint = self._asset_shape_hint(source_asset)
            if shape_hint:
                description = f"{description} Visual cue: {shape_hint}."

            gt_components.append(
                {
                    "component_key": f"C{len(gt_components) + 1}",
                    "component_id": f"{target['id']}_component_{idx:03d}",
                    "bbox": [float(v) for v in bbox],
                    "z_index": int(idx),
                    "component_type": str(comp_type or ""),
                    # For reverse/image branch, keep target crop as template reference.
                    "source_image_path": str(target_asset),
                    # For coordinate branch render, place source crop on target canvas.
                    "image_path": str(source_asset),
                    "source_bbox": [float(v) for v in source_bbox] if source_bbox is not None else None,
                    "description": description,
                    "bbox_source": "json_style",
                    "bbox_agreement_iou": float("nan"),
                }
            )

        if not gt_components:
            return None

        pair_id = str(row.get("pair_id") or f"{source['id']}_to_{target['id']}")
        sample_id = f"{pair_id}_{source['id']}_to_{target['id']}"

        raw_intent = self._first_nonempty(
            source["intent"],
            target["intent"],
            source["description"],
            target["description"],
            self.DEFAULT_PROMPT,
        )
        expected_texts = self._extract_texts(target["layout_row"].get("layout_config", {}))
        aesthetic_prompt = self._compose_layout4_aesthetic_prompt(
            intent=raw_intent,
            expected_texts=expected_texts,
            source_size=(source["width"], source["height"]),
            target_size=(target["width"], target["height"]),
        )

        gt_bundle = {
            "sample_id": sample_id,
            "pair_id": pair_id,
            "prompt": raw_intent,
            "aesthetic_prompt": aesthetic_prompt,
            "expected_texts": expected_texts,
            "source_image": source["image_path"],
            "base_image": source["image_path"],
            "ground_truth_image": target["image_path"],
            "canvas_width": int(target["width"]),
            "canvas_height": int(target["height"]),
            "placement_mode": "multiple",
            "components": gt_components,
        }

        return {
            "sample_id": sample_id,
            "pair_id": pair_id,
            "source_id": source["id"],
            "target_id": target["id"],
            "prompt": raw_intent,
            "aesthetic_prompt": aesthetic_prompt,
            "expected_texts": expected_texts,
            "source_image": source["image_path"],
            "target_image": target["image_path"],
            "source_canvas_width": int(source["width"]),
            "source_canvas_height": int(source["height"]),
            "canvas_width": int(target["width"]),
            "canvas_height": int(target["height"]),
            "placement_mode": "multiple",
            "components": gt_components,
            "ground_truth": gt_bundle,
        }

    def _load_pair_side(self, side: Any, *, root: Path) -> Optional[Dict[str, Any]]:
        if not isinstance(side, dict):
            return None
        sid = str(side.get("id") or "").strip()
        if not sid:
            return None

        layout_path = self._resolve_manifest_path(root, side.get("layout_path"))
        if not layout_path:
            layout_path = self._resolve_manifest_path(root, f"layouts/{sid}.json")
        if not layout_path:
            return None

        try:
            layout_row = json.loads(Path(layout_path).read_text(encoding="utf-8"))
        except Exception:
            return None

        width, height = self._extract_layout_size(layout_row)
        if width <= 0 or height <= 0:
            return None

        image_path = self._resolve_manifest_path(root, side.get("image_path"))
        if not image_path:
            image_path = self._resolve_manifest_path(root, f"images/{sid}.png")
        if not image_path:
            image_path = str(layout_row.get("layout_remotion_image_url") or "").strip()
        if not image_path:
            return None

        component_dir = self._resolve_component_render_dir(
            side.get("component_render_dir"),
            root=root,
            sample_id=sid,
        )

        semantic = layout_row.get("layout_semantic_description")
        if not isinstance(semantic, dict):
            semantic = {}

        return {
            "id": sid,
            "layout_row": layout_row,
            "components": (layout_row.get("layout_config") or {}).get("components", []) or [],
            "width": int(width),
            "height": int(height),
            "image_path": str(image_path),
            "component_dir": component_dir,
            "intent": self._first_nonempty(semantic.get("user_intent"), layout_row.get("user_intent")),
            "description": self._first_nonempty(semantic.get("description")),
        }

    def _resolve_component_render_dir(self, value: Any, *, root: Path, sample_id: str) -> Path:
        raw = str(value or "").strip()
        if raw:
            p = Path(raw)
            if p.is_dir():
                return p.resolve()
            candidate = (root / raw).resolve()
            if candidate.is_dir():
                return candidate
        return (root / "component_renders" / sample_id).resolve()

    def _collect_component_assets(self, component_dir: Path, sample_id: str) -> List[str]:
        if not component_dir.is_dir():
            return []
        pattern = re.compile(
            rf"^{re.escape(sample_id)}_component_(\d+)\.(png|jpg|jpeg|webp)$",
            re.IGNORECASE,
        )
        indexed: List[Tuple[int, str]] = []
        for path in component_dir.iterdir():
            if not path.is_file():
                continue
            m = pattern.match(path.name)
            if not m:
                continue
            indexed.append((int(m.group(1)), str(path.resolve())))
        indexed.sort(key=lambda item: item[0])
        return [p for _, p in indexed]

    def _extract_layout_size(self, layout_row: Dict[str, Any]) -> Tuple[int, int]:
        meta = layout_row.get("layout_metadata")
        if not isinstance(meta, dict):
            meta = {}
        width = self._safe_int(meta.get("width"), 0)
        height = self._safe_int(meta.get("height"), 0)
        if width > 0 and height > 0:
            return width, height

        style = ((layout_row.get("layout_config") or {}).get("style") or {})
        if isinstance(style, dict):
            width = self._safe_int(self._parse_number(style.get("width"), 0.0), 0)
            height = self._safe_int(self._parse_number(style.get("height"), 0.0), 0)
        return width, height

    @staticmethod
    def _select_directional_pair(
        side_a: Dict[str, Any],
        side_b: Dict[str, Any],
        *,
        direction: str,
    ) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        aw, ah = int(side_a.get("width", 0)), int(side_a.get("height", 0))
        bw, bh = int(side_b.get("width", 0)), int(side_b.get("height", 0))
        if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
            return None

        a_ratio = float(aw) / float(ah)
        b_ratio = float(bw) / float(bh)
        a_portrait = ah > aw
        b_portrait = bh > bw

        if direction == "short_to_long":
            if not a_portrait and b_portrait:
                return side_a, side_b
            if not b_portrait and a_portrait:
                return side_b, side_a
            return (side_a, side_b) if a_ratio > b_ratio else (side_b, side_a)

        # long_to_short (default)
        if a_portrait and not b_portrait:
            return side_a, side_b
        if b_portrait and not a_portrait:
            return side_b, side_a
        return (side_a, side_b) if a_ratio < b_ratio else (side_b, side_a)

    def _compose_layout4_aesthetic_prompt(
        self,
        *,
        intent: str,
        expected_texts: List[str],
        source_size: Tuple[int, int],
        target_size: Tuple[int, int],
    ) -> str:
        clean_intent = re.sub(r"\s+", " ", str(intent or "")).strip()
        if not clean_intent:
            clean_intent = "Adapt the poster layout while preserving hierarchy and readability."

        snippets = [re.sub(r"\s+", " ", str(t)).strip() for t in expected_texts if str(t).strip()]
        if snippets:
            snippet_text = ", ".join(f'"{s}"' for s in snippets[:8])
            return (
                f"{clean_intent} Adapt from {source_size[0]}x{source_size[1]} to "
                f"{target_size[0]}x{target_size[1]}. Preserve these texts: {snippet_text}."
            )
        return (
            f"{clean_intent} Adapt from {source_size[0]}x{source_size[1]} to "
            f"{target_size[0]}x{target_size[1]} while preserving visual hierarchy."
        )

    def _compose_coordinate_adaptation_prompt(
        self,
        sample: Dict[str, Any],
        *,
        with_images: bool,
    ) -> str:
        source_w = int(sample.get("source_canvas_width", 0))
        source_h = int(sample.get("source_canvas_height", 0))
        target_w = int(sample.get("canvas_width", 0))
        target_h = int(sample.get("canvas_height", 0))
        prompt_raw = sample.get("prompt")
        prompt = self.DEFAULT_PROMPT if prompt_raw is None else str(prompt_raw).strip()
        expected_texts = sample.get("expected_texts", [])
        if not isinstance(expected_texts, list):
            expected_texts = []

        lines = [
            "You are a responsive layout engine.",
            f"Task: adapt a design from {source_w}x{source_h} to {target_w}x{target_h}.",
            f"Intent: {prompt}",
            "",
            "Output one bbox [x,y,w,h] for each component key on the TARGET canvas.",
            "Preserve visual hierarchy, reading order, and relative grouping.",
            "Keep all boxes within target bounds.",
            "Return strict JSON only.",
        ]
        if with_images:
            lines.extend(
                [
                    "",
                    "Input mapping:",
                    "- Image #1 is the source composite.",
                    "- Images #2..#(N+1) are component assets to place.",
                ]
            )
        if expected_texts:
            lines.append("")
            lines.append("Text constraints (keep legible):")
            for t in expected_texts[:12]:
                norm = self._normalize_text_constraint(t)
                if norm:
                    lines.append(f'- "{norm}"')

        lines.append("")
        lines.append("Components:")
        for comp in sample.get("components", []):
            key = str(comp.get("component_key") or "")
            ctype = str(comp.get("component_type") or "UNKNOWN")
            desc = self._normalize_component_description_text(comp.get("description")) or key
            lines.append(f"- {key} type={ctype}: {desc}")

        lines.extend(
            [
                "",
                "JSON schema:",
                "{",
                '  "layout_config": {',
                f'    "width": {target_w},',
                f'    "height": {target_h},',
                '    "components": [',
                '      {"component_key": "C1", "bbox": [x, y, w, h]}',
                "    ]",
                "  }",
                "}",
            ]
        )
        return "\n".join(lines)

    def _compose_direct_adaptation_prompt(
        self,
        sample: Dict[str, Any],
        *,
        with_source_image: bool,
        direct_components: List[Dict[str, Any]],
        eval_size: Tuple[int, int],
    ) -> str:
        source_w = int(sample.get("source_canvas_width", 0))
        source_h = int(sample.get("source_canvas_height", 0))
        dataset_target_w = int(sample.get("canvas_width", 0))
        dataset_target_h = int(sample.get("canvas_height", 0))
        target_w, target_h = int(eval_size[0]), int(eval_size[1])
        _ = direct_components  # direct track uses source-only visual input

        lines = [
            "You are a professional design retargeting engine.",
            "",
            "Task:",
            (
                f"- Retarget the SAME design from {source_w}x{source_h} "
                f"to {target_w}x{target_h} (square)."
            ),
            f"- Reference dataset target ratio is {dataset_target_w}x{dataset_target_h}.",
            "- This is aspect-ratio adaptation, NOT a redesign.",
            "",
            "Input mapping:",
            (
                "- Image #1 is the source composite image (single source of truth)."
                if with_source_image
                else "- Use the provided source composite image as single source of truth."
            ),
            "",
            "Non-negotiable constraints:",
            "- Preserve the same scene, brand identity, visual assets, and overall style.",
            "- Preserve all visible source text faithfully (no rewriting, no translation, no paraphrase, no new copy).",
            "- Preserve visual hierarchy, reading order, and semantic grouping.",
            "- Keep key elements present; do not drop major content.",
            "- Do not invent new logos, slogans, objects, or decorative concepts.",
            "",
            "Allowed edits:",
            "- Reposition/scale/line-break existing elements only as needed for square composition.",
            "- Re-balance spacing for a natural 1:1 layout.",
            "- Extend background only when necessary for ratio retargeting.",
            "",
            "Forbidden:",
            "- New concept, new campaign message, new style direction, or creative reinterpretation.",
            "",
            "If any instruction conflicts, prioritize source fidelity over creativity.",
            "",
            "Output requirements:",
            f"- Return exactly one natural-looking {target_w}x{target_h} image.",
            "- No border/frame unless implied by source design.",
        ]
        return "\n".join(lines)

    @classmethod
    def _layout4_direct_eval_size(cls) -> Tuple[int, int]:
        raw = str(os.environ.get(cls.DIRECT_EVAL_SIZE_ENV, "")).strip().lower()
        if raw:
            m = re.match(r"^\s*(\d+)\s*[x:]\s*(\d+)\s*$", raw)
            if m:
                w = cls._safe_int(m.group(1), cls.DEFAULT_DIRECT_EVAL_SIZE[0])
                h = cls._safe_int(m.group(2), cls.DEFAULT_DIRECT_EVAL_SIZE[1])
                if w > 0 and h > 0:
                    return int(w), int(h)
        return int(cls.DEFAULT_DIRECT_EVAL_SIZE[0]), int(cls.DEFAULT_DIRECT_EVAL_SIZE[1])

    def _rescale_components_for_canvas(
        self,
        *,
        gt_components: List[Dict[str, Any]],
        from_canvas: Tuple[int, int],
        to_canvas: Tuple[int, int],
    ) -> List[Dict[str, Any]]:
        from_w, from_h = int(from_canvas[0]), int(from_canvas[1])
        to_w, to_h = int(to_canvas[0]), int(to_canvas[1])
        if from_w <= 0 or from_h <= 0 or to_w <= 0 or to_h <= 0:
            return [dict(c) for c in gt_components if isinstance(c, dict)]

        sx = float(to_w) / float(from_w)
        sy = float(to_h) / float(from_h)
        out: List[Dict[str, Any]] = []
        for comp in gt_components:
            if not isinstance(comp, dict):
                continue
            cloned = dict(comp)
            bbox = cloned.get("bbox")
            if isinstance(bbox, list) and len(bbox) >= 4:
                scaled = [
                    float(bbox[0]) * sx,
                    float(bbox[1]) * sy,
                    float(bbox[2]) * sx,
                    float(bbox[3]) * sy,
                ]
                clipped = self._clip_box(scaled, to_w, to_h)
                if clipped is not None:
                    cloned["bbox"] = [float(v) for v in clipped]
            out.append(cloned)
        return out


# ===========================================================================
# Implemented — layout-4 through layout-7
# ===========================================================================

VALID_RATIOS = ["1:1", "16:9", "9:16", "4:3", "3:4", "4:5", "5:4", "2:3", "3:2", "21:9"]
VALID_COMP_TYPES = ["text", "image", "vector", "group"]


@benchmark
class AspectRatioClassification(BaseBenchmark):
    """layout-4 — Classify canvas aspect ratio from ~10 common ratios."""

    pipeline_implemented = True

    meta = BenchmarkMeta(
        id="layout-4",
        name="Aspect Ratio Classification",
        task_type=TaskType.UNDERSTANDING,
        domain="layout",
        data_subpath="layout/AspectRatioClassification",
        description="Classify canvas aspect ratio from ~10 common ratios",
        input_spec="Rendered layout image",
        output_spec="Aspect ratio label (e.g. 16:9)",
        metrics=["accuracy", "f1", "precision", "recall"],
        tags=["P1", "low-complexity"],
    )

    def load_data(self, data_dir, *, n=None, dataset_root: Union[str, Path]):
        return load_csv_samples(data_dir, n=n, dataset_root=dataset_root)

    def build_model_input(self, sample, *, modality=None):
        return build_vision_input(sample, modality=modality)

    def parse_model_output(self, output):
        text = output.text.strip().lower().replace(" ", "")
        for ratio in VALID_RATIOS:
            if ratio.lower() in text:
                return ratio
        return text

    def evaluate(self, predictions, ground_truth):
        n = max(len(predictions), 1)
        correct = sum(1 for p, g in zip(predictions, ground_truth) if str(p) == str(g))
        result: Dict[str, float] = {"accuracy": correct / n}
        result.update(_macro_f1_precision_recall(predictions, ground_truth))
        return result


def _safe_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


@benchmark
class ComponentCount(BaseBenchmark):
    """layout-5 — Count the number of visible components in a layout."""

    pipeline_implemented = True

    meta = BenchmarkMeta(
        id="layout-5",
        name="Component Count",
        task_type=TaskType.UNDERSTANDING,
        domain="layout",
        data_subpath="layout/ComponentCount",
        description="Count the number of visible components in a layout",
        input_spec="Rendered layout image",
        output_spec="Integer component count",
        metrics=["mae", "mse"],
        tags=["P1", "low-complexity"],
    )

    def load_data(self, data_dir, *, n=None, dataset_root: Union[str, Path]):
        return load_csv_samples(data_dir, n=n, dataset_root=dataset_root)

    def build_model_input(self, sample, *, modality=None):
        return build_vision_input(sample, modality=modality)

    def parse_model_output(self, output):
        nums = re.findall(r"\d+", output.text.strip())
        return int(nums[0]) if nums else -1

    def evaluate(self, predictions, ground_truth):
        pairs = [
            (pf, gf)
            for p, g in zip(predictions, ground_truth)
            if (pf := _safe_float(p)) is not None and (gf := _safe_float(g)) is not None
        ]
        if not pairs:
            return {"mae": 0.0, "mse": 0.0}
        mae = sum(abs(pf - gf) for pf, gf in pairs) / len(pairs)
        mse = sum((pf - gf) ** 2 for pf, gf in pairs) / len(pairs)
        return {"mae": mae, "mse": mse}


@benchmark
class ComponentClassification(BaseBenchmark):
    """layout-6 — Classify component type (text / image / vector / group)."""

    pipeline_implemented = True

    meta = BenchmarkMeta(
        id="layout-6",
        name="Component Classification",
        task_type=TaskType.UNDERSTANDING,
        domain="layout",
        data_subpath="layout/ComponentClassification",
        description="Classify component type (text, image, vector, or group) from layout and coordinates",
        input_spec="Rendered layout image + component location",
        output_spec="Component type label",
        metrics=["accuracy", "f1_macro"],
        tags=["P1", "low-complexity"],
    )

    def load_data(self, data_dir, *, n=None, dataset_root: Union[str, Path]):
        return load_csv_samples(data_dir, n=n, dataset_root=dataset_root)

    def build_model_input(self, sample, *, modality=None):
        return build_vision_input(sample, modality=modality)

    def parse_model_output(self, output):
        low = output.text.strip().lower()
        for t in VALID_COMP_TYPES:
            if t in low:
                return t
        return low

    def evaluate(self, predictions, ground_truth):
        n = max(len(predictions), 1)
        correct = sum(1 for p, g in zip(predictions, ground_truth) if str(p) == str(g))
        result: Dict[str, float] = {"accuracy": correct / n}
        per_f1: List[float] = []
        for label in VALID_COMP_TYPES:
            tp = sum(1 for p, g in zip(predictions, ground_truth) if p == label and g == label)
            fp = sum(1 for p, g in zip(predictions, ground_truth) if p == label and g != label)
            fn = sum(1 for p, g in zip(predictions, ground_truth) if p != label and g == label)
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            result[f"f1_{label}"] = f1
            per_f1.append(f1)
        result["f1_macro"] = sum(per_f1) / max(len(per_f1), 1)
        return result


@benchmark
class ComponentDetection(BaseBenchmark):
    """layout-7 — Detect all components with bounding boxes and type labels."""

    pipeline_implemented = True

    meta = BenchmarkMeta(
        id="layout-7",
        name="Component Detection",
        task_type=TaskType.UNDERSTANDING,
        domain="layout",
        data_subpath="layout/ComponentDetection",
        description="Detect all components with bounding boxes and type labels",
        input_spec="Rendered layout image",
        output_spec="JSON list of detections (bbox, label, score)",
        metrics=["mAP@0.5", "mAP@0.5:0.95"],
        tags=["P2", "high-complexity"],
    )

    def load_data(self, data_dir, *, n=None, dataset_root: Union[str, Path]):
        return load_csv_samples(data_dir, n=n, dataset_root=dataset_root)

    def build_model_input(self, sample, *, modality=None):
        return build_vision_input(sample, modality=modality)

    def parse_model_output(self, output):
        parsed = extract_json_obj(output.text.strip())
        if parsed is None:
            return []
        if not isinstance(parsed, list):
            parsed = (
                parsed.get("detections")
                or parsed.get("components")
                or parsed.get("layout_config", {}).get("components")
                or []
            )
        detections: list = []
        for idx, item in enumerate(parsed):
            if not isinstance(item, dict):
                continue
            bbox = item.get("bbox")
            if not isinstance(bbox, list) or len(bbox) < 4:
                continue
            try:
                bbox = [float(v) for v in bbox[:4]]
            except (TypeError, ValueError):
                continue
            label = str(item.get("label", item.get("type", "unknown"))).lower()
            score = float(item.get("score", item.get("confidence", 1.0 - idx * 0.01)))
            detections.append({"bbox": bbox, "label": label, "score": score})
        return detections

    def evaluate(self, predictions, ground_truth):
        pred_lists = [p if isinstance(p, list) else [] for p in predictions]
        gt_lists = [g if isinstance(g, list) else [] for g in ground_truth]
        return _detection_map(pred_lists, gt_lists)


class _LayerInsertionImageUtils:
    @staticmethod
    def _to_rgb_array(value: Any) -> Optional[np.ndarray]:
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            return value
        try:
            from PIL import Image

            if isinstance(value, (str, Path)):
                img = Image.open(str(value)).convert("RGB")
                return np.array(img)
            if hasattr(value, "convert"):
                return np.array(value.convert("RGB"))
        except Exception:
            pass
        return None

    @staticmethod
    def _resize_to_match(image: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
        h, w = target_hw
        if image.shape[0] == h and image.shape[1] == w:
            return image
        try:
            from PIL import Image

            pil = Image.fromarray(image).resize((w, h), Image.LANCZOS)
            return np.array(pil)
        except Exception:
            return image

    @staticmethod
    def _inception_feature(image: np.ndarray) -> Optional[np.ndarray]:
        return None

    @staticmethod
    def _to_gray_mask(value: Any, target_hw: Tuple[int, int]) -> Optional[np.ndarray]:
        arr = _LayerInsertionImageUtils._to_rgb_array(value)
        if arr is None:
            return None
        gray = np.mean(arr, axis=2).astype(np.uint8)
        if gray.shape[:2] != target_hw:
            gray_arr = _LayerInsertionImageUtils._resize_to_match(
                np.stack([gray] * 3, axis=2), target_hw
            )
            gray = gray_arr[:, :, 0]
        return gray

    @staticmethod
    def _read_image_size(path: Any) -> Tuple[int, int]:
        try:
            from PIL import Image

            img = Image.open(str(path))
            return img.size
        except Exception:
            return (0, 0)


@benchmark
class LayerAwareObjectInsertion(BaseBenchmark):
    """layout-8 — Layer-aware object insertion (G15). Manifest ``mode``: reference | description."""

    pipeline_implemented = True

    meta = BenchmarkMeta(
        id="layout-8",
        name="Layer-Aware Object Insertion & Asset Synthesis",
        task_type=TaskType.GENERATION,
        domain="layout",
        data_subpath="image/image-9-10-Layer-Aware Inpainting",
        description=(
            "Insert an object into a masked layout region using either a reference "
            "asset image or a textual description (per-sample mode)"
        ),
        input_spec="Masked layout + insertion mask + reference asset or description (+ optional prompt/context)",
        output_spec="Composited layout with identity-preserving object insertion",
        metrics=[
            "clip_identity",
            "dino_identity",
            "dreamsim_distance",
            "fid",
            "lpips",
            "imagereward",
            "hpsv3",
        ],
    )

    DEFAULT_PROMPT_REFERENCE = (
        "Insert the reference asset into the masked region and blend it naturally "
        "with the surrounding layout."
    )
    DEFAULT_PROMPT_DESCRIPTION = (
        "Insert an object matching the provided description into the masked region "
        "and blend it naturally with the surrounding layout."
    )
    VALID_MODES = ("reference", "description")

    _clip_img_bundle: Any = None
    _dino_bundle: Any = None
    _dreamsim_bundle: Any = None
    _lpips_bundle: Any = None
    _sample_component_pattern = re.compile(r"^(?P<layout_id>.+)_component_(?P<index>\d+)$")
    MANIFEST_JSON_FILENAMES = (
        "g15_object_insertion_manifest.json",
        "layer_aware_insertion_manifest.json",
        "object_insertion_manifest.json",
        "manifest.json",
    )
    MANIFEST_CSV_FILENAMES = (
        "g15_object_insertion_manifest.csv",
        "layer_aware_insertion_manifest.csv",
        "object_insertion_manifest.csv",
        "manifest.csv",
    )

    def _resolve(self, base_dir: Path, value: str) -> str:
        p = Path(value)
        if p.is_absolute():
            return str(p)
        return str((base_dir / p).resolve())

    def _should_include_reference_asset(self, sample: Dict[str, Any]) -> bool:
        return sample.get("mode", "reference") == "reference" and bool(sample.get("reference_asset"))

    def _should_include_asset_description(self, sample: Dict[str, Any]) -> bool:
        return sample.get("mode", "reference") == "description"

    @staticmethod
    def _normalize_reference_alt(raw: Any, *, max_chars: int = 500) -> str:
        text = str(raw or "").strip()
        if not text:
            return ""
        compact = " ".join(text.split())
        if len(compact) <= max_chars:
            return compact
        return compact[: max_chars - 3].rstrip() + "..."

    @staticmethod
    def _normalize_context(raw: Any, *, max_chars: int = 1400) -> str:
        text = str(raw or "").strip()
        if not text:
            return ""
        compact = " ".join(text.split())
        if len(compact) <= max_chars:
            return compact
        return compact[: max_chars - 3].rstrip() + "..."

    def _resolve_reference_asset_alt(
        self,
        *,
        base_dir: Path,
        row: Dict[str, Any],
        sample_id: str,
    ) -> str:
        direct = self._normalize_reference_alt(
            row.get("reference_asset_alt") or row.get("asset_alt") or row.get("alt")
        )
        if direct:
            return direct

        parsed = self._parse_sample_component(row=row, sample_id=sample_id)
        if parsed is None:
            return ""

        layout_id, component_index = parsed
        return self._lookup_alt_from_layout(
            base_dir=base_dir,
            layout_id=layout_id,
            component_index=component_index,
        )

    @classmethod
    def _parse_sample_component(
        cls,
        *,
        row: Dict[str, Any],
        sample_id: str,
    ) -> Optional[Tuple[str, int]]:
        layout_id = str(row.get("layout_id") or "").strip()
        index_raw = row.get("removed_component_index")
        if layout_id and index_raw is not None:
            try:
                index_val = int(index_raw)
            except Exception:
                index_val = -1
            if index_val >= 0:
                return layout_id, index_val

        match = cls._sample_component_pattern.match(sample_id)
        if not match:
            return None
        return match.group("layout_id"), int(match.group("index"))

    def _lookup_alt_from_layout(
        self,
        *,
        base_dir: Path,
        layout_id: str,
        component_index: int,
    ) -> str:
        if component_index < 0:
            return ""

        layout_path = base_dir / "layouts" / f"{layout_id}.json"
        if not layout_path.exists():
            return ""
        try:
            with open(layout_path, "r", encoding="utf-8") as f:
                layout_row = json.load(f)
        except Exception:
            return ""

        components = (layout_row.get("layout_config") or {}).get("components") or []
        if (
            not isinstance(components, list)
            or component_index >= len(components)
            or component_index < 0
        ):
            return ""
        cfg = components[component_index]
        if not isinstance(cfg, dict):
            return ""

        direct = self._normalize_reference_alt(cfg.get("alt"))
        if direct:
            return direct

        elem = cfg.get("element")
        if isinstance(elem, dict):
            nested = self._normalize_reference_alt(
                elem.get("alt") or elem.get("description")
            )
            if nested:
                return nested

        return ""

    def load_data(
        self,
        data_dir: Union[str, Path],
        *,
        n: Optional[int] = None,
        dataset_root: Union[str, Path],
    ) -> List[Dict[str, Any]]:
        path = Path(data_dir).resolve()
        if path.is_dir():
            candidates = self.MANIFEST_JSON_FILENAMES + self.MANIFEST_CSV_FILENAMES
            matched = [path / name for name in candidates if (path / name).is_file()]
            if not matched:
                raise FileNotFoundError(
                    f"No G15 insertion manifest found under {path}. "
                    f"Tried: {', '.join(candidates)}"
                )
            path = matched[0]

        if not path.exists():
            raise FileNotFoundError(f"G15 insertion manifest not found: {path}")

        rows = self._read_insertion_manifest_rows(path)

        base_dir = path.parent
        samples: List[Dict[str, Any]] = []
        for i, row in enumerate(rows):
            masked_layout = (
                row.get("masked_layout")
                or row.get("input_image")
                or row.get("masked_image")
                or row.get("layout_masked_image")
                or row.get("image")
            )
            mask = row.get("mask") or row.get("insert_mask") or row.get("component_mask")
            reference_asset = (
                row.get("reference_asset")
                or row.get("reference_image")
                or row.get("asset_image")
                or row.get("target_asset")
            )
            gt_image = (
                row.get("ground_truth_image")
                or row.get("target_image")
                or row.get("ground_truth")
            )

            if not masked_layout or not mask or not reference_asset or not gt_image:
                logger.warning("Incomplete G15 sample at index %d, skipping", i)
                continue

            mode = str(row.get("mode") or "reference").strip().lower()
            if mode not in self.VALID_MODES:
                logger.warning(
                    "Unknown mode %r for sample index %d, defaulting to 'reference'",
                    mode, i,
                )
                mode = "reference"

            sid = str(row.get("sample_id") or f"g15_insert_{i:04d}")
            reference_asset_alt = self._resolve_reference_asset_alt(
                base_dir=base_dir,
                row=row,
                sample_id=sid,
            )

            default_prompt = (
                self.DEFAULT_PROMPT_DESCRIPTION if mode == "description"
                else self.DEFAULT_PROMPT_REFERENCE
            )
            prompt = str(row.get("prompt") or row.get("instruction") or default_prompt)
            context = row.get("contextual_cues") or row.get("context") or row.get("surrounding_layers")
            if isinstance(context, (dict, list)):
                context = json.dumps(context, ensure_ascii=False)
            context = str(context or "").strip()

            samples.append(
                {
                    "sample_id": sid,
                    "mode": mode,
                    "input_image": self._resolve(base_dir, str(masked_layout)),
                    "mask": self._resolve(base_dir, str(mask)),
                    "reference_asset": self._resolve(base_dir, str(reference_asset)),
                    "reference_asset_alt": reference_asset_alt,
                    "prompt": prompt,
                    "context": context,
                    "ground_truth": {
                        "image": self._resolve(base_dir, str(gt_image)),
                        "mask": self._resolve(base_dir, str(mask)),
                        "reference_asset": self._resolve(base_dir, str(reference_asset)),
                        "reference_asset_alt": reference_asset_alt,
                        "prompt": prompt,
                    },
                }
            )

        if n is not None:
            samples = samples[:n]
        return samples

    @classmethod
    def _read_insertion_manifest_rows(cls, path: Path) -> List[Dict[str, Any]]:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            try:
                with open(path, "r", encoding="utf-8-sig", newline="") as f:
                    reader = csv.DictReader(f)
                    if not reader.fieldnames:
                        raise ValueError(f"CSV manifest has no header row: {path}")
                    return [
                        cls._normalize_insertion_manifest_csv_row(row)
                        for row in reader
                        if isinstance(row, dict)
                    ]
            except Exception as exc:
                raise ValueError(f"Failed to parse CSV manifest {path}: {exc}") from exc

        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        rows = payload.get("samples") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError("Manifest must be a list or {'samples': [...]} format.")
        return [row for row in rows if isinstance(row, dict)]

    @classmethod
    def _normalize_insertion_manifest_csv_row(cls, row: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(row)
        for key in (
            "sample_id",
            "masked_layout",
            "input_image",
            "masked_image",
            "layout_masked_image",
            "image",
            "mask",
            "insert_mask",
            "component_mask",
            "reference_asset",
            "reference_image",
            "asset_image",
            "target_asset",
            "ground_truth_image",
            "target_image",
            "ground_truth",
            "prompt",
            "instruction",
            "contextual_cues",
            "context",
            "surrounding_layers",
            "reference_asset_alt",
            "asset_alt",
            "alt",
            "mode",
        ):
            value = row.get(key)
            if isinstance(value, str):
                out[key] = value.replace("\\r\\n", "\n").replace("\\n", "\n").strip()

        out["removed_component_index"] = cls._safe_int(row.get("removed_component_index"), -1)
        out["mask_area_ratio"] = cls._safe_float(row.get("mask_area_ratio"), float("nan"))
        return out

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(float(value))
        except Exception:
            return default

    @staticmethod
    def _safe_float(value: Any, default: float = float("nan")) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def build_model_input(self, sample: Dict[str, Any], *, modality: Any = None) -> Any:
        from design_benchmarks.models.base import ModelInput

        prompt = self._compose_prompt(sample)
        images = [sample["input_image"]]
        if self._should_include_reference_asset(sample) and sample.get("reference_asset"):
            images.append(sample["reference_asset"])

        metadata: Dict[str, Any] = {
            "mask": sample["mask"],
            "reference_asset": sample.get("reference_asset", ""),
            "reference_asset_alt": sample.get("reference_asset_alt", ""),
            "task": "g15_layer_aware_object_insertion",
            "benchmark_id": self.meta.id,
            "sample_id": str(sample.get("sample_id") or ""),
        }
        width, height = _LayerInsertionImageUtils._read_image_size(sample.get("input_image"))
        if width > 0 and height > 0:
            metadata["target_width"] = width
            metadata["target_height"] = height

        return ModelInput(
            text=prompt,
            images=images,
            metadata=metadata,
        )

    def _compose_prompt(self, sample: Dict[str, Any]) -> str:
        default_prompt = (
            self.DEFAULT_PROMPT_DESCRIPTION
            if sample.get("mode", "reference") == "description"
            else self.DEFAULT_PROMPT_REFERENCE
        )
        user_intent = str(sample.get("prompt") or default_prompt).strip()
        context = self._normalize_context(sample.get("context", ""))
        alt = self._normalize_reference_alt(sample.get("reference_asset_alt", ""))
        has_reference = self._should_include_reference_asset(sample)

        lines = [
            "You are an expert graphic design retoucher specialized in layer-aware object insertion.",
            "Task: insert exactly one target object into the editable masked region while preserving the rest of the layout.",
            "",
            "Objective:",
            f"- User intent: {user_intent}",
            "- Return one final composited image only (no text explanation).",
            "",
            "Input semantics:",
            "- Image #1 is the layout canvas with the target region removed/masked.",
            "- The mask defines editable pixels only (white=editable, black=preserve).",
        ]
        if has_reference:
            lines.extend(
                [
                    "- A reference asset image is provided as an additional input image.",
                    "- Preserve the reference asset's visual identity while matching local scene style.",
                ]
            )
        else:
            lines.extend(
                [
                    "- No reference asset image is provided.",
                    "- Reconstruct the target object from textual description and context.",
                ]
            )

        if self._should_include_asset_description(sample):
            if alt:
                lines.append(f"- Target object description: {alt}")
            else:
                lines.append("- Target object description: unavailable; infer from intent/context.")

        if context:
            lines.extend(["", "Contextual cues:", f"- {context}"])

        identity_requirement = (
            "- Identity: keep key shape/material/details consistent with the reference asset."
            if has_reference
            else "- Identity: generate an object that closely matches the target description."
        )

        lines.extend(
            [
                "",
                "Hard constraints (must satisfy all):",
                "- Edit only masked pixels; keep unmasked regions unchanged.",
                "- Keep the inserted object fully inside the editable mask.",
                "- Do not erase, warp, or occlude nearby text/logo/important elements.",
                "- Match perspective, lighting, shadow, and color grading to neighbors.",
                "- Insert exactly one coherent object (no duplicates/fragments).",
                "",
                "Quality checklist:",
                identity_requirement,
                "- Boundary blending: edges should look natural without obvious cutout artifacts.",
                "- Semantic fit: the inserted object should support the user intent and design context.",
                "",
                "Output: a single composited image.",
            ]
        )
        return "\n".join(lines)

    def parse_model_output(self, output: Any) -> Any:
        if output is None:
            return None

        images = getattr(output, "images", None)
        if images:
            return images[0]

        text = getattr(output, "text", "")
        if isinstance(text, str):
            cleaned = text.strip()
            cleaned = re.sub(r"^```(?:txt|text)?\s*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```\s*$", "", cleaned).strip()
            if cleaned.startswith(("http://", "https://")):
                return cleaned
            as_path = Path(cleaned)
            if as_path.exists():
                return str(as_path)
        return None

    def evaluate(self, predictions: List[Any], ground_truth: List[Any]) -> Dict[str, float]:
        clip_scores: List[float] = []
        dino_scores: List[float] = []
        dreamsim_scores: List[float] = []
        lpips_scores: List[float] = []
        imagereward_scores: List[float] = []
        hps_scores: List[float] = []
        fid_real_features: List[np.ndarray] = []
        fid_gen_features: List[np.ndarray] = []

        evaluated = 0
        identity_pairs = 0

        for pred_raw, gt_raw in zip(predictions, ground_truth):
            gt = self._normalize_gt_bundle(gt_raw)
            pred_img = _LayerInsertionImageUtils._to_rgb_array(self._extract_image_like(pred_raw))
            gt_img = _LayerInsertionImageUtils._to_rgb_array(self._extract_image_like(gt["image"]))
            ref_img = _LayerInsertionImageUtils._to_rgb_array(self._extract_image_like(gt["reference_asset"]))

            if pred_img is None or gt_img is None:
                continue

            pred_img = _LayerInsertionImageUtils._resize_to_match(pred_img, gt_img.shape[:2])
            evaluated += 1

            real_feat = _LayerInsertionImageUtils._inception_feature(gt_img)
            gen_feat = _LayerInsertionImageUtils._inception_feature(pred_img)
            if real_feat is not None and gen_feat is not None:
                fid_real_features.append(real_feat)
                fid_gen_features.append(gen_feat)

            lpips = self._lpips_distance(pred_img, gt_img)
            self._append_if_finite(lpips_scores, lpips)

            prompt = str(gt.get("prompt", ""))
            clip_fallback = IntentToLayoutGeneration._clip_score(prompt, pred_img)
            imagereward = IntentToLayoutGeneration._imagereward_score(prompt, pred_img)
            hps = IntentToLayoutGeneration._hpsv3_score(prompt, pred_img, clip_fallback=clip_fallback)
            self._append_if_finite(imagereward_scores, imagereward)
            self._append_if_finite(hps_scores, hps)

            if ref_img is None:
                continue

            mask = None
            if gt.get("mask"):
                mask = _LayerInsertionImageUtils._to_gray_mask(gt["mask"], gt_img.shape[:2])

            pred_obj = self._extract_object_region(pred_img, mask)
            ref_obj = self._extract_object_region(ref_img, None)

            clip_identity = self._clip_image_similarity(pred_obj, ref_obj)
            dino_identity = self._dino_similarity(pred_obj, ref_obj)
            dreamsim_distance = self._dreamsim_distance(pred_obj, ref_obj)

            self._append_if_finite(clip_scores, clip_identity)
            self._append_if_finite(dino_scores, dino_identity)
            self._append_if_finite(dreamsim_scores, dreamsim_distance)
            identity_pairs += 1

        fid_score = float("nan")
        if len(fid_real_features) >= 2 and len(fid_gen_features) >= 2:
            try:
                fid_score = float(fid_metric(np.stack(fid_real_features), np.stack(fid_gen_features)))
                if math.isfinite(fid_score):
                    fid_score = max(0.0, fid_score)
            except Exception:
                fid_score = float("nan")

        denom = max(evaluated, 1)
        return {
            "clip_identity": self._mean_or_nan(clip_scores),
            "dino_identity": self._mean_or_nan(dino_scores),
            "dreamsim_distance": self._mean_or_nan(dreamsim_scores),
            "fid": fid_score,
            "lpips": self._mean_or_nan(lpips_scores),
            "imagereward": self._mean_or_nan(imagereward_scores),
            "hpsv3": self._mean_or_nan(hps_scores),
            "evaluated_samples": float(evaluated),
            "identity_pair_count": float(identity_pairs),
            "identity_coverage": identity_pairs / denom,
        }

    @staticmethod
    def _normalize_gt_bundle(raw: Any) -> Dict[str, Any]:
        if isinstance(raw, dict):
            return {
                "image": raw.get("image", raw.get("ground_truth_image", raw)),
                "reference_asset": (
                    raw.get("reference_asset")
                    or raw.get("reference_image")
                    or raw.get("asset_image")
                ),
                "mask": raw.get("mask") or raw.get("insert_mask") or raw.get("component_mask"),
                "prompt": str(raw.get("prompt", "")),
            }
        return {"image": raw, "reference_asset": None, "mask": None, "prompt": ""}

    @staticmethod
    def _extract_image_like(value: Any) -> Any:
        if isinstance(value, dict):
            for key in ("image", "output_image", "predicted_image", "path"):
                if key in value:
                    return value[key]

        images = getattr(value, "images", None)
        if images:
            return images[0]
        return value

    @staticmethod
    def _extract_object_region(image: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
        if mask is None:
            return image

        ys, xs = np.where(mask > 127)
        if ys.size == 0 or xs.size == 0:
            return image

        y1, y2 = int(ys.min()), int(ys.max()) + 1
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        region = image[y1:y2, x1:x2].copy()
        local_mask = (mask[y1:y2, x1:x2] > 127)

        isolated = np.full_like(region, 255)
        isolated[local_mask] = region[local_mask]
        return isolated

    @classmethod
    def _clip_image_similarity(cls, img_a: np.ndarray, img_b: np.ndarray) -> float:
        if cls._clip_img_bundle is None:
            try:
                import torch
                from transformers import CLIPModel, CLIPProcessor

                device = "cuda" if torch.cuda.is_available() else "cpu"
                processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
                cls._clip_img_bundle = (model, processor, torch, device)
            except Exception as exc:
                logger.info("CLIP image-image metric unavailable: %s", exc)
                cls._clip_img_bundle = False

        if not cls._clip_img_bundle:
            return float("nan")

        model, processor, torch, device = cls._clip_img_bundle
        try:
            from PIL import Image

            inputs = processor(images=[Image.fromarray(img_a), Image.fromarray(img_b)], return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            with torch.no_grad():
                feats = model.get_image_features(pixel_values=pixel_values)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            return float((feats[0] @ feats[1]).item())
        except Exception:
            return float("nan")

    @classmethod
    def _dino_similarity(cls, img_a: np.ndarray, img_b: np.ndarray) -> float:
        if cls._dino_bundle is None:
            try:
                import torch
                from transformers import AutoImageProcessor, AutoModel

                device = "cuda" if torch.cuda.is_available() else "cpu"
                processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
                model = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()
                cls._dino_bundle = (model, processor, torch, device)
            except Exception as exc:
                logger.info("DINO metric unavailable: %s", exc)
                cls._dino_bundle = False

        if not cls._dino_bundle:
            return float("nan")

        model, processor, torch, device = cls._dino_bundle
        try:
            from PIL import Image

            inputs = processor(images=[Image.fromarray(img_a), Image.fromarray(img_b)], return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                feats = outputs.last_hidden_state[:, 0, :]
                feats = feats / feats.norm(dim=-1, keepdim=True)
            return float((feats[0] @ feats[1]).item())
        except Exception:
            return float("nan")

    @classmethod
    def _dreamsim_distance(cls, img_a: np.ndarray, img_b: np.ndarray) -> float:
        if cls._dreamsim_bundle is None:
            try:
                import torch
                from dreamsim import (
                    dreamsim as dreamsim_factory,  # type: ignore[reportMissingImports]
                )

                device = "cuda" if torch.cuda.is_available() else "cpu"
                model, preprocess = dreamsim_factory(pretrained=True, device=device)
                model.eval()
                cls._dreamsim_bundle = (model, preprocess, torch, device)
            except Exception as exc:
                logger.info("DreamSim unavailable, falling back to LPIPS proxy: %s", exc)
                cls._dreamsim_bundle = False

        if not cls._dreamsim_bundle:
            return cls._lpips_distance(img_a, img_b)

        model, preprocess, torch, device = cls._dreamsim_bundle
        try:
            from PIL import Image

            ta = preprocess(Image.fromarray(img_a))
            tb = preprocess(Image.fromarray(img_b))
            if hasattr(ta, "ndim") and ta.ndim == 3:
                ta = ta.unsqueeze(0)
            if hasattr(tb, "ndim") and tb.ndim == 3:
                tb = tb.unsqueeze(0)
            ta = ta.to(device)
            tb = tb.to(device)
            with torch.no_grad():
                score = model(ta, tb)
            if hasattr(score, "item"):
                score = score.item()
            return float(score)
        except Exception:
            return cls._lpips_distance(img_a, img_b)

    @classmethod
    def _lpips_distance(cls, img_a: np.ndarray, img_b: np.ndarray) -> float:
        if img_a.shape[:2] != img_b.shape[:2]:
            img_b = _LayerInsertionImageUtils._resize_to_match(img_b, img_a.shape[:2])

        if cls._lpips_bundle is None:
            try:
                import lpips
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = lpips.LPIPS(net="alex").to(device).eval()
                cls._lpips_bundle = (model, torch, device)
            except Exception as exc:
                logger.info("LPIPS unavailable, using MSE proxy: %s", exc)
                cls._lpips_bundle = False

        if not cls._lpips_bundle:
            mse = float(np.mean((img_a.astype(np.float32) - img_b.astype(np.float32)) ** 2))
            return mse / (255.0 ** 2)

        model, torch, device = cls._lpips_bundle
        try:
            ta = cls._to_lpips_tensor(img_a, torch).to(device)
            tb = cls._to_lpips_tensor(img_b, torch).to(device)
            with torch.no_grad():
                score = model(ta, tb)
            return float(score.item())
        except Exception:
            mse = float(np.mean((img_a.astype(np.float32) - img_b.astype(np.float32)) ** 2))
            return mse / (255.0 ** 2)

    @staticmethod
    def _to_lpips_tensor(image: np.ndarray, torch: Any) -> Any:
        x = image.astype(np.float32) / 127.5 - 1.0
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
        return x

    @staticmethod
    def _append_if_finite(bucket: List[float], value: float) -> None:
        if isinstance(value, float) and math.isfinite(value):
            bucket.append(value)

    @staticmethod
    def _mean_or_nan(values: List[float]) -> float:
        if not values:
            return float("nan")
        return float(sum(values) / len(values))


# ===========================================================================
