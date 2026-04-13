"""API-based model wrappers (OpenAI, Anthropic, Google, etc.)."""

from __future__ import annotations

import base64
import io
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .base import BaseModel, Modality, ModelInput, ModelOutput
from .registry import register_model

# ---------------------------------------------------------------------------
# OpenAI / GPT
# ---------------------------------------------------------------------------


@register_model("openai")
class OpenAIModel(BaseModel):
    """Template for OpenAI API models (GPT-4o, GPT-4-turbo, etc.)."""

    modality = Modality.TEXT_AND_IMAGE

    def __init__(
        self,
        model_id: str = "gpt-4o",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs: Any,
    ):
        self.model_id = model_id
        self.name = model_id
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.temperature = temperature
        self.max_tokens = max_tokens

    def predict(self, inp: ModelInput) -> ModelOutput:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                'openai is required for this model. Install with: pip install -e ".[openai]"'
            )

        client = OpenAI(api_key=self.api_key)

        content: list = []
        if inp.text:
            content.append({"type": "text", "text": inp.text})
        for img in inp.images:
            content.append({"type": "image_url", "image_url": {"url": _to_data_url(img)}})

        response = client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": content}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        choice = response.choices[0]
        return ModelOutput(
            text=choice.message.content or "",
            raw=response,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            },
        )


# ---------------------------------------------------------------------------
# OpenAI Images (generation/edit)
# ---------------------------------------------------------------------------


@register_model("openai_image")
class OpenAIImageModel(BaseModel):
    """OpenAI image model wrapper for generation/editing workflows."""

    modality = Modality.IMAGE_GENERATION

    def __init__(
        self,
        model_id: str = "gpt-image-1.5",
        api_key: Optional[str] = None,
        size: str = "1024x1024",
        adaptive_size: bool = True,
        **kwargs: Any,
    ):
        self.model_id = model_id
        self.name = model_id
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.size = size
        self.adaptive_size = adaptive_size

    def predict(self, inp: ModelInput) -> ModelOutput:
        """Run image generation or image edit based on ModelInput."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                'openai is required for this model. Install with: pip install -e ".[openai]"'
            )

        client = OpenAI(api_key=self.api_key)
        prompt = inp.text or str(inp.metadata.get("prompt", ""))
        request_size = self._resolve_request_size(inp.metadata)

        if inp.images:
            image_upload = _to_upload_file(inp.images[0], "input.png")
            mask_source = inp.metadata.get("mask")
            if mask_source is not None:
                mask_upload = _to_upload_file(mask_source, "mask.png", as_mask=True)
                response = client.images.edit(
                    model=self.model_id,
                    image=image_upload,
                    mask=mask_upload,
                    prompt=prompt,
                    size=request_size,
                )
            else:
                # If no mask is provided, fall back to plain image generation.
                response = client.images.generate(
                    model=self.model_id,
                    prompt=prompt,
                    size=request_size,
                )
        else:
            response = client.images.generate(
                model=self.model_id,
                prompt=prompt,
                size=request_size,
            )

        return ModelOutput(
            images=_decode_openai_images_response(response),
            raw=response,
        )

    def _resolve_request_size(self, metadata: Dict[str, Any]) -> str:
        """Select image size from target metadata when available.

        OpenAI image models accept a small set of canonical sizes. We map
        layout target width/height to the closest supported aspect ratio to
        avoid always forcing square outputs for non-square layouts.
        """
        if not self.adaptive_size:
            return self.size

        if not isinstance(metadata, dict):
            return self.size

        # Keep behavior stable for non-layout benchmarks.
        if str(metadata.get("benchmark_id", "")) != "layout-1":
            return self.size

        width = self._safe_int(
            metadata.get("target_width", metadata.get("width")),
        )
        height = self._safe_int(
            metadata.get("target_height", metadata.get("height")),
        )
        if width <= 0 or height <= 0:
            return self.size

        # If caller explicitly requested a non-canonical size, respect it.
        canonical = {"1024x1024", "1536x1024", "1024x1536", "auto"}
        if self.size not in canonical:
            return self.size

        target_ratio = float(width) / float(height)
        candidates = [
            ("1024x1024", 1.0),
            ("1536x1024", 1.5),
            ("1024x1536", 1024.0 / 1536.0),
        ]
        best_size, _ = min(
            candidates,
            key=lambda item: abs(math.log(target_ratio) - math.log(item[1])),
        )
        return best_size

    @staticmethod
    def _safe_int(value: Any) -> int:
        try:
            return int(float(value))
        except Exception:
            return 0


# ---------------------------------------------------------------------------
# Anthropic / Claude
# ---------------------------------------------------------------------------


@register_model("anthropic")
class AnthropicModel(BaseModel):
    """Template for Anthropic API models (Claude Opus, Sonnet, Haiku)."""

    modality = Modality.TEXT_AND_IMAGE

    def __init__(
        self,
        model_id: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs: Any,
    ):
        self.model_id = model_id
        self.name = model_id
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.temperature = temperature
        self.max_tokens = max_tokens

    def predict(self, inp: ModelInput) -> ModelOutput:
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                'anthropic is required for this model. Install with: pip install -e ".[anthropic]"'
            )

        client = anthropic.Anthropic(api_key=self.api_key)

        content: list = []
        for img in inp.images:
            data, media_type = _to_base64(img)
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": data},
            })
        if inp.text:
            content.append({"type": "text", "text": inp.text})

        response = client.messages.create(
            model=self.model_id,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": content}],
        )
        text = "".join(
            block.text for block in response.content if hasattr(block, "text")
        )
        return ModelOutput(
            text=text,
            raw=response,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )


# ---------------------------------------------------------------------------
# Google / Gemini
# ---------------------------------------------------------------------------


@register_model("google")
class GeminiModel(BaseModel):
    """Google Gemini model wrapper via the ``google-genai`` SDK.

    Supports two credential modes:
    - Simple JSON with an ``api_key`` field (Google AI Studio key).
    - Google Cloud service-account JSON (uses Vertex AI endpoint).
    """

    modality = Modality.ANY

    MODEL_ALIASES = {
        "nanobanana-pro": "gemini-3.1-flash-image-preview",
        "nano-banana-pro": "gemini-3.1-flash-image-preview",
        "nanobanana": "gemini-3.1-flash-image-preview",
        "nano-banana": "gemini-3.1-flash-image-preview",
    }
    SUPPORTED_ASPECT_RATIOS = (
        "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4",
        "9:16", "16:9", "21:9", "1:8", "8:1", "1:4", "4:1",
    )
    SUPPORTED_IMAGE_SIZES = ("512", "1K", "2K", "4K")

    def __init__(
        self,
        model_id: str = "gemini-3.1-flash-lite-preview",
        api_key: Optional[str] = None,
        credentials_path: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        adaptive_image_config: bool = True,
        image_size: Optional[str] = "1K",
        retry_attempts: int = 6,
        retry_backoff_seconds: float = 5.0,
        **kwargs: Any,
    ):
        self.model_id = self._resolve_model_id(model_id)
        self.name = self.model_id
        self.modality = self._infer_modality(self.model_id)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.adaptive_image_config = adaptive_image_config
        size_token = str(image_size).upper() if image_size is not None else "1K"
        if size_token == "AUTO":
            # Explicit opt-in for automatic tiering (can choose 2K/4K).
            self.image_size = None
        elif size_token in self.SUPPORTED_IMAGE_SIZES:
            self.image_size = size_token
        else:
            self.image_size = "1K"
        self.retry_attempts = max(1, int(retry_attempts))
        self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))
        self._credentials_path = credentials_path

        if api_key:
            self._auth_mode = "api_key"
            self.api_key = api_key
        elif credentials_path:
            creds = json.loads(Path(credentials_path).read_text())
            if "api_key" in creds:
                self._auth_mode = "api_key"
                self.api_key = creds["api_key"]
            elif creds.get("type") == "service_account":
                self._auth_mode = "service_account"
                self._project_id = creds.get("project_id", "")
                self.api_key = None
            else:
                raise ValueError(
                    "Credentials JSON must contain either an 'api_key' field "
                    "or be a service-account key (type=service_account)."
                )
        else:
            self._auth_mode = "api_key"
            self.api_key = os.environ.get("GOOGLE_API_KEY", "")

    def _build_client(self) -> Any:
        try:
            from google import genai
        except ImportError:
            raise ImportError(
                'google-genai is required for Gemini models. Install with: pip install -e ".[gemini]"'
            )

        if self._auth_mode == "service_account":
            from google.oauth2 import service_account

            scopes = ["https://www.googleapis.com/auth/cloud-platform"]
            sa_creds = service_account.Credentials.from_service_account_file(
                self._credentials_path, scopes=scopes,
            )
            return genai.Client(
                vertexai=True,
                project=self._project_id,
                location="global",
                credentials=sa_creds,
            )
        return genai.Client(api_key=self.api_key)

    def predict(self, inp: ModelInput) -> ModelOutput:
        try:
            from google.genai import types
        except ImportError:
            raise ImportError(
                'google-genai is required for Gemini models. Install with: pip install -e ".[gemini]"'
            )

        client = self._build_client()

        contents: list[Any] = []
        for img in inp.images:
            contents.append(self._load_image(img))
        if inp.text:
            contents.append(inp.text)

        config = self._build_generate_content_config(inp, types)
        response = self._generate_content_with_retry(client, contents, config)

        text = self._extract_text(response)
        images = self._extract_images(response)
        usage: dict[str, Any] = {}
        if response.usage_metadata:
            usage = {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "completion_tokens": response.usage_metadata.candidates_token_count,
            }

        return ModelOutput(text=text, images=images, raw=response, usage=usage)

    def _build_generate_content_config(self, inp: ModelInput, types: Any) -> Any:
        kwargs: Dict[str, Any] = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
        }

        if self.modality == Modality.IMAGE_GENERATION:
            kwargs["response_modalities"] = ["IMAGE"]
            image_config = self._resolve_image_config(inp.metadata, types)
            if image_config is not None:
                kwargs["image_config"] = image_config

        return types.GenerateContentConfig(**kwargs)

    def _generate_content_with_retry(self, client: Any, contents: List[Any], config: Any) -> Any:
        last_error: Optional[Exception] = None
        for attempt in range(self.retry_attempts):
            try:
                return client.models.generate_content(
                    model=self.model_id,
                    contents=contents,
                    config=config,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if not self._is_retryable_error(exc) or attempt >= self.retry_attempts - 1:
                    raise
                sleep_s = self.retry_backoff_seconds * (2 ** attempt)
                time.sleep(sleep_s)
        if last_error is not None:
            raise last_error
        raise RuntimeError("Gemini request failed without exception context.")

    @staticmethod
    def _is_retryable_error(exc: Exception) -> bool:
        text = str(exc).upper()
        return any(
            token in text
            for token in (
                "RESOURCE_EXHAUSTED",
                "RATE_LIMIT",
                "429",
                "UNAVAILABLE",
                "503",
                "DEADLINE_EXCEEDED",
                "TIMEOUT",
            )
        )

    def _resolve_image_config(self, metadata: Dict[str, Any], types: Any) -> Optional[Any]:
        aspect_ratio: Optional[str] = None
        image_size: Optional[str] = self.image_size

        if (
            self.adaptive_image_config
            and isinstance(metadata, dict)
            and str(metadata.get("benchmark_id", "")) == "layout-1"
        ):
            width = self._safe_int(metadata.get("target_width", metadata.get("width")))
            height = self._safe_int(metadata.get("target_height", metadata.get("height")))
            if width > 0 and height > 0:
                aspect_ratio = self._closest_aspect_ratio(width, height)
                if image_size is None:
                    image_size = self._closest_image_size(width, height)

        if image_size and image_size not in self.SUPPORTED_IMAGE_SIZES:
            image_size = None
        if aspect_ratio and aspect_ratio not in self.SUPPORTED_ASPECT_RATIOS:
            aspect_ratio = None

        if aspect_ratio is None and image_size is None:
            return None
        return types.ImageConfig(aspect_ratio=aspect_ratio, image_size=image_size)

    def _closest_aspect_ratio(self, width: int, height: int) -> str:
        target = float(width) / float(height)
        candidates = []
        for spec in self.SUPPORTED_ASPECT_RATIOS:
            a, b = spec.split(":")
            ratio = float(a) / float(b)
            candidates.append((spec, ratio))
        best, _ = min(
            candidates,
            key=lambda item: abs(math.log(target) - math.log(item[1])),
        )
        return best

    def _closest_image_size(self, width: int, height: int) -> str:
        long_side = max(width, height)
        candidates = {
            "512": 512,
            "1K": 1024,
            "2K": 2048,
            "4K": 4096,
        }
        best, _ = min(
            candidates.items(),
            key=lambda item: abs(math.log(long_side) - math.log(item[1])),
        )
        return best

    @staticmethod
    def _safe_int(value: Any) -> int:
        try:
            return int(float(value))
        except Exception:
            return 0

    @classmethod
    def _resolve_model_id(cls, model_id: str) -> str:
        return cls.MODEL_ALIASES.get(model_id.strip().lower(), model_id)

    @staticmethod
    def _infer_modality(model_id: str) -> Modality:
        lowered = model_id.lower()
        if any(token in lowered for token in ("flash-image", "imagen", "image-preview", "nano-banana", "nanobanana")):
            return Modality.IMAGE_GENERATION
        return Modality.TEXT_AND_IMAGE

    @staticmethod
    def _extract_images(response: Any) -> List[Any]:
        out: List[Any] = []
        try:
            from PIL import Image
        except ImportError:
            Image = None  # type: ignore[assignment]

        for candidate in getattr(response, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) or []
            for part in parts:
                inline = getattr(part, "inline_data", None) or getattr(part, "inlineData", None)
                if inline is not None:
                    data = getattr(inline, "data", None)
                    if data is None:
                        continue

                    if isinstance(data, str):
                        try:
                            img_bytes = base64.b64decode(data)
                        except Exception:
                            img_bytes = data.encode("utf-8")
                    else:
                        img_bytes = bytes(data)

                    if Image is None:
                        out.append(img_bytes)
                    else:
                        try:
                            out.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
                        except Exception:
                            out.append(img_bytes)
                    continue

                file_data = getattr(part, "file_data", None) or getattr(part, "fileData", None)
                if file_data is not None:
                    uri = getattr(file_data, "file_uri", None) or getattr(file_data, "fileUri", None)
                    if uri:
                        out.append(uri)
        return out

    @staticmethod
    def _extract_text(response: Any) -> str:
        chunks: List[str] = []
        for candidate in getattr(response, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) or []
            for part in parts:
                text = getattr(part, "text", None)
                if isinstance(text, str) and text:
                    chunks.append(text)
        return "\n".join(chunks).strip()

    @staticmethod
    def _load_image(image: Union[str, Path, bytes]) -> Any:
        """Convert an image source to a PIL Image for the Gemini SDK."""
        try:
            from PIL import Image
        except ImportError:
            raise ImportError(
                'Pillow is required for Gemini image inputs. Install with: pip install Pillow'
            )

        if isinstance(image, bytes):
            return Image.open(io.BytesIO(image))
        path = Path(image) if isinstance(image, str) else image
        if path.exists():
            return Image.open(path)
        raise FileNotFoundError(f"Image not found: {image}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_base64(image: Union[str, Path, bytes]) -> tuple[str, str]:
    """Convert an image source to (base64_data, media_type)."""
    if isinstance(image, bytes):
        return base64.b64encode(image).decode(), "image/png"

    path = Path(image) if isinstance(image, str) and not image.startswith("http") else None
    if path and path.exists():
        suffix = path.suffix.lower()
        media_type = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }.get(suffix, "image/png")
        return base64.b64encode(path.read_bytes()).decode(), media_type

    # URL — caller should handle URL-based images per provider API
    return str(image), "image/png"


def _to_data_url(image: Union[str, Path, bytes]) -> str:
    """Convert an image source to a data URL for OpenAI's API."""
    if isinstance(image, str) and image.startswith("http"):
        return image

    data, media_type = _to_base64(image)
    return f"data:{media_type};base64,{data}"


def _to_upload_file(image: Any, filename: str, as_mask: bool = False) -> io.BytesIO:
    """Convert image-like input to a file-like object for OpenAI image APIs."""
    if as_mask:
        data = _to_openai_mask_png_bytes(image)
    else:
        data = _read_image_bytes(image)

    upload = io.BytesIO(data)
    upload.name = filename
    return upload


def _read_image_bytes(image: Any) -> bytes:
    """Read image-like input into bytes (preserves file bytes for path/string)."""
    if isinstance(image, (bytes, bytearray)):
        return bytes(image)
    if isinstance(image, Path):
        return image.read_bytes()
    if isinstance(image, str):
        p = Path(image)
        if not p.exists():
            raise FileNotFoundError(f"Image path not found: {image}")
        return p.read_bytes()

    try:
        from PIL import Image
    except ImportError as exc:
        raise ValueError(
            "Unsupported image type. Use bytes/path or install Pillow for PIL support."
        ) from exc

    if isinstance(image, Image.Image):
        buf = io.BytesIO()
        image.convert("RGBA").save(buf, format="PNG")
        return buf.getvalue()

    raise ValueError(f"Unsupported image input type: {type(image)}")


def _to_openai_mask_png_bytes(mask: Any) -> bytes:
    """Convert mask to PNG bytes suitable for OpenAI image edit.

    If the input already has transparency, keep it.
    Otherwise interpret bright pixels as editable regions and convert them to
    transparent alpha (OpenAI edit convention).
    """
    try:
        from PIL import Image
    except ImportError:
        return _read_image_bytes(mask)

    if isinstance(mask, Image.Image):
        pil = mask
    elif isinstance(mask, (bytes, bytearray)):
        pil = Image.open(io.BytesIO(mask))
    elif isinstance(mask, Path):
        pil = Image.open(mask)
    elif isinstance(mask, str):
        p = Path(mask)
        if not p.exists():
            raise FileNotFoundError(f"Mask path not found: {mask}")
        pil = Image.open(p)
    else:
        raise ValueError(f"Unsupported mask input type: {type(mask)}")

    rgba = pil.convert("RGBA")
    alpha = rgba.getchannel("A")
    lo, hi = alpha.getextrema()
    if lo < 255:
        out = rgba
    else:
        gray = pil.convert("L")
        alpha = gray.point(lambda p: 0 if p > 127 else 255, mode="L")
        out = Image.new("RGBA", pil.size, (0, 0, 0, 255))
        out.putalpha(alpha)

    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()


def _decode_openai_images_response(response: Any) -> List[Any]:
    """Decode OpenAI image response items to PIL Images or URLs."""
    out: List[Any] = []
    for item in getattr(response, "data", []) or []:
        b64_json = getattr(item, "b64_json", None)
        if b64_json:
            img_bytes = base64.b64decode(b64_json)
            try:
                from PIL import Image
            except ImportError:
                out.append(img_bytes)
                continue
            out.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
            continue

        url = getattr(item, "url", None)
        if url:
            out.append(url)
    return out
