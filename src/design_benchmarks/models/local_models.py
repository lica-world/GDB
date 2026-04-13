"""Local / open-source model templates (HuggingFace, vLLM, diffusion, custom).

These are starter templates for running open-source models on local GPUs.
Custom user-provided Python entrypoints also live here so local integrations
share the same extension surface.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .base import BaseModel, Modality, ModelInput, ModelOutput
from .registry import register_model

logger = logging.getLogger(__name__)

def _is_fatal_load_error(exc: BaseException) -> bool:
    """True for errors that should abort the model-loader fallback loop.

    We re-raise OOM and I/O errors immediately instead of masking them
    behind a generic "Failed to load" message.  Architecture mismatches
    (ValueError, KeyError, non-OOM RuntimeError) are expected when trying
    multiple AutoModel classes and should fall through.
    """
    if isinstance(exc, (MemoryError, OSError)):
        return True
    try:
        import torch
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    return False


_VLM_MODEL_ID_HINTS = (
    "qwen-vl",
    "qwen2-vl",
    "qwen2.5-vl",
    "qwen3-vl",
    "llava",
    "internvl",
    "pixtral",
    "molmo",
    "vision",
    "visual",
    "-vl-",
    "/vl-",
)


def _looks_like_vlm_model_id(model_id: str) -> bool:
    lowered = str(model_id or "").strip().lower()
    return any(token in lowered for token in _VLM_MODEL_ID_HINTS)


def _modality_from_value(value: Any, *, default: Modality = Modality.ANY) -> Modality:
    if isinstance(value, Modality):
        return value
    text = str(value or "").strip().lower()
    mapping = {
        "text": Modality.TEXT,
        "image": Modality.IMAGE,
        "both": Modality.TEXT_AND_IMAGE,
        "text_and_image": Modality.TEXT_AND_IMAGE,
        "image_generation": Modality.IMAGE_GENERATION,
        "generation": Modality.IMAGE_GENERATION,
        "any": Modality.ANY,
    }
    return mapping.get(text, default)


def _resolve_text_model_modality(
    model_id: str,
    modality: Optional[Union[str, Modality]],
) -> Modality:
    if modality is None:
        return Modality.TEXT_AND_IMAGE if _looks_like_vlm_model_id(model_id) else Modality.TEXT

    resolved = _modality_from_value(modality, default=Modality.TEXT)
    if resolved == Modality.TEXT:
        return Modality.TEXT
    if resolved in {Modality.IMAGE, Modality.TEXT_AND_IMAGE, Modality.ANY}:
        return Modality.TEXT_AND_IMAGE
    raise ValueError(
        "hf/vllm providers only support text or text-and-image modalities. "
        f"Got {modality!r} for model {model_id!r}."
    )


# ---------------------------------------------------------------------------
# HuggingFace Transformers (single GPU / multi-GPU)
# ---------------------------------------------------------------------------


@register_model("hf")
class HuggingFaceModel(BaseModel):
    """Template for HuggingFace Transformers models.

    Supports both text-only and vision-language models.  The model and
    processor are loaded lazily on first ``predict`` call so importing
    this module doesn't require a GPU.

    Example::

        model = load_model("hf", model_id="google/gemma-3-4b-it")
        model = load_model("hf", model_id="llava-hf/llava-v1.6-mistral-7b-hf",
                           modality=Modality.TEXT_AND_IMAGE)
    """

    def __init__(
        self,
        model_id: str = "google/gemma-3-4b-it",
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_new_tokens: int = 4096,
        modality: Optional[Union[str, Modality]] = None,
        **kwargs: Any,
    ):
        if kwargs:
            warnings.warn(
                f"HuggingFaceModel received unexpected kwargs: {list(kwargs)}",
                stacklevel=2,
            )
        self.model_id = model_id
        self.name = model_id.split("/")[-1]
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.modality = _resolve_text_model_modality(model_id, modality)
        self.supports_image_output = False
        self.supports_video_output = False
        self.supports_mask_editing = False
        self.supports_image_input = self.modality != Modality.TEXT
        self._model = None
        self._processor = None
        self._runtime_device: Any = None

    def _load(self) -> None:
        """Lazy-load model and tokenizer/processor."""
        try:
            import torch  # type: ignore[reportMissingImports]
            from transformers import (  # type: ignore[reportMissingImports]
                AutoModelForCausalLM,
                AutoModelForImageTextToText,
                AutoModelForVision2Seq,
                AutoProcessor,
                AutoTokenizer,
            )
        except ImportError:
            raise ImportError(
                "torch and transformers are required for HuggingFace models. "
                "Install with: pip install torch transformers"
            )

        torch_dtype = getattr(torch, self.dtype, torch.bfloat16)

        if self.modality == Modality.TEXT:
            self._processor = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                device_map=self.device,
            )
        else:
            # Vision-language models typically use AutoProcessor
            self._processor = AutoProcessor.from_pretrained(self.model_id)
            load_errors: List[str] = []
            for loader in (
                AutoModelForImageTextToText,
                AutoModelForVision2Seq,
                AutoModelForCausalLM,
            ):
                try:
                    self._model = loader.from_pretrained(
                        self.model_id,
                        torch_dtype=torch_dtype,
                        device_map=self.device,
                    )
                    break
                except Exception as exc:  # noqa: BLE001
                    if _is_fatal_load_error(exc):
                        raise
                    load_errors.append(f"{loader.__name__}: {type(exc).__name__}: {exc}")
            if self._model is None:
                joined = "\n".join(load_errors)
                raise RuntimeError(
                    f"Failed to load multimodal HuggingFace model {self.model_id!r}.\n{joined}"
                )
        self._runtime_device = self._resolve_runtime_device()

    def _resolve_runtime_device(self) -> Any:
        if self._model is None:
            return self.device
        if self.device != "auto":
            return self.device
        model_device = getattr(self._model, "device", None)
        if model_device is not None:
            return model_device
        try:
            first_param = next(self._model.parameters())
            return first_param.device
        except Exception:
            return "cuda"

    @staticmethod
    def _load_images(inp: ModelInput) -> List[Any]:
        from PIL import Image  # type: ignore[reportMissingImports]

        images: List[Any] = []
        for img in inp.images:
            if isinstance(img, Path):
                images.append(Image.open(img).convert("RGB"))
            elif isinstance(img, str):
                images.append(Image.open(img).convert("RGB"))
            else:
                images.append(img)
        return images

    def _prepare_text_only_inputs(self, inp: ModelInput) -> Any:
        return self._processor(inp.text, return_tensors="pt").to(self._runtime_device)

    def _prepare_multimodal_inputs(self, inp: ModelInput) -> Any:
        images = self._load_images(inp)
        apply_chat_template = getattr(self._processor, "apply_chat_template", None)
        if callable(apply_chat_template):
            content: List[Dict[str, Any]] = []
            can_use_chat_template = True
            for raw in inp.images:
                if isinstance(raw, Path):
                    image_ref: Any = str(raw.resolve())
                elif isinstance(raw, str):
                    image_ref = str(Path(raw).resolve()) if Path(raw).exists() else raw
                else:
                    can_use_chat_template = False
                    break
                content.append({"type": "image", "image": image_ref})
            content.append({"type": "text", "text": inp.text})
            if can_use_chat_template:
                try:
                    return apply_chat_template(
                        [{"role": "user", "content": content}],
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=True,
                        return_tensors="pt",
                    ).to(self._runtime_device)
                except Exception as exc:
                    logger.debug(
                        "apply_chat_template failed for %s, falling back to processor: %s",
                        self.model_id,
                        exc,
                    )
        return self._processor(
            text=inp.text,
            images=images or None,
            return_tensors="pt",
        ).to(self._runtime_device)

    def _decode_output_ids(self, inputs: Any, output_ids: Any) -> str:
        input_ids = inputs.get("input_ids") if hasattr(inputs, "get") else None
        batch_decode = getattr(self._processor, "batch_decode", None)
        if input_ids is not None:
            trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(input_ids, output_ids)
            ]
            if callable(batch_decode):
                texts = batch_decode(
                    trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                return str(texts[0] if texts else "")
            return str(self._processor.decode(trimmed[0], skip_special_tokens=True))
        if callable(batch_decode):
            texts = batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return str(texts[0] if texts else "")
        return str(self._processor.decode(output_ids[0], skip_special_tokens=True))

    def predict(self, inp: ModelInput) -> ModelOutput:
        try:
            import torch  # type: ignore[reportMissingImports]
        except ImportError:
            raise ImportError(
                "torch is required for HuggingFace models. Install with: pip install torch"
            )

        if self._model is None:
            self._load()

        if self.modality == Modality.TEXT:
            inputs = self._prepare_text_only_inputs(inp)
        else:
            inputs = self._prepare_multimodal_inputs(inp)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
            )

        text = self._decode_output_ids(inputs, output_ids)

        return ModelOutput(text=text)


# ---------------------------------------------------------------------------
# vLLM (high-throughput serving for text/VLM models)
# ---------------------------------------------------------------------------


@register_model("vllm")
class VLLMModel(BaseModel):
    """High-throughput local inference via vLLM.

    Supports both text-only and vision-language models.  VL models
    (auto-detected by name containing ``VL``, ``vision``, or ``Visual``,
    or when *modality* is set explicitly) use the ``llm.chat()`` API
    with proper image handling.  For diffusion models, see
    ``VLLMDiffusionModel`` below.

    Example::

        model = load_model("vllm", model_id="meta-llama/Llama-3-8b-instruct")
        model = load_model("vllm", model_id="Qwen/Qwen3-VL-8B-Instruct",
                           temperature=0.7, top_k=20, top_p=0.8,
                           presence_penalty=1.5)
    """

    def __init__(
        self,
        model_id: str = "meta-llama/Llama-3-8b-instruct",
        tensor_parallel_size: int = 1,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        modality: Optional[Union[str, Modality]] = None,
        limit_mm_per_prompt: Optional[Dict[str, int]] = None,
        allowed_local_media_path: Optional[str] = None,
        max_num_batched_tokens: Optional[int] = None,
        enable_thinking: bool = True,
        **kwargs: Any,
    ):
        self.model_id = model_id
        self.name = model_id.split("/")[-1]
        self.tensor_parallel_size = tensor_parallel_size
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.presence_penalty = presence_penalty
        self.limit_mm_per_prompt = limit_mm_per_prompt or {"image": 5}
        self.allowed_local_media_path = allowed_local_media_path or "/"
        self.max_num_batched_tokens = max_num_batched_tokens
        self.enable_thinking = enable_thinking

        self.modality = _resolve_text_model_modality(model_id, modality)

        self.supports_image_output = False
        self.supports_video_output = False
        self.supports_mask_editing = False
        self.supports_image_input = self.modality == Modality.TEXT_AND_IMAGE

        self._llm = None

    def _load(self) -> None:
        try:
            from vllm import LLM
        except ImportError:
            raise ImportError(
                'vllm is required for this model. Install with: pip install -e ".[vllm]"'
            )

        kwargs: Dict[str, Any] = dict(
            model=self.model_id,
            tensor_parallel_size=self.tensor_parallel_size,
        )
        kwargs["allowed_local_media_path"] = self.allowed_local_media_path
        if self.modality == Modality.TEXT_AND_IMAGE:
            kwargs["limit_mm_per_prompt"] = self.limit_mm_per_prompt
        if self.max_num_batched_tokens is not None:
            kwargs["max_num_batched_tokens"] = self.max_num_batched_tokens

        self._llm = LLM(**kwargs)

    def _sampling_params(self) -> Any:
        try:
            from vllm import SamplingParams
        except ImportError:
            raise ImportError(
                'vllm is required for this model. Install with: pip install -e ".[vllm]"'
            )

        return SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            presence_penalty=self.presence_penalty,
        )

    # -- message builders for chat API --

    @staticmethod
    def _build_message(inp: ModelInput) -> List[Dict[str, Any]]:
        """Convert a ModelInput into vLLM chat messages."""
        content: List[Dict[str, Any]] = []
        for img in inp.images:
            if isinstance(img, (str, Path)):
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"file://{Path(img).resolve()}"},
                })
            elif isinstance(img, bytes):
                import base64

                b64 = base64.b64encode(img).decode()
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                })
            else:
                # PIL Image — save to temp file
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                    img.save(f, format="PNG")
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"file://{f.name}"},
                    })
        content.append({"type": "text", "text": inp.text})
        return [{"role": "user", "content": content}]

    # -- predict --

    def predict(self, inp: ModelInput) -> ModelOutput:
        if self._llm is None:
            self._load()

        chat_kwargs = {}
        if not self.enable_thinking:
            chat_kwargs["chat_template_kwargs"] = {"enable_thinking": False}

        messages = self._build_message(inp)
        outputs = self._llm.chat(messages, self._sampling_params(), **chat_kwargs)

        text = outputs[0].outputs[0].text
        return ModelOutput(text=text)

    def predict_batch(self, inputs: List[ModelInput]) -> List[ModelOutput]:
        """vLLM natively supports batched inference."""
        if self._llm is None:
            self._load()

        chat_kwargs = {}
        if not self.enable_thinking:
            chat_kwargs["chat_template_kwargs"] = {"enable_thinking": False}

        messages_list = [self._build_message(inp) for inp in inputs]
        outputs = self._llm.chat(messages_list, self._sampling_params(), **chat_kwargs)

        return [ModelOutput(text=o.outputs[0].text) for o in outputs]


# ---------------------------------------------------------------------------
# vLLM-Omni / Diffusion (image generation via vllm-project/vllm-omni)
# ---------------------------------------------------------------------------


@register_model("diffusion")
class VLLMDiffusionModel(BaseModel):
    """Diffusion models via vllm-omni or official FLUX.2 runtime.

    Uses https://github.com/vllm-project/vllm-omni for high-throughput
    diffusion inference (Flux, SDXL, etc.).  For ``flux.2-*`` model IDs,
    this class delegates to ``Flux2Model`` while keeping the same CLI
    surface (`--provider diffusion`).

    Requires: ``pip install -e ".[vllm-omni]"``

    Example::

        model = load_model("diffusion",
                           model_id="black-forest-labs/FLUX.1-schnell")

        out = model.predict(ModelInput(text="a cat sitting on a desk"))
        out.images[0].save("output.png")
    """

    modality = Modality.IMAGE_GENERATION

    def __init__(
        self,
        model_id: str = "black-forest-labs/FLUX.1-schnell",
        resolution: int = 1024,
        seed: Optional[int] = None,
        **kwargs: Any,
    ):
        self.model_id = model_id
        self.name = model_id.split("/")[-1]
        self.resolution = resolution
        self.seed = seed
        self.supports_image_output = True
        self.supports_video_output = False
        self.supports_mask_editing = self._uses_flux2()
        self.supports_image_input = self._uses_flux2()
        self._omni = None
        self._delegate: Optional[BaseModel] = None

    def _uses_flux2(self) -> bool:
        return str(self.model_id or "").strip().lower().startswith("flux.2-")

    def _load(self) -> None:
        if self._uses_flux2():
            self._delegate = Flux2Model(
                model_name=self.model_id,
                seed=self.seed,
                default_width=self.resolution,
                default_height=self.resolution,
            )
            return
        try:
            from vllm_omni import Omni  # type: ignore[reportMissingImports]
        except ImportError:
            raise ImportError(
                'vllm-omni is required for diffusion models. Install with: pip install -e ".[vllm-omni]"'
            )

        self._omni = Omni(model=self.model_id)

    def _sampling_params(self) -> Any:
        try:
            from vllm_omni.inputs.data import OmniDiffusionSamplingParams  # type: ignore[reportMissingImports]
        except ImportError:
            raise ImportError(
                'vllm-omni is required for diffusion models. Install with: pip install -e ".[vllm-omni]"'
            )

        return OmniDiffusionSamplingParams(
            resolution=self.resolution,
            seed=self.seed,
        )

    def predict(self, inp: ModelInput) -> ModelOutput:
        if self._omni is None and self._delegate is None:
            self._load()
        if self._delegate is not None:
            return self._delegate.predict(inp)

        outputs = self._omni.generate(
            prompts=inp.text,
            sampling_params_list=[self._sampling_params()],
        )
        images = outputs[0].images if outputs else []

        return ModelOutput(images=images, raw=outputs)

    def predict_batch(self, inputs: List[ModelInput]) -> List[ModelOutput]:
        """vllm-omni supports batched diffusion generation."""
        if self._omni is None and self._delegate is None:
            self._load()
        if self._delegate is not None:
            return self._delegate.predict_batch(inputs)

        prompts = [inp.text for inp in inputs]
        sp = self._sampling_params()
        outputs = self._omni.generate(
            prompts=prompts,
            sampling_params_list=[sp],
        )

        return [
            ModelOutput(images=o.images, raw=o) for o in outputs
        ]

# ---------------------------------------------------------------------------
# User-provided local Python models / wrappers
# ---------------------------------------------------------------------------

_MEDIA_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".mp4",
    ".webm",
    ".mov",
    ".avi",
}


def _looks_like_media(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (bytes, bytearray, Path)):
        return True
    if isinstance(value, str):
        lowered = value.lower().strip()
        if lowered.startswith(("http://", "https://", "data:image/", "file://")):
            return True
        suffix = Path(lowered).suffix.lower()
        return suffix in _MEDIA_SUFFIXES
    try:
        from PIL import Image  # type: ignore[reportMissingImports]
    except ImportError:
        return False
    return isinstance(value, Image.Image)


def _coerce_model_output(raw: Any) -> ModelOutput:
    if isinstance(raw, ModelOutput):
        return raw

    if isinstance(raw, dict):
        text = str(raw.get("text", "")) if "text" in raw else ""
        images_obj = raw.get("images")
        if images_obj is None and "image" in raw:
            images_obj = [raw.get("image")]
        if images_obj is None:
            images = []
        elif isinstance(images_obj, list):
            images = images_obj
        else:
            images = [images_obj]
        usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
        return ModelOutput(text=text, images=images, raw=raw, usage=usage)

    if isinstance(raw, (list, tuple)):
        return ModelOutput(images=list(raw), raw=raw)

    if _looks_like_media(raw):
        return ModelOutput(images=[raw], raw=raw)

    if raw is None:
        return ModelOutput(text="", raw=raw)

    return ModelOutput(text=str(raw), raw=raw)


@register_model("custom")
class CustomEntrypointModel(BaseModel):
    """Dynamic Python entrypoint model loader.

    Supported entrypoints:
    - ``module.submodule:MyModelClass`` (class, preferably ``BaseModel`` subclass)
    - ``module.submodule:build_model`` (factory returning ``BaseModel`` or callable)
    - ``module.submodule:predict`` (callable: ``ModelInput -> ModelOutput/Any``)

    Optional capability attributes on the returned implementation improve
    preflight warnings for generation tasks:
    - ``supports_image_output``
    - ``supports_image_input``
    - ``supports_mask_editing``
    - ``supports_video_output``
    """

    modality = Modality.ANY

    def __init__(
        self,
        entrypoint: str,
        init_kwargs: Optional[Dict[str, Any]] = None,
        modality: Union[str, Modality] = Modality.ANY,
        **kwargs: Any,
    ):
        if ":" not in str(entrypoint):
            raise ValueError(
                "custom provider requires entrypoint in 'module.path:attr' format "
                f"(got {entrypoint!r})"
            )
        self.entrypoint = str(entrypoint)
        self.name = f"custom:{self.entrypoint}"
        self.modality = _modality_from_value(modality)
        self.init_kwargs = dict(init_kwargs or {})
        self._impl = self._load_entrypoint()
        self.supports_image_output = bool(
            getattr(
                self._impl,
                "supports_image_output",
                self.modality in {Modality.IMAGE_GENERATION, Modality.ANY},
            )
        )
        self.supports_video_output = bool(getattr(self._impl, "supports_video_output", False))
        self.supports_mask_editing = bool(getattr(self._impl, "supports_mask_editing", False))
        self.supports_image_input = bool(
            getattr(
                self._impl,
                "supports_image_input",
                self.modality in {Modality.IMAGE, Modality.TEXT_AND_IMAGE, Modality.ANY},
            )
        )

    def _load_entrypoint(self) -> Any:
        module_name, attr_name = self.entrypoint.split(":", 1)
        module = importlib.import_module(module_name)
        if not hasattr(module, attr_name):
            raise AttributeError(f"{self.entrypoint!r}: attribute {attr_name!r} not found.")
        target = getattr(module, attr_name)

        if inspect.isclass(target):
            instance = target(**self.init_kwargs)
            return instance

        if callable(target):
            if self.init_kwargs:
                produced = target(**self.init_kwargs)
                if isinstance(produced, BaseModel) or callable(produced):
                    return produced
                raise TypeError(
                    f"{self.entrypoint!r} returned unsupported factory value: {type(produced)!r}",
                )
            try:
                sig = inspect.signature(target)
                required = [
                    p
                    for p in sig.parameters.values()
                    if p.default is inspect._empty
                    and p.kind in (
                        inspect.Parameter.POSITIONAL_ONLY,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.KEYWORD_ONLY,
                    )
                ]
            except Exception:  # noqa: BLE001
                required = [object()]
            if not required:
                produced = target()
                if isinstance(produced, BaseModel) or callable(produced):
                    return produced
                raise TypeError(
                    f"{self.entrypoint!r} returned unsupported factory value: {type(produced)!r}",
                )
            return target

        raise TypeError(
            f"{self.entrypoint!r} must point to a class or callable, got {type(target)!r}",
        )

    def predict(self, inp: ModelInput) -> ModelOutput:
        if isinstance(self._impl, BaseModel):
            return _coerce_model_output(self._impl.predict(inp))

        result = self._impl(inp)
        return _coerce_model_output(result)

    def predict_batch(self, inputs: List[ModelInput]) -> List[ModelOutput]:
        if isinstance(self._impl, BaseModel):
            return [_coerce_model_output(out) for out in self._impl.predict_batch(inputs)]

        batch_fn = getattr(self._impl, "predict_batch", None)
        if callable(batch_fn):
            outputs = batch_fn(inputs)
            return [_coerce_model_output(out) for out in outputs]

        return [self.predict(inp) for inp in inputs]


class Flux2Model(BaseModel):
    """Official FLUX.2 wrapper for text-to-image and multi-image editing.

    This wrapper is intended to be used through the existing ``custom`` provider
    so ``run_benchmarks.py`` does not need any new provider-specific logic.

    Example::

        python scripts/run_benchmarks.py --benchmarks layout-8 \
            --provider custom \
            --custom-entry design_benchmarks.models.local_models:Flux2Model \
            --custom-init-kwargs '{"model_name":"flux.2-klein-4b"}' \
            --custom-modality image_generation
    """

    modality = Modality.IMAGE_GENERATION
    supports_image_output = True
    supports_image_input = True
    supports_mask_editing = True
    supports_video_output = False

    _INSTALL_HINT = (
        'FLUX.2 wrapper requires the official package. Install with: '
        'python -m pip install --no-deps --ignore-requires-python '
        '"git+https://github.com/black-forest-labs/flux2.git"'
    )

    def __init__(
        self,
        model_name: str = "flux.2-klein-4b",
        device: str = "cuda",
        num_steps: Optional[int] = None,
        guidance: Optional[float] = None,
        seed: Optional[int] = None,
        debug_mode: bool = False,
        preserve_unmasked_regions: bool = True,
        default_width: int = 1024,
        default_height: int = 1024,
        **kwargs: Any,
    ):
        if kwargs:
            warnings.warn(
                f"Flux2Model received unexpected kwargs: {list(kwargs)}",
                stacklevel=2,
            )
        self.model_name = str(model_name).strip().lower()
        self.name = self.model_name
        self.device = str(device or "cuda").strip() or "cuda"
        self.num_steps = int(num_steps) if num_steps is not None else None
        self.guidance = float(guidance) if guidance is not None else None
        self.seed = int(seed) if seed is not None else None
        self.debug_mode = bool(debug_mode)
        self.preserve_unmasked_regions = bool(preserve_unmasked_regions)
        self.default_width = max(64, int(default_width))
        self.default_height = max(64, int(default_height))
        self._bundle: Optional[Dict[str, Any]] = None

    def _load(self) -> None:
        try:
            import huggingface_hub  # type: ignore[reportMissingImports]
            import torch  # type: ignore[reportMissingImports]
            from einops import rearrange  # type: ignore[reportMissingImports]
            from flux2.sampling import (  # type: ignore[reportMissingImports]
                batched_prc_img,
                batched_prc_txt,
                denoise,
                denoise_cached,
                denoise_cfg,
                encode_image_refs,
                get_schedule,
                scatter_ids,
            )
            from flux2.util import (  # type: ignore[reportMissingImports]
                FLUX2_MODEL_INFO,
                load_ae,
                load_flow_model,
                load_text_encoder,
            )
            from flux2.text_encoder import Qwen3Embedder  # type: ignore[reportMissingImports]
        except ImportError as exc:
            raise ImportError(self._INSTALL_HINT) from exc

        device_obj = torch.device(self.device)
        if device_obj.type != "cuda":
            raise ValueError("Flux2Model currently requires a CUDA device.")
        if device_obj.index is not None:
            torch.cuda.set_device(device_obj.index)

        if self.model_name not in FLUX2_MODEL_INFO:
            available = ", ".join(sorted(FLUX2_MODEL_INFO))
            raise ValueError(
                f"Unknown FLUX.2 model {self.model_name!r}. "
                f"Choose from: {available}"
            )

        model_info = FLUX2_MODEL_INFO[self.model_name]
        self._ensure_hf_artifact(
            huggingface_hub=huggingface_hub,
            repo_id=str(model_info["repo_id"]),
            filename=str(model_info["filename"]),
            env_var=str(model_info.get("model_path") or ""),
            label=f"{self.model_name} weights",
        )
        self._ensure_hf_artifact(
            huggingface_hub=huggingface_hub,
            repo_id=str(model_info.get("ae_repo_id") or model_info["repo_id"]),
            filename=str(model_info["filename_ae"]),
            env_var="AE_MODEL_PATH",
            label=f"{self.model_name} autoencoder",
        )

        model = load_flow_model(
            self.model_name,
            debug_mode=self.debug_mode,
            device=device_obj,
        )
        ae = load_ae(self.model_name, device=device_obj)
        text_encoder = self._load_flux2_text_encoder(
            torch=torch,
            load_text_encoder=load_text_encoder,
            qwen3_embedder_cls=Qwen3Embedder,
            model_name=self.model_name,
            device=device_obj,
        )

        model.eval()
        ae.eval()
        text_encoder.eval()

        self._bundle = {
            "torch": torch,
            "rearrange": rearrange,
            "model_info": model_info,
            "model": model,
            "ae": ae,
            "text_encoder": text_encoder,
            "batched_prc_img": batched_prc_img,
            "batched_prc_txt": batched_prc_txt,
            "denoise": denoise,
            "denoise_cached": denoise_cached,
            "denoise_cfg": denoise_cfg,
            "encode_image_refs": encode_image_refs,
            "get_schedule": get_schedule,
            "scatter_ids": scatter_ids,
            "device": device_obj,
        }

    def _ensure_loaded(self) -> Dict[str, Any]:
        if self._bundle is None:
            self._load()
        assert self._bundle is not None
        return self._bundle

    @staticmethod
    def _device_supports_fp8(torch: Any, device_obj: Any) -> bool:
        if getattr(device_obj, "type", None) != "cuda":
            return False
        major, minor = torch.cuda.get_device_capability(device_obj)
        return (major, minor) >= (8, 9)

    def _load_flux2_text_encoder(
        self,
        *,
        torch: Any,
        load_text_encoder: Any,
        qwen3_embedder_cls: Any,
        model_name: str,
        device: Any,
    ) -> Any:
        if "klein" not in model_name:
            return load_text_encoder(model_name, device=device)

        if self._device_supports_fp8(torch, device):
            return load_text_encoder(model_name, device=device)

        variant = "8B" if "9b" in model_name else "4B"
        fallback_model_id = f"Qwen/Qwen3-{variant}"
        return qwen3_embedder_cls(model_spec=fallback_model_id, device=device)

    def _ensure_hf_artifact(
        self,
        *,
        huggingface_hub: Any,
        repo_id: str,
        filename: str,
        env_var: str,
        label: str,
    ) -> str:
        """Download a HF artifact if not already cached.

        Side effect: when *env_var* is non-empty the resolved local path is
        written to ``os.environ[env_var]`` so that downstream FLUX.2 loaders
        (which read those env vars) find the file without extra configuration.
        """
        configured_path = os.environ.get(env_var) if env_var else None
        if configured_path:
            path = Path(configured_path).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"{env_var} points to a missing file: {path}")
            return str(path)

        token = self._resolve_hf_token(huggingface_hub)
        try:
            path = huggingface_hub.hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="model",
                token=token,
            )
        except huggingface_hub.errors.GatedRepoError as exc:
            env_hint = (
                f" set `{env_var}` to the downloaded file path,"
                if env_var
                else ""
            )
            raise PermissionError(
                f"{label} is gated on Hugging Face ({repo_id}/{filename}). "
                f"Request access at https://huggingface.co/{repo_id}, then export "
                f"`HF_TOKEN`/`HF_HUB_TOKEN` before running or{env_hint} and retry."
            ) from exc
        except huggingface_hub.errors.RepositoryNotFoundError as exc:
            raise FileNotFoundError(
                f"Could not resolve {label} at {repo_id}/{filename}."
            ) from exc
        except huggingface_hub.errors.HfHubHTTPError as exc:
            raise RuntimeError(
                f"Failed to download {label} from {repo_id}/{filename}: {exc}"
            ) from exc

        if env_var:
            os.environ[env_var] = str(path)
        return str(path)

    @staticmethod
    def _resolve_hf_token(huggingface_hub: Any) -> Optional[str]:
        for key in ("HF_TOKEN", "HF_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            value = os.environ.get(key)
            if value:
                return value

        token = huggingface_hub.get_token()
        if token:
            return token

        for candidate in (
            Path.home() / ".cache" / "huggingface" / "token",
            Path.home() / ".huggingface" / "token",
        ):
            try:
                text = candidate.read_text(encoding="utf-8").strip()
            except Exception:
                continue
            if text:
                return text
        return None

    @staticmethod
    def _coerce_int(value: Any) -> int:
        try:
            return int(float(value))
        except Exception:
            return 0

    @staticmethod
    def _normalize_size_dim(value: int) -> int:
        value = max(64, int(value))
        return max(64, ((value + 8) // 16) * 16)

    @staticmethod
    def _coerce_pil_image(value: Any) -> Optional[Any]:
        try:
            from PIL import Image  # type: ignore[reportMissingImports]
            import io
        except ImportError:
            return None

        if value is None:
            return None
        if isinstance(value, Image.Image):
            return value.convert("RGB")
        if isinstance(value, (bytes, bytearray)):
            return Image.open(io.BytesIO(value)).convert("RGB")
        if isinstance(value, (str, Path)):
            path = Path(value)
            if path.is_file():
                return Image.open(path).convert("RGB")
        return None

    @staticmethod
    def _normalize_reference_image(image: Any) -> Any:
        try:
            from PIL import Image  # type: ignore[reportMissingImports]
        except ImportError:
            return image

        width, height = image.size
        min_side = min(width, height)
        if min_side >= 64:
            return image

        scale = 64.0 / float(max(min_side, 1))
        new_width = max(64, int(round(width * scale)))
        new_height = max(64, int(round(height * scale)))
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def _collect_input_images(self, inp: ModelInput) -> List[Any]:
        images: List[Any] = []
        for value in inp.images:
            image = self._coerce_pil_image(value)
            if image is not None:
                images.append(self._normalize_reference_image(image))
        return images

    def _resolve_output_size(self, inp: ModelInput, input_images: List[Any]) -> Tuple[int, int]:
        meta = inp.metadata or {}
        width = (
            self._coerce_int(meta.get("target_width"))
            or self._coerce_int(meta.get("dataset_target_width"))
            or self._coerce_int(meta.get("width"))
        )
        height = (
            self._coerce_int(meta.get("target_height"))
            or self._coerce_int(meta.get("dataset_target_height"))
            or self._coerce_int(meta.get("height"))
        )
        if width > 0 and height > 0:
            return (
                self._normalize_size_dim(width),
                self._normalize_size_dim(height),
            )

        if input_images:
            return (
                self._normalize_size_dim(input_images[0].size[0]),
                self._normalize_size_dim(input_images[0].size[1]),
            )

        mask_image = self._coerce_pil_image(meta.get("mask"))
        if mask_image is not None:
            return (
                self._normalize_size_dim(mask_image.size[0]),
                self._normalize_size_dim(mask_image.size[1]),
            )

        return (
            self._normalize_size_dim(self.default_width),
            self._normalize_size_dim(self.default_height),
        )

    def _resolve_seed(self, bundle: Dict[str, Any], inp: ModelInput) -> int:
        torch = bundle["torch"]
        if self.seed is not None:
            return self.seed
        candidate = self._coerce_int((inp.metadata or {}).get("seed"))
        if candidate > 0:
            return candidate
        return int(torch.randint(0, 2**31 - 1, (1,), device="cpu").item())

    def _resolve_num_steps(self, model_info: Dict[str, Any]) -> int:
        default_steps = self._coerce_int((model_info.get("defaults") or {}).get("num_steps"))
        if self.num_steps is None:
            return default_steps or 4
        fixed = set(model_info.get("fixed_params", set()))
        if "num_steps" in fixed and self.num_steps != default_steps:
            raise ValueError(
                f"{self.model_name} requires num_steps={default_steps}, got {self.num_steps}."
            )
        return self.num_steps

    def _resolve_guidance(self, model_info: Dict[str, Any]) -> float:
        default_guidance = float((model_info.get("defaults") or {}).get("guidance", 1.0))
        if self.guidance is None:
            return default_guidance
        fixed = set(model_info.get("fixed_params", set()))
        if "guidance" in fixed and abs(self.guidance - default_guidance) > 1e-6:
            raise ValueError(
                f"{self.model_name} requires guidance={default_guidance}, got {self.guidance}."
            )
        return self.guidance

    def _compose_masked_output(self, inp: ModelInput, input_images: List[Any], generated: Any) -> Any:
        if not self.preserve_unmasked_regions or not input_images:
            return generated

        if (inp.metadata or {}).get("skip_mask_composition", False):
            return generated

        mask = self._coerce_pil_image((inp.metadata or {}).get("mask"))
        if mask is None:
            return generated

        try:
            from PIL import Image  # type: ignore[reportMissingImports]
        except ImportError:
            return generated

        base = input_images[0].convert("RGB")
        output = generated.convert("RGB")
        if output.size != base.size:
            output = output.resize(base.size, Image.Resampling.LANCZOS)
        mask_l = mask.convert("L")
        if mask_l.size != base.size:
            mask_l = mask_l.resize(base.size, Image.Resampling.NEAREST)
        if mask_l.getextrema() == (255, 255):
            return generated
        return Image.composite(output, base, mask_l)

    def predict(self, inp: ModelInput) -> ModelOutput:
        bundle = self._ensure_loaded()
        torch = bundle["torch"]
        rearrange = bundle["rearrange"]
        model_info = bundle["model_info"]
        device_obj = bundle["device"]

        if device_obj.index is not None:
            torch.cuda.set_device(device_obj.index)

        prompt = str(inp.text or (inp.metadata or {}).get("prompt") or "").strip()
        input_images = self._collect_input_images(inp)
        width, height = self._resolve_output_size(inp, input_images)
        num_steps = self._resolve_num_steps(model_info)
        guidance = self._resolve_guidance(model_info)
        seed = self._resolve_seed(bundle, inp)

        with torch.inference_mode():
            ref_tokens = None
            ref_ids = None
            if input_images:
                ref_tokens, ref_ids = bundle["encode_image_refs"](bundle["ae"], input_images)

            if model_info.get("guidance_distilled", False):
                ctx = bundle["text_encoder"]([prompt]).to(torch.bfloat16)
            else:
                ctx_empty = bundle["text_encoder"]([""]).to(torch.bfloat16)
                ctx_prompt = bundle["text_encoder"]([prompt]).to(torch.bfloat16)
                ctx = torch.cat([ctx_empty, ctx_prompt], dim=0)
            ctx, ctx_ids = bundle["batched_prc_txt"](ctx)

            shape = (1, 128, height // 16, width // 16)
            generator = torch.Generator(device=str(device_obj)).manual_seed(seed)
            randn = torch.randn(shape, generator=generator, dtype=torch.bfloat16, device=device_obj)
            x, x_ids = bundle["batched_prc_img"](randn)

            timesteps = bundle["get_schedule"](num_steps, x.shape[1])
            if model_info.get("guidance_distilled", False):
                denoise_fn = (
                    bundle["denoise_cached"]
                    if model_info.get("use_kv_cache") and ref_tokens is not None
                    else bundle["denoise"]
                )
                x = denoise_fn(
                    bundle["model"],
                    x,
                    x_ids,
                    ctx,
                    ctx_ids,
                    timesteps=timesteps,
                    guidance=guidance,
                    img_cond_seq=ref_tokens,
                    img_cond_seq_ids=ref_ids,
                )
            else:
                x = bundle["denoise_cfg"](
                    bundle["model"],
                    x,
                    x_ids,
                    ctx,
                    ctx_ids,
                    timesteps=timesteps,
                    guidance=guidance,
                    img_cond_seq=ref_tokens,
                    img_cond_seq_ids=ref_ids,
                )

            x = torch.cat(bundle["scatter_ids"](x, x_ids)).squeeze(2)
            x = bundle["ae"].decode(x).float().clamp(-1, 1)
            x = rearrange(x[0], "c h w -> h w c")

            from PIL import Image  # type: ignore[reportMissingImports]

            generated = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

        generated = self._compose_masked_output(inp, input_images, generated)

        return ModelOutput(
            images=[generated],
            raw={
                "model_name": self.model_name,
                "seed": seed,
                "width": width,
                "height": height,
                "num_steps": num_steps,
                "guidance": guidance,
                "input_image_count": len(input_images),
            },
            usage={
                "seed": seed,
                "width": width,
                "height": height,
                "num_steps": num_steps,
                "guidance": guidance,
                "input_image_count": len(input_images),
            },
        )


def build_flux2_model(**kwargs: Any) -> Flux2Model:
    """Convenience factory for ``--custom-entry ...:build_flux2_model``."""
    return Flux2Model(**kwargs)

