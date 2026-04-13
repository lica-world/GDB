"""Local / open-source model templates (HuggingFace, vLLM, diffusion).

These are starter templates for running open-source models on local GPUs.
Fill in or adjust the ``predict`` method as needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .base import BaseModel, Modality, ModelInput, ModelOutput
from .registry import register_model

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
        modality: Modality = Modality.TEXT,
        **kwargs: Any,
    ):
        self.model_id = model_id
        self.name = model_id.split("/")[-1]
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.modality = modality
        self._model = None
        self._processor = None

    def _load(self) -> None:
        """Lazy-load model and tokenizer/processor."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
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
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                device_map=self.device,
            )

    def predict(self, inp: ModelInput) -> ModelOutput:
        try:
            import torch
        except ImportError:
            raise ImportError(
                "torch is required for HuggingFace models. Install with: pip install torch"
            )

        if self._model is None:
            self._load()

        if self.modality == Modality.TEXT:
            inputs = self._processor(inp.text, return_tensors="pt").to(self.device)
        else:
            # For multimodal: adapt based on the specific model's processor
            from PIL import Image

            images = [
                Image.open(img) if isinstance(img, (str, Path)) else img
                for img in inp.images
            ]
            inputs = self._processor(
                text=inp.text,
                images=images or None,
                return_tensors="pt",
            ).to(self.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
            )

        # Decode only the new tokens
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        text = self._processor.decode(new_tokens, skip_special_tokens=True)

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

    _VL_PATTERNS = ("VL", "vision", "Visual")

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

        # Auto-detect VL modality from model name
        if modality is not None:
            self.modality = Modality(modality) if isinstance(modality, str) else modality
        elif any(p.lower() in model_id.lower() for p in self._VL_PATTERNS):
            self.modality = Modality.TEXT_AND_IMAGE
        else:
            self.modality = Modality.TEXT

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
    """Diffusion models via vllm-omni (offline Python API).

    Uses https://github.com/vllm-project/vllm-omni for high-throughput
    diffusion inference (Flux, SDXL, etc.).  The model is loaded lazily
    on first ``predict`` call.

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
        self._omni = None

    def _load(self) -> None:
        try:
            from vllm_omni import Omni
        except ImportError:
            raise ImportError(
                'vllm-omni is required for diffusion models. Install with: pip install -e ".[vllm-omni]"'
            )

        self._omni = Omni(model=self.model_id)

    def _sampling_params(self) -> Any:
        try:
            from vllm_omni.inputs.data import OmniDiffusionSamplingParams
        except ImportError:
            raise ImportError(
                'vllm-omni is required for diffusion models. Install with: pip install -e ".[vllm-omni]"'
            )

        return OmniDiffusionSamplingParams(
            resolution=self.resolution,
            seed=self.seed,
        )

    def predict(self, inp: ModelInput) -> ModelOutput:
        if self._omni is None:
            self._load()

        outputs = self._omni.generate(
            prompts=inp.text,
            sampling_params_list=[self._sampling_params()],
        )
        images = outputs[0].images if outputs else []

        return ModelOutput(images=images, raw=outputs)

    def predict_batch(self, inputs: List[ModelInput]) -> List[ModelOutput]:
        """vllm-omni supports batched diffusion generation."""
        if self._omni is None:
            self._load()

        prompts = [inp.text for inp in inputs]
        sp = self._sampling_params()
        outputs = self._omni.generate(
            prompts=prompts,
            sampling_params_list=[sp],
        )

        return [
            ModelOutput(images=o.images, raw=o) for o in outputs
        ]
