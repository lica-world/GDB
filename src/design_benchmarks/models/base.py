"""Base model interface for benchmarking."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Union


class Modality(Enum):
    """What a model can accept and produce.

    Used to describe *capabilities* — a TEXT_AND_IMAGE model can handle both
    text-only and multimodal inputs.  An IMAGE_GENERATION model produces
    images (e.g. diffusion models).
    """

    TEXT = auto()                # text in → text out
    IMAGE = auto()              # image in → text out (vision-only, rare)
    TEXT_AND_IMAGE = auto()     # text+image in → text out (VLMs)
    IMAGE_GENERATION = auto()   # text/image in → image out (diffusion)
    ANY = auto()                # multimodal in → multimodal out (omni models)


@dataclass
class ModelInput:
    """Uniform input container for all model types.

    Attributes:
        text: Text prompt or instruction.
        images: Optional list of image inputs (file paths, URLs, or raw bytes).
        metadata: Extra context passed through from the benchmark (e.g. SVG code,
                  layout JSON).  Models can inspect or ignore this as needed.
    """

    text: str = ""
    images: List[Union[str, Path, bytes]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelOutput:
    """Uniform output container from all model types.

    Supports both text and image outputs — a model can return either or both.

    Attributes:
        text: The model's text response (empty string if image-only output).
        images: List of generated images (file paths, PIL Images, or raw bytes).
                Empty for text-only models.
        raw: The full API / inference response for debugging.
        usage: Token counts or other usage info, if available.
    """

    text: str = ""
    images: List[Any] = field(default_factory=list)
    raw: Any = None
    usage: Dict[str, Any] = field(default_factory=dict)


class BaseModel:
    """Abstract base for all model wrappers.

    Subclasses must implement ``predict``.  They should also set ``name``,
    ``modality``, and optionally override ``predict_batch`` for models that
    support batched inference (e.g. vLLM).

    Example — text model::

        class MyModel(BaseModel):
            name = "my-model"
            modality = Modality.TEXT

            def predict(self, inp: ModelInput) -> ModelOutput:
                response = my_api_call(inp.text)
                return ModelOutput(text=response)

    Example — diffusion model::

        class MyDiffusionModel(BaseModel):
            name = "my-diffusion"
            modality = Modality.IMAGE_GENERATION

            def predict(self, inp: ModelInput) -> ModelOutput:
                image = my_pipeline(inp.text)
                return ModelOutput(images=[image])
    """

    name: str = ""
    modality: Modality = Modality.TEXT

    def predict(self, inp: ModelInput) -> ModelOutput:
        """Run inference on a single input. Must be implemented by subclasses."""
        raise NotImplementedError

    def predict_batch(self, inputs: List[ModelInput]) -> List[ModelOutput]:
        """Run inference on a batch.  Default: sequential calls to ``predict``."""
        return [self.predict(inp) for inp in inputs]

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} modality={self.modality.name}>"
