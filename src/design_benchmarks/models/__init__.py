"""Model interfaces for benchmarking.

Provides a uniform interface so the benchmark runner can call any model
(API-based, local open-source, text-only, or multimodal) through the same
``predict`` method.

Quick start::

    from design_benchmarks.models import load_model

    model = load_model("openai", model_id="gpt-4o", api_key="sk-...")
    model = load_model("anthropic", model_id="claude-sonnet-4-20250514")
    model = load_model("hf", model_id="google/gemma-3-4b-it", device="cuda")
"""

from .base import BaseModel, Modality
from .registry import load_model, register_model

__all__ = [
    "BaseModel",
    "Modality",
    "load_model",
    "register_model",
]
