"""Simple model registry for instantiating models by provider name."""

from __future__ import annotations

from typing import Any, Callable, Dict, Type

from .base import BaseModel

_REGISTRY: Dict[str, Type[BaseModel]] = {}


def register_model(provider: str) -> Callable:
    """Decorator to register a model class under a provider name.

    Usage::

        @register_model("openai")
        class OpenAIModel(BaseModel):
            ...
    """

    def decorator(cls: Type[BaseModel]) -> Type[BaseModel]:
        _REGISTRY[provider] = cls
        return cls

    return decorator


def load_model(provider: str, **kwargs: Any) -> BaseModel:
    """Instantiate a registered model by provider name.

    Args:
        provider: One of the registered provider names (e.g. "openai",
                  "anthropic", "hf", "vllm").
        **kwargs: Passed to the model constructor (model_id, api_key, etc.).

    Example::

        model = load_model("openai", model_id="gpt-4o", api_key="sk-...")
    """
    if provider not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(
            f"Unknown model provider '{provider}'. Available: {available}"
        )
    return _REGISTRY[provider](**kwargs)


# Trigger registration of built-in model templates.
from . import api_models, local_models  # noqa: E402, F401
