"""Shared inference layer for design benchmarks.

Four runners available — all share the same contract:

- ``run(requests)``        → submit + poll + return (blocking)
- ``submit(requests)``     → fire-and-forget, returns batch_id
- ``collect(batch_id, …)`` → poll + return results

Runners:

- ``GeminiBatchRunner``    — Vertex AI batch prediction (~50% cheaper).
- ``OpenAIBatchRunner``    — OpenAI Batch API (~50% cheaper).
- ``AnthropicBatchRunner`` — Anthropic Message Batches API (~50% cheaper).
- ``BatchRunner``          — concurrent individual calls (any model, full price).

Job manifests (``save_job_manifest`` / ``load_job_manifest``) let you
submit a batch, close your terminal, and collect results later.

**GCS bucket:** provider batch runners require a Google Cloud Storage bucket
for image uploads unless inputs are already ``https://`` or ``gs://`` URLs.
Pass ``bucket=...`` to the runner or set ``DESIGN_BENCHMARKS_GCS_BUCKET``.
"""

from .api_batch_runners import (
    AnthropicBatchRunner,
    GeminiBatchRunner,
    OpenAIBatchRunner,
)
from .batch import (
    BatchRequest,
    BatchResult,
    BatchRunner,
    load_job_manifest,
    save_job_manifest,
    write_results_csv,
)

BATCH_PROVIDERS = {"gemini", "openai", "anthropic"}


def make_batch_runner(provider: str, **kwargs):
    """Factory for provider-specific batch runners.

    ``provider`` must be one of: ``gemini``, ``openai``, ``anthropic``.
    Extra kwargs are passed to the runner constructor.
    """
    if provider == "openai":
        return OpenAIBatchRunner(**kwargs)
    elif provider == "anthropic":
        return AnthropicBatchRunner(**kwargs)
    elif provider == "gemini":
        return GeminiBatchRunner(**kwargs)
    raise ValueError(
        f"No batch runner for provider {provider!r}. "
        f"Choose from: {', '.join(sorted(BATCH_PROVIDERS))}"
    )


__all__ = [
    "AnthropicBatchRunner",
    "BATCH_PROVIDERS",
    "BatchRequest",
    "BatchResult",
    "BatchRunner",
    "GeminiBatchRunner",
    "OpenAIBatchRunner",
    "load_job_manifest",
    "make_batch_runner",
    "save_job_manifest",
    "write_results_csv",
]
