"""Generic batch inference runner for design benchmarks.

Runs model inference over a list of requests concurrently, then returns
results keyed by custom_id.  Results can be serialized to the pivot CSV
format that ``BenchmarkRunner.run_from_csv()`` consumes.

Pipeline position::

    ┌─────────────────────────┐
    │  task.load_data()       │  ← task-specific (tasks/*.py)
    │  task.build_model_input │
    │  → List[BatchRequest]   │
    └──────────┬──────────────┘
               │
    ┌──────────▼──────────────┐
    │   BatchRunner.run()     │  ← SHARED (this module)
    │  → Dict[id, BatchResult]│
    └──────────┬──────────────┘
               │
    ┌──────────▼──────────────┐
    │  task.evaluate()        │  ← task-specific (tasks/*.py)
    │  → metrics dict         │
    └─────────────────────────┘

Usage::

    from design_benchmarks.inference import BatchRunner, BatchRequest
    from design_benchmarks.models.base import ModelInput

    requests = [
        BatchRequest("sample_001", ModelInput(text="...", images=["img.png"])),
        BatchRequest("sample_002", ModelInput(text="...", images=["img2.png"])),
    ]

    runner = BatchRunner(model, max_workers=4)
    results = runner.run(requests)

    for rid, result in results.items():
        print(rid, result.text, result.elapsed_s)
"""

from __future__ import annotations

import concurrent.futures
import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from ..models.base import BaseModel, ModelInput, ModelOutput


@dataclass
class BatchRequest:
    """A single inference request with a unique ID."""

    custom_id: str
    model_input: ModelInput


@dataclass
class BatchResult:
    """Result of a single inference request."""

    custom_id: str
    model_output: ModelOutput
    elapsed_s: float = 0.0
    error: Optional[str] = None

    @property
    def text(self) -> str:
        return self.model_output.text

    @property
    def success(self) -> bool:
        return self.error is None


ProgressCallback = Callable[[int, int, BatchResult], None]


class BatchRunner:
    """Concurrent inference runner for any BaseModel.

    Sends requests through a thread pool. This is the shared middle layer
    that all task demo scripts and future task runners use — task-specific
    logic stays in the load_data / build_model_input / evaluate steps.

    For providers that support native async batch APIs (OpenAI /v1/batches,
    Anthropic Message Batches, Vertex BatchPrediction), subclass and override
    ``run()`` to submit+poll instead of running concurrently.
    """

    def __init__(
        self,
        model: BaseModel,
        max_workers: int = 4,
        on_result: Optional[ProgressCallback] = None,
    ):
        self.model = model
        self.max_workers = max_workers
        self.on_result = on_result

    def run(self, requests: List[BatchRequest]) -> Dict[str, BatchResult]:
        """Run inference on all requests concurrently.

        Returns a dict mapping custom_id → BatchResult, preserving every
        request even on per-item failures.
        """
        results: Dict[str, BatchResult] = {}
        completed = 0
        total = len(requests)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers,
        ) as executor:
            future_to_req = {
                executor.submit(self._run_one, req): req for req in requests
            }
            for future in concurrent.futures.as_completed(future_to_req):
                req = future_to_req[future]
                completed += 1
                try:
                    result = future.result()
                except Exception as e:
                    result = BatchResult(
                        custom_id=req.custom_id,
                        model_output=ModelOutput(text=""),
                        error=str(e),
                    )
                results[req.custom_id] = result
                if self.on_result:
                    self.on_result(completed, total, result)

        return results

    def _run_one(self, req: BatchRequest) -> BatchResult:
        t0 = time.time()
        output = self.model.predict(req.model_input)
        elapsed = time.time() - t0
        return BatchResult(
            custom_id=req.custom_id,
            model_output=output,
            elapsed_s=round(elapsed, 2),
        )


def save_job_manifest(
    path: Union[str, Path],
    *,
    provider: str,
    batch_id: str,
    model_id: str,
    custom_ids: List[str],
    ground_truths: Dict[str, str],
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save a job manifest so results can be collected later.

    The manifest contains everything needed to resume: the provider's
    batch ID, the model used, sample IDs, and ground truths for eval.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "provider": provider,
        "batch_id": batch_id,
        "model_id": model_id,
        "custom_ids": custom_ids,
        "ground_truths": ground_truths,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        **(extra or {}),
    }
    path.write_text(json.dumps(manifest, indent=2))
    return path


def load_job_manifest(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a previously saved job manifest."""
    return json.loads(Path(path).read_text())


def write_results_csv(
    path: Union[str, Path],
    requests: List[BatchRequest],
    results: Dict[str, BatchResult],
    ground_truths: Dict[str, str],
    model_name: str,
    task_name: str,
) -> None:
    """Write results in the pivot CSV format for ``run_from_csv()``.

    Produces columns: ``sample_id, task, expected_output, {model_name}_output``
    — exactly the schema ``BenchmarkRunner.run_from_csv()`` expects.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    output_col = f"{model_name}_output"
    fieldnames = ["sample_id", "task", "expected_output", output_col]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for req in requests:
            result = results.get(req.custom_id)
            if result and result.success:
                pred = result.text.strip()
            else:
                err = result.error if result else "missing"
                pred = f"ERROR: {err}"
            writer.writerow(
                {
                    "sample_id": req.custom_id,
                    "task": task_name,
                    "expected_output": ground_truths.get(req.custom_id, ""),
                    output_col: pred,
                }
            )
