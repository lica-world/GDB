"""HELM Metric that wraps GDB's corpus-level ``evaluate()`` method.

Overrides ``MetricInterface.evaluate()`` so we can collect all predictions
in one pass and delegate to ``bench.evaluate(predictions, ground_truths)``.
This preserves exact metric parity with standalone GDB.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.metrics.metric import MetricInterface, MetricResult
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat

logger = logging.getLogger(__name__)


def _extract_completion_text(request_state: Any) -> str:
    """Pull the raw text from HELM's RequestState result."""
    if request_state.result is None:
        return ""
    completions = request_state.result.completions
    if not completions:
        return ""
    return completions[0].text or ""


def _extract_image_location(request_state: Any) -> Optional[str]:
    """Pull a generated image location from RequestState (HEIM pattern)."""
    if request_state.result is None:
        return None
    completions = request_state.result.completions
    if not completions:
        return None
    mc = completions[0].multimodal_content
    if mc is None or mc.size == 0:
        return None
    loc = mc.media_objects[0].location
    return loc


class GDBCorpusMetric(MetricInterface):
    """Evaluates a GDB benchmark by delegating to its ``evaluate()`` method.

    This metric collects all model completions across the full evaluation set,
    runs them through ``bench.parse_model_output()``, then calls
    ``bench.evaluate(predictions, ground_truths)`` to get the scores.
    """

    def __init__(self, benchmark_id: str, image_gen: bool = False):
        self.benchmark_id = benchmark_id
        self.image_gen = image_gen

    def _get_benchmark(self) -> Any:
        from gdb.registry import BenchmarkRegistry

        registry = BenchmarkRegistry()
        registry.discover()
        return registry.get(self.benchmark_id)

    def evaluate(
        self,
        scenario_state: ScenarioState,
        metric_service: MetricService,
        eval_cache_path: str,
        parallelism: int = 1,
    ) -> MetricResult:
        from gdb.models.base import ModelOutput

        bench = self._get_benchmark()
        predictions: List[Any] = []
        ground_truths: List[Any] = []

        for request_state in scenario_state.request_states:
            gt_text = ""
            if request_state.instance.references:
                gt_text = request_state.instance.references[0].output.text

            extra = request_state.instance.extra_data or {}
            gt = extra.get("ground_truth", gt_text)

            if self.image_gen:
                img_loc = _extract_image_location(request_state)
                if img_loc:
                    output = ModelOutput(text="", images=[img_loc])
                else:
                    output = ModelOutput(text=_extract_completion_text(request_state))
            else:
                output = ModelOutput(text=_extract_completion_text(request_state))

            try:
                pred = bench.parse_model_output(output)
            except Exception as exc:
                logger.warning(
                    "GDB parse_model_output failed for %s sample %s: %s",
                    self.benchmark_id,
                    request_state.instance.id,
                    exc,
                )
                pred = ""

            predictions.append(pred)
            ground_truths.append(gt)

        try:
            scores: Dict[str, float] = bench.evaluate(predictions, ground_truths)
        except Exception as exc:
            logger.error(
                "GDB evaluate() failed for %s: %s", self.benchmark_id, exc
            )
            scores = {}

        aggregate_stats: List[Stat] = []
        per_instance_stats: List[Stat] = []
        for metric_name, value in scores.items():
            aggregate_stats.append(
                Stat(f"gdb_{metric_name}").add(value)
            )

        return MetricResult(
            aggregated_stats=aggregate_stats,
            per_instance_stats=per_instance_stats,
        )
