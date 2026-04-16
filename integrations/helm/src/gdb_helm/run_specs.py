"""HELM run spec function for GDB benchmarks.

Registered via ``@run_spec_function("gdb")`` so that HELM discovers it
through the entry-point system. Usage::

    helm-run --run-entries gdb:benchmark_id=category-1,model=openai/gpt-4o \\
             --suite gdb-eval --max-eval-instances 50
"""

from typing import Optional

from helm.benchmark.adaptation.adapter_spec import (
    ADAPT_GENERATION,
    ADAPT_GENERATION_MULTIMODAL,
    AdapterSpec,
)
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec

from gdb_helm._benchmark_info import BENCHMARK_INFO


def _get_adapter_spec(
    method: str,
    max_tokens: int,
    image_gen: bool = False,
) -> AdapterSpec:
    kwargs = dict(
        method=method,
        global_prefix="",
        global_suffix="",
        instructions="",
        input_prefix="",
        input_suffix="",
        output_prefix="",
        output_suffix="",
        instance_prefix="",
        max_train_instances=0,
        num_outputs=1,
        temperature=0.0,
        max_tokens=max_tokens,
        stop_sequences=[],
    )

    if image_gen:
        from helm.benchmark.adaptation.adapter_spec import ImageGenerationParameters

        kwargs["image_generation_parameters"] = ImageGenerationParameters()
        kwargs["method"] = ADAPT_GENERATION
        kwargs["max_tokens"] = 0

    return AdapterSpec(**kwargs)


@run_spec_function("gdb")
def get_gdb_run_spec(
    benchmark_id: str,
    dataset_root: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> RunSpec:
    if benchmark_id not in BENCHMARK_INFO:
        available = ", ".join(sorted(BENCHMARK_INFO.keys()))
        raise ValueError(
            f"Unknown GDB benchmark_id={benchmark_id!r}. "
            f"Available: {available}"
        )

    info = BENCHMARK_INFO[benchmark_id]

    scenario_args = {"benchmark_id": benchmark_id}
    if dataset_root:
        scenario_args["dataset_root"] = dataset_root
    if max_samples is not None:
        scenario_args["max_samples"] = int(max_samples)

    scenario_spec = ScenarioSpec(
        class_name="gdb_helm.scenarios.GDBScenario",
        args=scenario_args,
    )

    adapter_spec = _get_adapter_spec(
        method=info.method,
        max_tokens=info.max_tokens,
        image_gen=info.image_gen,
    )

    metric_spec = MetricSpec(
        class_name="gdb_helm.metrics.GDBCorpusMetric",
        args={
            "benchmark_id": benchmark_id,
            "image_gen": info.image_gen,
        },
    )

    domain = benchmark_id.rsplit("-", 1)[0]
    name_suffix = f"benchmark_id={benchmark_id}"
    if dataset_root:
        name_suffix += f",dataset_root={dataset_root}"

    return RunSpec(
        name=f"gdb:{name_suffix}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=[metric_spec],
        groups=[f"gdb_{domain}", "gdb"],
    )
