"""HELM Scenario that wraps any GDB benchmark.

One parameterized class handles all 40 benchmarks by delegating data loading
and prompt construction to the ``gdb`` package.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    TEST_SPLIT,
    Instance,
    Input,
    Output,
    Reference,
    Scenario,
    ScenarioMetadata,
)
from helm.common.media_object import MediaObject, MultimediaObject


def _content_type_for_path(path: str) -> str:
    suffix = Path(path).suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".mp4": "video/mp4",
    }.get(suffix, "image/png")


def _model_input_to_helm_input(model_input: Any) -> Input:
    """Convert a ``gdb.models.base.ModelInput`` to a HELM ``Input``."""
    image_paths = [
        str(p) for p in (model_input.images or [])
        if p and str(p).strip()
    ]

    if not image_paths:
        return Input(text=model_input.text)

    media_objects: List[MediaObject] = []
    for img_path in image_paths:
        media_objects.append(
            MediaObject(
                content_type=_content_type_for_path(img_path),
                location=img_path,
            )
        )
    if model_input.text:
        media_objects.append(
            MediaObject(content_type="text/plain", text=model_input.text)
        )

    return Input(multimedia_content=MultimediaObject(media_objects))


class GDBScenario(Scenario):
    """Wraps a single GDB benchmark as a HELM Scenario.

    All benchmark logic (data loading, prompt construction) is delegated
    to the ``gdb`` package.  This class just converts between types.
    """

    name = "gdb"
    description = "GraphicDesignBench (GDB) benchmark scenarios"
    tags = ["gdb", "graphic_design", "multimodal"]

    def __init__(
        self,
        benchmark_id: str,
        dataset_root: Optional[str] = None,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.benchmark_id = benchmark_id
        self.dataset_root = dataset_root
        self.max_samples = max_samples

    def _get_benchmark(self) -> Any:
        from gdb.registry import BenchmarkRegistry

        registry = BenchmarkRegistry()
        registry.discover()
        return registry.get(self.benchmark_id)

    def _load_samples(self) -> List[Dict[str, Any]]:
        bench = self._get_benchmark()

        if self.dataset_root:
            data_dir = bench.resolve_data_dir(self.dataset_root)
            return bench.load_data(
                data_dir, n=self.max_samples, dataset_root=self.dataset_root
            )

        from gdb.hf import load_from_hub

        return load_from_hub(self.benchmark_id, n=self.max_samples)

    def get_instances(self, output_path: str) -> List[Instance]:
        bench = self._get_benchmark()
        samples = self._load_samples()

        instances: List[Instance] = []
        for sample in samples:
            model_input = bench.build_model_input(sample)
            helm_input = _model_input_to_helm_input(model_input)

            gt = sample.get("ground_truth", "")
            gt_text = str(gt) if not isinstance(gt, str) else gt

            instances.append(
                Instance(
                    id=str(sample.get("sample_id", len(instances))),
                    input=helm_input,
                    references=[
                        Reference(
                            output=Output(text=gt_text),
                            tags=[CORRECT_TAG],
                        )
                    ],
                    split=TEST_SPLIT,
                    extra_data=sample,
                )
            )

        return instances

    def get_metadata(self) -> ScenarioMetadata:
        bench = self._get_benchmark()
        main_metric = bench.meta.metrics[0] if bench.meta.metrics else "accuracy"
        return ScenarioMetadata(
            name=f"gdb_{self.benchmark_id}",
            main_metric=main_metric,
            main_split="test",
        )
