"""Base classes for the benchmark framework."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union


class TaskType(Enum):
    """What kind of task the benchmark evaluates.

    - UNDERSTANDING: model receives an input and answers a question about it
      or edits an existing artifact (e.g. "what font is this?", "fix this SVG").
      Ground truth is typically a string, label, or corrected artifact.
    - GENERATION: model produces a new artifact from a specification
      (e.g. "generate a layout for a social media ad").  Evaluation uses
      quality metrics (FID, human eval, etc.).
    """

    UNDERSTANDING = "understanding"
    GENERATION = "generation"


@dataclass
class BenchmarkMeta:
    """Metadata describing a single benchmark task.

    Required: ``id``, ``name``, ``task_type``, ``domain``, ``description``.
    ``data_subpath`` is the path under ``<dataset_root>/benchmarks/`` for this
    task's inputs; if empty, ``domain`` is used (one shared folder per domain).
    """

    id: str
    name: str
    task_type: TaskType
    domain: str
    description: str
    data_subpath: str = ""
    input_spec: str = ""
    output_spec: str = ""
    metrics: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


# Populated by @benchmark at import time; registry.discover() registers these.
_REGISTERED_BENCHMARKS: List["BaseBenchmark"] = []


def benchmark(cls: Type["BaseBenchmark"]) -> Type["BaseBenchmark"]:
    """Decorator that auto-registers a benchmark class.

    Usage::

        @benchmark
        class MyBenchmark(BaseBenchmark):
            meta = BenchmarkMeta(id="svg-15", ...)

    The instance is created and registered automatically.
    """
    _REGISTERED_BENCHMARKS.append(cls())
    return cls


class BaseBenchmark:
    """Abstract base for all benchmark implementations.

    Subclasses set ``meta`` and implement the pipeline methods below.

    Set ``pipeline_implemented = False`` on placeholder tasks so ``--list`` marks
    them as not ready until implementations are complete; set
    ``pipeline_implemented = True`` (or remove the attribute) when ready.
    The four methods map to the three-phase benchmark pipeline:

    **Phase 1 — Data preparation:**

    - ``load_data(data_dir, *, n=..., dataset_root=...)`` — discover samples
      on disk, extract ground truth.  Returns a list of dicts, each with at
      least ``sample_id`` (str) and ``ground_truth`` (Any), plus any
      task-specific fields (image paths, metadata, etc.).
    - ``build_model_input(sample)`` — convert one sample dict into a
      ``ModelInput`` ready for ``model.predict()``.  This is where the
      prompt template lives.

    **Phase 2 — Inference** (handled by ``BenchmarkRunner``):

    The runner calls ``model.predict(model_input)`` for each sample.

    **Phase 3 — Evaluation:**

    - ``parse_model_output(output)`` — extract the prediction from a raw
      ``ModelOutput``.  Default: ``output.text.strip()``.  Override for
      tasks that need custom parsing (e.g. extracting ratios from text,
      returning images).
    - ``evaluate(predictions, ground_truth)`` — compute metric scores.

    ``build_model_input`` receives the model's ``Modality`` so the same
    benchmark can adapt its input to different model types:

    - **Text + Image** (VLMs like GPT-4o, Gemini, Claude):
      ``ModelInput(text=prompt, images=[path])``
    - **Text only** (LLMs without vision):
      ``ModelInput(text=prompt_with_description)``
    - **Image only** (pure vision models):
      ``ModelInput(images=[path])``
    - **Image generation** (diffusion / edit models):
      ``ModelInput(text=prompt, images=[input_img], metadata={"mask": ...})``

    Minimal example::

        @benchmark
        class MyBenchmark(BaseBenchmark):
            meta = BenchmarkMeta(id="my-1", ...)

            def load_data(self, data_dir, *, n=None, dataset_root: Union[str, Path]):
                samples = []
                for img in sorted(Path(data_dir).glob("*.png")):
                    samples.append({
                        "sample_id": img.stem,
                        "ground_truth": "expected_label",
                        "image_path": str(img),
                    })
                return samples[:n] if n else samples

            def build_model_input(self, sample, *, modality=None):
                from design_benchmarks.models.base import ModelInput, Modality
                if modality == Modality.TEXT:
                    # Text-only fallback for non-vision models
                    return ModelInput(text="Describe the image at: " + sample["image_path"])
                return ModelInput(text="Classify this image.", images=[sample["image_path"]])

            def evaluate(self, predictions, ground_truth):
                correct = sum(1 for p, g in zip(predictions, ground_truth) if str(p) == str(g))
                return {"accuracy": correct / len(predictions) if predictions else 0.0}
    """

    meta: BenchmarkMeta

    # ------------------------------------------------------------------
    # Phase 1: Data loading & prompt construction
    # ------------------------------------------------------------------

    def load_data(
        self,
        data_dir: Union[str, Path],
        *,
        n: Optional[int] = None,
        dataset_root: Union[str, Path],
    ) -> List[Dict[str, Any]]:
        """Load evaluation samples from a data directory.

        Returns a list of dicts, each containing at minimum:

        - ``sample_id`` (str) — unique identifier for the sample
        - ``ground_truth`` (Any) — expected value for evaluation

        Additional task-specific keys (``image_path``, ``width``, etc.)
        are passed through to ``build_model_input``.

        Parameters
        ----------
        data_dir : str or Path
            Directory containing the task-specific benchmark data
            (e.g. ``benchmarks/typography/FontFamilyClassification``).
        n : int, optional
            Limit the number of samples returned (for quick dev runs).
        dataset_root : str or Path
            Top-level dataset directory (e.g. ``data/lica-benchmarks-dataset``).
            Paths stored in data files (CSV ``image_path``, JSON ``data_root``,
            etc.) are resolved as absolute paths against this root.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.load_data() is not implemented. "
            "Override it to load samples from a data directory."
        )

    def resolve_data_dir(
        self,
        dataset_root: Union[str, Path],
        *,
        benchmarks_subdir: str = "benchmarks",
    ) -> Path:
        rel = self.meta.data_subpath or self.meta.domain
        root = Path(dataset_root).resolve()
        path = root / benchmarks_subdir / rel
        if not path.is_dir():
            raise FileNotFoundError(f"Missing benchmark data directory: {path}")
        return path

    def build_model_input(
        self,
        sample: Dict[str, Any],
        *,
        modality: Optional[Any] = None,
    ) -> Any:
        """Convert a loaded sample dict into a ``ModelInput`` for inference.

        This is where task-specific prompt templates and image handling
        are defined.  The returned ``ModelInput`` is passed directly to
        ``model.predict()``.

        The ``modality`` parameter (a ``Modality`` enum value from
        ``design_benchmarks.models.base``) tells the benchmark what kind
        of input the model accepts, so the same benchmark can build
        different inputs for different model types — e.g. sending an
        image to a VLM but a text description to a text-only LLM.

        Benchmarks may ignore ``modality`` if they only target one input
        mode.

        Parameters
        ----------
        sample : dict
            One element from the list returned by ``load_data()``.
        modality : Modality, optional
            The model's input modality.  ``None`` means unknown — the
            benchmark should fall back to its default input mode.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.build_model_input() is not implemented. "
            "Override it to construct a ModelInput from a sample dict."
        )

    def parse_model_output(self, output: Any) -> Any:
        """Extract the prediction value from a raw ``ModelOutput``.

        Default implementation returns ``output.text.strip()``.  Override
        for tasks that need custom parsing — e.g. extracting aspect ratios
        from free-form text, returning generated images, or parsing JSON.

        Parameters
        ----------
        output : ModelOutput
            The return value of ``model.predict()``.
        """
        return output.text.strip()

    # ------------------------------------------------------------------
    # Phase 3: Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self, predictions: List[Any], ground_truth: List[Any]
    ) -> Dict[str, float]:
        """Compute metrics for this benchmark.

        Returns a dict mapping metric names to their scalar scores.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.meta.id}: {self.meta.name}>"
