"""Benchmark runner that orchestrates evaluation."""

import io
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .base import BaseBenchmark
from .evaluation.reporting import BenchmarkResult, RunReport
from .evaluation.tracker import EvaluationTracker
from .registry import BenchmarkRegistry

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Run ``load_data`` → ``build_model_input`` → ``predict`` → ``evaluate`` per benchmark.

    Example::

        from design_benchmarks.models import load_model

        runner = BenchmarkRunner(registry)
        report = runner.run(
            benchmark_ids=["typography-1"],
            models={"gemini": load_model("google", model_id="gemini-2.0-flash")},
            dataset_root="data/lica-benchmarks-dataset",
        )
        print(report.summary())
    """

    def __init__(self, registry: BenchmarkRegistry) -> None:
        self.registry = registry
        self.tracker = EvaluationTracker()

    @staticmethod
    def _safe_fs_name(value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))
        cleaned = cleaned.strip("._")
        return cleaned or "item"

    @staticmethod
    def _batch_output_text(output: Any) -> str:
        """String form of a batch item's raw model output (for tracker JSONL)."""
        return getattr(output, "text", str(output))

    @staticmethod
    def _save_pil_to(path: Path, img: Any) -> str:
        try:
            img.save(path)
        except Exception:
            img.convert("RGB").save(path)
        return str(path)

    @staticmethod
    def _prediction_to_pil(prediction: Any, Image: Any) -> Optional[Any]:
        if isinstance(prediction, Image.Image):
            return prediction
        if isinstance(prediction, (bytes, bytearray)):
            try:
                return Image.open(io.BytesIO(bytes(prediction))).convert("RGB")
            except Exception:
                return None
        try:
            import numpy as np
        except Exception:
            return None
        if not isinstance(prediction, np.ndarray):
            return None
        arr = prediction
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            return Image.fromarray(arr, mode="L")
        if arr.ndim == 3:
            if arr.shape[2] == 4:
                return Image.fromarray(arr, mode="RGBA")
            return Image.fromarray(arr[:, :, :3], mode="RGB")
        return None

    def _maybe_save_prediction_image(
        self,
        prediction: Any,
        *,
        root_dir: Optional[Path],
        benchmark_id: str,
        model_name: str,
        sample_id: str,
    ) -> Any:
        """Persist image-like predictions to disk and return file path."""
        if root_dir is None:
            return prediction
        try:
            from PIL import Image
        except ImportError:
            return prediction

        out_dir = (
            root_dir
            / self._safe_fs_name(benchmark_id)
            / self._safe_fs_name(model_name)
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = self._safe_fs_name(sample_id)
        png_path = out_dir / f"{stem}.png"

        pil = self._prediction_to_pil(prediction, Image)
        if pil is not None:
            return self._save_pil_to(png_path, pil)

        if isinstance(prediction, (bytes, bytearray)):
            bin_path = out_dir / f"{stem}.bin"
            bin_path.write_bytes(bytes(prediction))
            return str(bin_path)

        return prediction

    def _parse_and_store_prediction(
        self,
        bench: Any,
        output: Any,
        *,
        benchmark_id: str,
        sample_id: str,
        save_root: Optional[Path],
        model_name: str,
    ) -> Tuple[Any, Optional[str]]:
        try:
            pred = bench.parse_model_output(output)
        except Exception as exc:
            logger.warning(
                "[%s] parse failed for sample %s: %s",
                benchmark_id, sample_id, exc,
            )
            return "", f"parse_failed: {exc}"
        stored = self._maybe_save_prediction_image(
            pred,
            root_dir=save_root,
            benchmark_id=benchmark_id,
            model_name=model_name,
            sample_id=sample_id,
        )
        return stored, None

    def _persist_incremental_outputs(
        self,
        *,
        save_root: Optional[Path],
        benchmark_id: str,
        model_name: str,
        sample_ids: List[str],
        predictions: List[Any],
        ground_truths: List[Any],
        report: RunReport,
    ) -> None:
        out_root = save_root or Path("outputs")
        out_root.mkdir(parents=True, exist_ok=True)
        self.tracker.save(str(out_root / "tracker.jsonl"))
        report.save(str(out_root / "report_incremental.json"))

        preds_path = out_root / "predictions" / f"{benchmark_id}_{model_name}.jsonl"
        preds_path.parent.mkdir(parents=True, exist_ok=True)
        with open(preds_path, "w", encoding="utf-8") as pf:
            for sid, pred, gt in zip(sample_ids, predictions, ground_truths):
                pf.write(
                    json.dumps(
                        {
                            "sample_id": sid,
                            "prediction": pred,
                            "ground_truth": gt,
                        },
                        ensure_ascii=False,
                        default=str,
                    )
                    + "\n"
                )
        logger.info(
            "Saved %d full predictions to %s", len(predictions), preds_path
        )

    def run(
        self,
        benchmark_ids: List[str],
        models: Dict[str, Any],
        *,
        data_dir: Optional[Union[str, Path]] = None,
        dataset_root: Union[str, Path],
        n: Optional[int] = None,
        batch_size: Optional[int] = None,
        prediction_save_dir: Optional[Union[str, Path]] = None,
        input_modality: Optional[Any] = None,
    ) -> RunReport:
        """Run benchmarks and return a report.

        Parameters
        ----------
        benchmark_ids : list of str
            Benchmark IDs to evaluate (e.g. ``["svg-1", "category-1"]``).
        models : dict
            Mapping of model name → ``BaseModel`` instance or callable.
            ``BaseModel`` instances get the full structured pipeline
            (build_model_input → predict → parse_model_output).
            Callables receive the ``ModelInput`` directly and must
            return the final prediction value.
        data_dir : str or Path, optional
            Use this directory for every benchmark in the run. If ``None``,
            each benchmark uses ``<dataset_root>/benchmarks/<data_subpath or domain>``.
        dataset_root : str or Path
            Top-level dataset directory.  Paths inside data files
            (CSV image_path, JSON data_root, etc.) are resolved against
            this root.
        n : int, optional
            Limit samples per benchmark (passed to ``load_data``).
        batch_size : int, optional
            If set, call ``model.predict_batch()`` with chunks of this
            size instead of sequential ``predict()`` calls.  Useful for
            models with native batching (e.g. vLLM).
        prediction_save_dir : str or Path, optional
            If set, persist image-like predictions under this directory
            (per benchmark / model / sample).
        input_modality : Modality, optional
            When set, passed to ``build_model_input(..., modality=...)`` for
            every sample.  If omitted, each model's ``modality`` attribute
            is used when present.
        """
        from .models.base import BaseModel as _BaseModel

        modality_label = None
        if input_modality is not None:
            modality_label = (
                input_modality.value
                if hasattr(input_modality, "value")
                else str(input_modality)
            )

        report = RunReport(
            metadata={
                "benchmarks": benchmark_ids,
                "models": list(models.keys()),
                "input_modality": modality_label,
            }
        )
        save_root = Path(prediction_save_dir).resolve() if prediction_save_dir else None
        if save_root is not None:
            save_root.mkdir(parents=True, exist_ok=True)

        override_dir = Path(data_dir).resolve() if data_dir is not None else None

        for bid in benchmark_ids:
            bench = self.registry.get(bid)
            report.results[bid] = {}

            resolved_dir = override_dir or bench.resolve_data_dir(dataset_root)
            samples = bench.load_data(
                resolved_dir, n=n, dataset_root=dataset_root,
            )
            logger.info(
                "Loaded %d samples for %s from %s",
                len(samples), bid, resolved_dir,
            )

            for model_name, model_or_fn in models.items():
                ground_truths = [s["ground_truth"] for s in samples]
                sample_ids = [s.get("sample_id", str(i)) for i, s in enumerate(samples)]
                modality = input_modality or getattr(model_or_fn, "modality", None)
                model_inputs = [
                    bench.build_model_input(s, modality=modality)
                    for s in samples
                ]

                t0 = time.time()
                predictions: List[Any] = []
                raw_outputs: List[str] = []
                errors: List[Optional[str]] = []

                if batch_size and isinstance(model_or_fn, _BaseModel):
                    for chunk_start in range(0, len(model_inputs), batch_size):
                        chunk_end = min(chunk_start + batch_size, len(model_inputs))
                        chunk = model_inputs[chunk_start:chunk_end]
                        try:
                            outputs = model_or_fn.predict_batch(chunk)
                        except Exception as exc:
                            logger.error(
                                "[%s] batch predict failed (samples %d–%d): %s",
                                bid, chunk_start, chunk_end - 1, exc,
                            )
                            for _ in chunk:
                                predictions.append("")
                                raw_outputs.append("")
                                errors.append(f"batch_predict_failed: {exc}")
                            continue
                        for offset, output in enumerate(outputs):
                            sid = sample_ids[chunk_start + offset]
                            pred, parse_error = self._parse_and_store_prediction(
                                bench,
                                output,
                                benchmark_id=bid,
                                sample_id=sid,
                                save_root=save_root,
                                model_name=model_name,
                            )
                            predictions.append(pred)
                            raw_outputs.append(self._batch_output_text(output))
                            errors.append(parse_error)
                    self.tracker.log_batch(
                        benchmark_id=bid,
                        model=model_name,
                        sample_ids=sample_ids,
                        predictions=predictions,
                        ground_truths=ground_truths,
                        model_outputs=raw_outputs,
                        errors=errors,
                        elapsed_s=time.time() - t0,
                    )
                else:
                    for i, model_input in enumerate(model_inputs):
                        t_sample = time.time()
                        try:
                            if isinstance(model_or_fn, _BaseModel):
                                output = model_or_fn.predict(model_input)
                                pred, parse_error = self._parse_and_store_prediction(
                                    bench,
                                    output,
                                    benchmark_id=bid,
                                    sample_id=sample_ids[i],
                                    save_root=save_root,
                                    model_name=model_name,
                                )
                                raw = getattr(output, "text", None)
                                err = parse_error
                            else:
                                output = None
                                pred = model_or_fn(model_input)
                                pred = self._maybe_save_prediction_image(
                                    pred,
                                    root_dir=save_root,
                                    benchmark_id=bid,
                                    model_name=model_name,
                                    sample_id=sample_ids[i],
                                )
                                raw = None
                                err = None
                            predictions.append(pred)
                            errors.append(err)
                            self.tracker.log(
                                benchmark_id=bid,
                                model=model_name,
                                sample_id=sample_ids[i],
                                prediction=pred,
                                ground_truth=ground_truths[i],
                                model_output=raw,
                                error=err,
                                elapsed_s=time.time() - t_sample,
                            )
                        except Exception as exc:
                            predictions.append("")
                            errors.append(str(exc))
                            self.tracker.log(
                                benchmark_id=bid,
                                model=model_name,
                                sample_id=sample_ids[i],
                                error=str(exc),
                                elapsed_s=time.time() - t_sample,
                            )

                scores = bench.evaluate(predictions, ground_truths)
                failure_count = sum(1 for e in errors if e)
                success_count = max(len(samples) - failure_count, 0)
                report.results[bid][model_name] = BenchmarkResult(
                    benchmark_id=bid,
                    model=model_name,
                    scores=scores,
                    count=len(samples),
                    success_count=success_count,
                    failure_count=failure_count,
                )
                logger.info(
                    "[%s] %s: %s (n=%d, ok=%d, fail=%d)",
                    bid, model_name,
                    ", ".join(f"{k}={v:.4f}" for k, v in sorted(scores.items())),
                    len(samples),
                    success_count,
                    failure_count,
                )

                self._persist_incremental_outputs(
                    save_root=save_root,
                    benchmark_id=bid,
                    model_name=model_name,
                    sample_ids=sample_ids,
                    predictions=predictions,
                    ground_truths=ground_truths,
                    report=report,
                )

        return report

    def submit(
        self,
        benchmark_id: str,
        batch_runner: Any,
        *,
        data_dir: Optional[Union[str, Path]] = None,
        dataset_root: Union[str, Path],
        n: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Submit a benchmark to a batch runner (fire-and-forget).

        Returns a manifest dict with ``batch_id``, ``custom_ids``,
        ``ground_truths``, and ``benchmark_id`` — everything needed to
        call ``collect()`` later.
        """
        from .inference import BatchRequest

        bench = self.registry.get(benchmark_id)
        resolved_dir = (
            Path(data_dir).resolve()
            if data_dir is not None
            else bench.resolve_data_dir(dataset_root)
        )
        samples = bench.load_data(resolved_dir, n=n, dataset_root=dataset_root)
        requests = [
            BatchRequest(
                custom_id=s["sample_id"],
                model_input=bench.build_model_input(s),
            )
            for s in samples
        ]
        ground_truths = {s["sample_id"]: s["ground_truth"] for s in samples}

        batch_id = batch_runner.submit(requests)
        logger.info(
            "Submitted batch %s for %s (%d samples)",
            batch_id, benchmark_id, len(requests),
        )

        return {
            "batch_id": batch_id,
            "benchmark_id": benchmark_id,
            "custom_ids": [r.custom_id for r in requests],
            "ground_truths": ground_truths,
        }

    def collect(
        self,
        benchmark_id: str,
        batch_runner: Any,
        *,
        batch_id: str,
        custom_ids: List[str],
        ground_truths: Dict[str, Any],
        **collect_kwargs: Any,
    ) -> RunReport:
        """Collect batch results and evaluate.

        Pass the fields from the manifest returned by ``submit()``
        (or loaded from a saved job manifest JSON).
        """
        from .models.base import ModelOutput

        bench = self.registry.get(benchmark_id)
        model_name = collect_kwargs.pop("model_id", benchmark_id)
        results = batch_runner.collect(
            batch_id=batch_id, custom_ids=custom_ids, **collect_kwargs,
        )

        predictions, gt_list = [], []
        for cid in custom_ids:
            r = results.get(cid)
            if r and r.success:
                raw_text = r.text
                pred = bench.parse_model_output(ModelOutput(text=raw_text))
                self.tracker.log(
                    benchmark_id=benchmark_id, model=model_name,
                    sample_id=cid, prediction=pred,
                    ground_truth=ground_truths[cid],
                    model_output=raw_text,
                )
            else:
                pred = ""
                err = getattr(r, "error", "missing") if r else "missing"
                logger.warning(
                    "[%s] sample %s failed: %s",
                    benchmark_id, cid, err,
                )
                self.tracker.log(
                    benchmark_id=benchmark_id, model=model_name,
                    sample_id=cid, error=str(err),
                )
            predictions.append(pred)
            gt_list.append(ground_truths[cid])

        scores = bench.evaluate(predictions, gt_list)
        logger.info(
            "Collected %s: %s (n=%d)",
            benchmark_id,
            ", ".join(f"{k}={v:.4f}" for k, v in sorted(scores.items())),
            len(predictions),
        )

        report = RunReport()
        failure_count = sum(
            1 for cid in custom_ids
            if not (results.get(cid) and results.get(cid).success)
        )
        success_count = max(len(predictions) - failure_count, 0)
        report.results[benchmark_id] = {
            model_name: BenchmarkResult(
                benchmark_id=benchmark_id,
                model=model_name,
                scores=scores,
                count=len(predictions),
                success_count=success_count,
                failure_count=failure_count,
            )
        }
        return report

    def run_from_csv(
        self,
        csv_path: str,
        task_column: str = "task",
        expected_column: str = "expected_output",
    ) -> RunReport:
        """Evaluate benchmarks from a pre-computed CSV of model outputs."""
        from .evaluation.reporting import load_from_csv

        task_to_benchmark: Dict[str, BaseBenchmark] = {}
        for bid, bench in self.registry._benchmarks.items():
            task_to_benchmark[bench.meta.name] = bench
            task_to_benchmark[bid] = bench

        return load_from_csv(
            csv_path, task_to_benchmark,
            task_column=task_column, expected_column=expected_column,
        )
