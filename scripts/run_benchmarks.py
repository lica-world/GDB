#!/usr/bin/env python3
"""CLI to list and run design benchmarks (stub model, API providers, batch APIs).

Usage:
    # Stub smoke test (no API keys)
    python scripts/run_benchmarks.py --stub-model --benchmarks layout-4 layout-5 \\
        --dataset-root data/lica-benchmarks-dataset --n 5

    # API run (shipped Lica layout)
    python scripts/run_benchmarks.py --benchmarks svg-1 \\
        --provider gemini --credentials auth/google-cloud-key.json \\
        --dataset-root data/lica-benchmarks-dataset

    python scripts/run_benchmarks.py --benchmarks layout-1 \\
        --provider openai_image --model-id gpt-image-1.5 \\
        --data /path/to/custom/layout2_folder --dataset-root data/lica-benchmarks-dataset \\
        --n 200 -o outputs/baseline.json

    # Batch submit (~50% cheaper, fire-and-forget)
    python scripts/run_benchmarks.py --batch-submit --benchmarks svg-1 \\
        --provider gemini --credentials /path/to/credentials.json \\
        --dataset-root data/lica-benchmarks-dataset

    # Collect results from a previous submit
    python scripts/run_benchmarks.py --collect jobs/job_manifest.json

    # List all benchmarks
    python scripts/run_benchmarks.py --list
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

try:
    from design_benchmarks import (
        BaseBenchmark,
        BenchmarkRegistry,
        BenchmarkRunner,
        RunReport,
    )
except ModuleNotFoundError as exc:
    print(
        "Package 'lica-bench' is not installed. From the repository root run:\n"
        "  pip install -e .\n",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc

REPO_ROOT = Path(__file__).resolve().parent.parent


PROVIDER_TO_REGISTRY = {
    "gemini": "google",
    "openai": "openai",
    "openai_image": "openai_image",
    "anthropic": "anthropic",
    "hf": "hf",
    "vllm": "vllm",
    "diffusion": "diffusion",
}

DEFAULT_MODEL_IDS = {
    "gemini": "gemini-2.0-flash",
    "openai": "gpt-4o",
    "openai_image": "gpt-image-1.5",
    "anthropic": "claude-sonnet-4-20250514",
    "hf": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "vllm": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "diffusion": "black-forest-labs/FLUX.1-schnell",
}


def _make_stub_model() -> Any:
    from design_benchmarks.models.base import BaseModel, Modality, ModelOutput

    class StubModel(BaseModel):
        name = "stub"
        modality = Modality.ANY

        def predict(self, inp: Any) -> ModelOutput:
            return ModelOutput(text="", images=[])

    return StubModel()


def _build_model(args: argparse.Namespace) -> Any:
    provider = args.provider
    model_id = args.model_id or DEFAULT_MODEL_IDS[provider]
    return _build_model_from_parts(provider, model_id, args)


def _model_name(args: argparse.Namespace) -> str:
    return args.model_id or DEFAULT_MODEL_IDS.get(args.provider, "unknown")


def _build_model_from_parts(
    provider: str, model_id: str, args: argparse.Namespace
) -> Any:
    from design_benchmarks.models import load_model

    if provider == "diffusion":
        return load_model("diffusion", model_id=model_id, resolution=args.resolution)

    kwargs: Dict[str, Any] = {"model_id": model_id, "temperature": args.temperature}
    if args.credentials:
        kwargs["credentials_path"] = args.credentials
    if args.max_tokens is not None:
        kwargs["max_tokens"] = args.max_tokens
    if provider == "hf":
        kwargs["device"] = args.device
    if provider == "vllm":
        kwargs["tensor_parallel_size"] = args.tensor_parallel_size
        kwargs["top_p"] = args.top_p
        kwargs["top_k"] = args.top_k
        kwargs["repetition_penalty"] = args.repetition_penalty
        if hasattr(args, "presence_penalty") and args.presence_penalty is not None:
            kwargs["presence_penalty"] = args.presence_penalty
        if (
            hasattr(args, "limit_mm_per_prompt")
            and args.limit_mm_per_prompt is not None
        ):
            kwargs["limit_mm_per_prompt"] = {"image": args.limit_mm_per_prompt}
        if (
            hasattr(args, "max_num_batched_tokens")
            and args.max_num_batched_tokens is not None
        ):
            kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
        if hasattr(args, "no_thinking") and args.no_thinking:
            kwargs["enable_thinking"] = False

    return load_model(PROVIDER_TO_REGISTRY[provider], **kwargs)


def _parse_model_spec(spec: str) -> tuple[str, str, str]:
    """
    Parse model spec format:
      - provider:model_id
      - alias=provider:model_id
    """
    alias = ""
    body = spec.strip()
    if "=" in body:
        alias, body = body.split("=", 1)
        alias = alias.strip()

    if ":" not in body:
        raise ValueError(
            f"Invalid --multi-models spec {spec!r}. "
            "Use provider:model_id or alias=provider:model_id."
        )
    provider, model_id = body.split(":", 1)
    provider = provider.strip()
    model_id = model_id.strip()

    if provider not in PROVIDER_TO_REGISTRY:
        raise ValueError(
            f"Unknown provider {provider!r} in spec {spec!r}. "
            f"Choose from: {', '.join(sorted(PROVIDER_TO_REGISTRY))}"
        )

    name = alias or f"{provider}:{model_id}"
    return name, provider, model_id


def _build_models_from_specs(args: argparse.Namespace) -> Dict[str, Any]:
    models: Dict[str, Any] = {}
    for spec in args.multi_models:
        name, provider, model_id = _parse_model_spec(spec)
        models[name] = _build_model_from_parts(provider, model_id, args)
    return models


def _benchmark_pipeline_ready(bench: BaseBenchmark) -> bool:
    """True if the task overrides the default stubs (see tasks' ``pipeline_implemented``)."""
    cls = type(bench)
    if getattr(cls, "pipeline_implemented", True) is False:
        return False
    return cls.load_data is not BaseBenchmark.load_data


def cmd_list(registry: BenchmarkRegistry) -> None:
    runnable = {b.meta.id for b in registry.list() if _benchmark_pipeline_ready(b)}
    benchmarks = sorted(registry.list(), key=lambda b: b.meta.id)

    print(f"{'ID':<30s} {'NAME':<40s} {'PIPELINE':>8s}")
    print("-" * 80)
    for b in benchmarks:
        s = "ready" if b.meta.id in runnable else "-"
        print(f"{b.meta.id:<30s} {b.meta.name:<40s} {s:>8s}")
    print(
        f"\n{len(runnable)} / {len(benchmarks)} benchmarks have the structured pipeline."
    )


def cmd_run(
    registry: BenchmarkRegistry,
    benchmark_ids: List[str],
    models: Dict[str, Any],
    data_override: str | None,
    n: int | None,
    output_path: str | None,
    batch_size: int | None = None,
    no_log: bool = False,
    save_images: bool = False,
    images_dir: str | None = None,
    input_modality: Any = None,
    dataset_root: str | None = None,
) -> bool:
    runner = BenchmarkRunner(registry)
    save_dir = None
    if save_images:
        save_dir = (
            Path(images_dir)
            if images_dir
            else (REPO_ROOT / "outputs" / "generated-images")
        )
        save_dir.mkdir(parents=True, exist_ok=True)
    all_ok = True

    combined = RunReport()

    for bid in benchmark_ids:
        bench = registry.get(bid)
        print(f"\n[{bid}] {bench.meta.name}")
        try:
            if data_override:
                data_path = data_override
            else:
                data_path = str(bench.resolve_data_dir(dataset_root))
        except FileNotFoundError as exc:
            print(f"  FAILED: {exc}")
            all_ok = False
            continue

        print(f"  data: {data_path}")
        t0 = time.time()
        try:
            report = runner.run(
                benchmark_ids=[bid],
                models=models,
                data_dir=data_override,
                dataset_root=dataset_root,
                n=n,
                batch_size=batch_size,
                prediction_save_dir=save_dir,
                input_modality=input_modality,
            )
            for model_name, result in sorted(report.results[bid].items()):
                scores = [f"{k}={v:.4f}" for k, v in sorted(result.scores.items())]
                print(
                    f"  {model_name}: {', '.join(scores)}  "
                    f"(n={result.count}, ok={result.success_count}, "
                    f"fail={result.failure_count}, fail_rate={result.failure_rate:.1%}, "
                    f"{time.time() - t0:.1f}s)"
                )
            combined.results[bid] = report.results[bid]
        except Exception as e:
            print(f"  FAILED: {e}")
            all_ok = False

    if combined.results:
        if output_path:
            combined.save(output_path)
            print(f"Saved to {output_path}")
        else:
            out_dir = REPO_ROOT / "outputs"
            out_dir.mkdir(parents=True, exist_ok=True)
            for bid in combined.results:
                single = RunReport(results={bid: combined.results[bid]})
                single.save(str(out_dir / f"{bid}.csv"))
            print(f"Saved to {out_dir}/")

    if not no_log and len(runner.tracker) > 0:
        log_dir = REPO_ROOT / "outputs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "tracker.jsonl"
        runner.tracker.save(str(log_path))
        print(f"Tracker log: {log_path}")

    if save_dir is not None:
        print(f"Generated images: {save_dir}")

    return all_ok


def cmd_submit(
    registry: BenchmarkRegistry,
    benchmark_id: str,
    args: argparse.Namespace,
) -> bool:
    from design_benchmarks.inference import make_batch_runner, save_job_manifest

    model_id = _model_name(args)
    runner = BenchmarkRunner(registry)

    batch_kwargs: Dict[str, Any] = {
        "model_id": model_id,
        "temperature": args.temperature,
        "poll_interval": args.poll_interval,
        "on_status": lambda msg: print(f"  {msg}"),
    }
    if args.credentials:
        batch_kwargs["credentials_path"] = args.credentials
    if args.bucket:
        batch_kwargs["bucket"] = args.bucket

    batch_runner = make_batch_runner(args.provider, **batch_kwargs)

    bench = registry.get(benchmark_id)
    data_display = (
        args.data
        if args.data
        else str(bench.resolve_data_dir(args.dataset_root))
    )

    print(f"\n[{benchmark_id}] {bench.meta.name}")
    print(f"  data: {data_display}")
    print(f"  provider: {args.provider} / {model_id}")

    manifest_data = runner.submit(
        benchmark_id,
        batch_runner,
        data_dir=args.data,
        dataset_root=args.dataset_root,
        n=args.n,
    )

    extra = {"benchmark_id": benchmark_id}
    if args.provider == "gemini" and hasattr(batch_runner, "_last_submit_meta"):
        extra["job_prefix"] = batch_runner._last_submit_meta["job_prefix"]

    jobs_dir = REPO_ROOT / "jobs"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = save_job_manifest(
        jobs_dir / f"job_{ts}_{args.provider}.json",
        provider=args.provider,
        batch_id=manifest_data["batch_id"],
        model_id=model_id,
        custom_ids=manifest_data["custom_ids"],
        ground_truths=manifest_data["ground_truths"],
        extra=extra,
    )
    print(f"\n  Job submitted: {manifest_data['batch_id']}")
    print(f"  Manifest: {manifest_path}")
    print(f"\n  To collect: python scripts/run_benchmarks.py --collect {manifest_path}")
    return True


def cmd_collect(args: argparse.Namespace) -> bool:
    from design_benchmarks.inference import load_job_manifest, make_batch_runner

    manifest = load_job_manifest(args.collect)
    provider = manifest["provider"]
    model_id = manifest["model_id"]
    benchmark_id = manifest.get("benchmark_id") or manifest.get("extra", {}).get(
        "benchmark_id"
    )

    print(f"Collecting {provider} batch: {manifest['batch_id']}")
    print(f"  model: {model_id}, samples: {len(manifest['custom_ids'])}")

    batch_kwargs: Dict[str, Any] = {
        "model_id": model_id,
        "poll_interval": args.poll_interval,
        "on_status": lambda msg: print(f"  {msg}"),
    }
    if args.credentials:
        batch_kwargs["credentials_path"] = args.credentials
    if args.bucket:
        batch_kwargs["bucket"] = args.bucket

    batch_runner = make_batch_runner(provider, **batch_kwargs)

    collect_kwargs: Dict[str, Any] = {}
    _jp = manifest.get("job_prefix")
    if provider == "gemini" and _jp:
        collect_kwargs["job_prefix"] = _jp

    if benchmark_id:
        registry = BenchmarkRegistry()
        registry.discover()
        runner = BenchmarkRunner(registry)
        report = runner.collect(
            benchmark_id,
            batch_runner,
            batch_id=manifest["batch_id"],
            custom_ids=manifest["custom_ids"],
            ground_truths=manifest["ground_truths"],
            model_id=model_id,
            **collect_kwargs,
        )
        result = report.results[benchmark_id][model_id]
        scores = [f"{k}={v:.4f}" for k, v in sorted(result.scores.items())]
        print(
            f"\n  [{benchmark_id}] {', '.join(scores)}  "
            f"(n={result.count}, ok={result.success_count}, "
            f"fail={result.failure_count}, fail_rate={result.failure_rate:.1%})"
        )

        if args.output:
            report.save(args.output)
            print(f"  Saved to {args.output}")
    else:
        results = batch_runner.collect(
            batch_id=manifest["batch_id"],
            custom_ids=manifest["custom_ids"],
            **collect_kwargs,
        )
        ok = sum(1 for r in results.values() if r.success)
        print(f"\n  {ok}/{len(results)} succeeded")

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified benchmark runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true")
    mode.add_argument(
        "--collect",
        metavar="MANIFEST",
        help="Collect results from a previous --batch-submit",
    )
    mode.add_argument(
        "--batch-submit",
        action="store_true",
        help="Submit to provider batch API (~50%% cheaper) and exit; collect later with --collect",
    )

    parser.add_argument("--benchmarks", nargs="+", metavar="ID")
    parser.add_argument(
        "--stub-model",
        action="store_true",
        help="Run with built-in stub model (no API keys).",
    )
    parser.add_argument("--provider", choices=list(PROVIDER_TO_REGISTRY.keys()))
    parser.add_argument("--model-id", default=None)
    parser.add_argument(
        "--multi-models",
        nargs="+",
        metavar="SPEC",
        default=None,
        help=(
            "Run multiple models in one pass. "
            "Format: provider:model_id or alias=provider:model_id "
            "(e.g. openai_image:gpt-image-1.5 "
            "nano=gemini:gemini-3.1-flash-image-preview)"
        ),
    )
    parser.add_argument("--credentials", default=None)

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max output tokens (default: model-specific)",
    )
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling")
    parser.add_argument("--top-k", type=int, default=-1, help="Top-k sampling (-1=off)")
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Repetition penalty (vLLM)",
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=0.0,
        help="Presence penalty (vLLM, recommended 1.5 for Qwen3-VL)",
    )

    parser.add_argument(
        "--device", default="auto", help="HF device (auto/cpu/cuda/mps)"
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1, help="vLLM TP size"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Diffusion output resolution (pixels)",
    )
    parser.add_argument(
        "--limit-mm-per-prompt",
        type=int,
        default=None,
        help="Max images per prompt for vLLM VLMs (default: 5)",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="vLLM encoder cache / chunked prefill budget (default: 16384)",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable thinking mode for Qwen3.5 models",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for predict_batch() (vLLM/diffusion)",
    )

    parser.add_argument(
        "--bucket",
        default=None,
        help="GCS bucket for batch image uploads (or set DESIGN_BENCHMARKS_GCS_BUCKET)",
    )
    parser.add_argument("--poll-interval", type=int, default=30)

    parser.add_argument(
        "--input-modality",
        choices=["text", "image", "both"],
        default=None,
        help=(
            "Override input modality for template benchmarks: "
            "text=layout JSON only, image=rendered image only, "
            "both=layout JSON + image (default: model's native modality)"
        ),
    )

    parser.add_argument(
        "--data",
        default=None,
        help="Task data directory (default: under --dataset-root/benchmarks/ per benchmark)",
    )
    parser.add_argument(
        "--dataset-root",
        required=False,
        default=None,
        help="Lica bundle root (lica-data/ + benchmarks/). Required for runs.",
    )
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument(
        "--output", "-o", default=None, help="Save report (.json or .csv)"
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Persist image predictions under outputs/generated-images/",
    )
    parser.add_argument(
        "--images-dir", default=None, help="Custom directory for --save-images"
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="JSON",
        help="Load defaults from a JSON config file (CLI args override)",
    )
    parser.add_argument(
        "--no-log", action="store_true", help="Disable per-sample tracker JSONL output"
    )

    _pre, _ = parser.parse_known_args()
    if _pre.config:
        with open(_pre.config, "r", encoding="utf-8") as _cf:
            _config_data = json.load(_cf)
        _models_map = _config_data.pop("models", None)
        parser.set_defaults(**_config_data)
        if _models_map:
            _pre2, _ = parser.parse_known_args()
            _mid = _pre2.model_id
            if _mid and _mid in _models_map:
                _model_cfg = dict(_models_map[_mid])
                _modality_overrides = _model_cfg.pop("modality_overrides", None)
                parser.set_defaults(**_model_cfg)
                if _modality_overrides and _pre2.input_modality in _modality_overrides:
                    parser.set_defaults(**_modality_overrides[_pre2.input_modality])

    args = parser.parse_args()

    if args.stub_model:
        if args.provider or args.multi_models:
            parser.error(
                "--stub-model cannot be used with --provider or --multi-models"
            )

    if args.list:
        registry = BenchmarkRegistry()
        registry.discover()
        cmd_list(registry)
        return

    if args.collect:
        sys.exit(0 if cmd_collect(args) else 1)

    if args.batch_submit:
        from design_benchmarks.inference import BATCH_PROVIDERS

        if not args.provider:
            parser.error("--provider required")
        if args.provider not in BATCH_PROVIDERS:
            parser.error(
                f"--batch-submit requires {', '.join(sorted(BATCH_PROVIDERS))}"
            )
        if not args.benchmarks or len(args.benchmarks) != 1:
            parser.error("--batch-submit requires exactly one --benchmarks ID")
        if not args.dataset_root:
            parser.error("--dataset-root required")
        registry = BenchmarkRegistry()
        registry.discover()
        if not args.data:
            try:
                registry.get(args.benchmarks[0]).resolve_data_dir(args.dataset_root)
            except FileNotFoundError as exc:
                parser.error(str(exc))
        sys.exit(0 if cmd_submit(registry, args.benchmarks[0], args) else 1)

    models: Dict[str, Any] = {}

    if args.stub_model:
        models = {"stub": _make_stub_model()}
    elif args.multi_models:
        try:
            models = _build_models_from_specs(args)
        except ValueError as exc:
            parser.error(str(exc))
    elif args.provider:
        models = {_model_name(args): _build_model(args)}
    else:
        parser.error("--provider, --multi-models, or --stub-model required")

    registry = BenchmarkRegistry()
    registry.discover()

    if not args.benchmarks:
        parser.error("--benchmarks required (or --list)")
    if not args.dataset_root:
        parser.error("--dataset-root required")

    _input_modality = None
    if args.input_modality:
        from design_benchmarks.models.base import Modality

        _modality_map = {
            "text": Modality.TEXT,
            "image": Modality.IMAGE,
            "both": Modality.TEXT_AND_IMAGE,
        }
        _input_modality = _modality_map[args.input_modality]

    sys.exit(
        0
        if cmd_run(
            registry,
            args.benchmarks,
            models,
            args.data,
            args.n,
            args.output,
            batch_size=args.batch_size,
            no_log=args.no_log,
            save_images=args.save_images,
            images_dir=args.images_dir,
            input_modality=_input_modality,
            dataset_root=args.dataset_root,
        )
        else 1
    )


if __name__ == "__main__":
    main()
