"""GDB command-line interface.

Installed as the ``gdb`` console script and also exposed as ``python -m gdb``.

Subcommands
-----------

Introspection (no API keys, no data downloads):

* ``gdb list``      — list registered benchmarks and pipeline readiness
* ``gdb info ID``   — show metadata for a single benchmark
* ``gdb suites``    — list named suites (``v0-all``, ``v0-smoke``, …)

Evaluation:

* ``gdb verify``    — run the stub model against ``v0-smoke``. Zero API keys;
  confirms the install is functional.
* ``gdb eval``      — run a real model (online / streaming inference).
* ``gdb submit``    — submit to a provider batch API (~50 %% cheaper).
* ``gdb collect``   — collect results from a previous ``gdb submit``.

Reporting:

* ``gdb score PATH``  — re-score a precomputed CSV of model outputs.
* ``gdb report PATH`` — pretty-print a run-report JSON to markdown.

Every subcommand that runs inference accepts ``--suite NAME`` (preferred) or
an explicit ``--benchmarks ID [ID …]`` list.  Reported results should cite
the suite name **and** the ``lica-gdb`` package version.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .base import BaseBenchmark, TaskType
from .evaluation.reporting import RunReport
from .registry import BenchmarkRegistry
from .runner import BenchmarkRunner
from .suites import describe_suite, list_suites, resolve_suite

logger = logging.getLogger(__name__)

PROVIDER_TO_REGISTRY: Dict[str, str] = {
    "gemini": "google",
    "openai": "openai",
    "openai_image": "openai_image",
    "anthropic": "anthropic",
    "hf": "hf",
    "vllm": "vllm",
    "diffusion": "diffusion",
    "custom": "custom",
}

DEFAULT_MODEL_IDS: Dict[str, str] = {
    "gemini": "gemini-2.0-flash",
    "openai": "gpt-4o",
    "openai_image": "gpt-image-1.5",
    "anthropic": "claude-sonnet-4-20250514",
    "hf": "Qwen/Qwen3-VL-4B-Instruct",
    "vllm": "Qwen/Qwen3-VL-4B-Instruct",
    "diffusion": "flux.2-klein-4b",
    "custom": "custom-entrypoint",
}

_MODALITY_CHOICES = [
    "text",
    "image",
    "both",
    "text_and_image",
    "image_generation",
    "any",
]


# ----------------------------------------------------------------------------
# Output roots
# ----------------------------------------------------------------------------


def _default_output_root() -> Path:
    """Where to write reports / tracker logs when the user doesn't specify.

    When running from the repo checkout, ``./outputs`` already exists and is
    gitignored; when running from a ``pip install``-ed copy we still resolve
    to ``./outputs`` under the caller's cwd, which is the conventional
    "results live next to the command I ran" behaviour.
    """
    return Path.cwd() / "outputs"


def _default_jobs_root() -> Path:
    return Path.cwd() / "jobs"


# ----------------------------------------------------------------------------
# Registry helpers
# ----------------------------------------------------------------------------


def _build_registry() -> BenchmarkRegistry:
    registry = BenchmarkRegistry()
    registry.discover()
    return registry


def _benchmark_pipeline_ready(bench: BaseBenchmark) -> bool:
    """True if the task overrides the default stubs in :class:`BaseBenchmark`."""
    cls = type(bench)
    if getattr(cls, "pipeline_implemented", True) is False:
        return False
    return cls.load_data is not BaseBenchmark.load_data


def _resolve_benchmark_ids(
    args: argparse.Namespace, registry: BenchmarkRegistry
) -> List[str]:
    """Resolve ``--suite`` or ``--benchmarks`` to a concrete list of IDs."""
    suite = getattr(args, "suite", None)
    explicit = getattr(args, "benchmarks", None)
    if suite and explicit:
        raise SystemExit("Specify either --suite or --benchmarks, not both.")
    if suite:
        try:
            return resolve_suite(suite, registry)
        except KeyError as exc:
            raise SystemExit(str(exc)) from exc
    if explicit:
        return list(explicit)
    raise SystemExit("One of --suite or --benchmarks is required.")


# ----------------------------------------------------------------------------
# Model construction
# ----------------------------------------------------------------------------


def _make_stub_model() -> Any:
    from gdb.models.base import BaseModel, Modality, ModelOutput

    class StubModel(BaseModel):
        name = "stub"
        modality = Modality.ANY

        def predict(self, inp: Any) -> ModelOutput:
            return ModelOutput(text="", images=[])

    return StubModel()


def _parse_json_dict_arg(value: Any, *, field_name: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        path = Path(text)
        if path.is_file():
            text = path.read_text(encoding="utf-8")
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError(f"{field_name} must be a JSON object/dict.")
        return parsed
    raise ValueError(f"{field_name} must be a JSON object/dict or JSON file path.")


def _resolve_model_modality(
    args: argparse.Namespace, *, provider: str
) -> Optional[str]:
    if provider == "custom":
        return (
            getattr(args, "custom_modality", None)
            or getattr(args, "model_modality", None)
            or getattr(args, "modality", None)
            or "any"
        )
    return (
        getattr(args, "model_modality", None)
        or getattr(args, "modality", None)
        or None
    )


def _build_model_from_parts(
    provider: str, model_id: str, args: argparse.Namespace
) -> Any:
    from gdb.models import load_model

    if provider == "custom":
        entrypoint = (
            getattr(args, "custom_entry", None)
            or getattr(args, "entrypoint", None)
            or (model_id if model_id != DEFAULT_MODEL_IDS["custom"] else "")
        )
        if not entrypoint:
            raise ValueError(
                "custom provider requires an entrypoint. "
                "Set --custom-entry module.path:attr, "
                "or use custom:module.path:attr in --multi-models."
            )
        init_kwargs = _parse_json_dict_arg(
            getattr(args, "custom_init_kwargs", None)
            or getattr(args, "init_kwargs", None),
            field_name="custom init kwargs",
        )
        custom_modality = _resolve_model_modality(args, provider=provider)
        return load_model(
            "custom",
            entrypoint=entrypoint,
            init_kwargs=init_kwargs,
            modality=custom_modality,
        )

    if provider == "diffusion":
        return load_model(
            "diffusion",
            model_id=model_id,
            resolution=getattr(args, "resolution", 1024),
        )

    kwargs: Dict[str, Any] = {
        "model_id": model_id,
        "temperature": getattr(args, "temperature", 0.0),
    }
    if getattr(args, "credentials", None):
        kwargs["credentials_path"] = args.credentials
    if getattr(args, "max_tokens", None) is not None:
        kwargs["max_tokens"] = args.max_tokens
    if provider == "hf":
        kwargs["device"] = getattr(args, "device", "auto")
        if getattr(args, "max_tokens", None) is not None:
            kwargs["max_new_tokens"] = args.max_tokens
            kwargs.pop("max_tokens", None)
        model_modality = _resolve_model_modality(args, provider=provider)
        if model_modality is not None:
            kwargs["modality"] = model_modality
    if provider == "vllm":
        kwargs["tensor_parallel_size"] = getattr(args, "tensor_parallel_size", 1)
        kwargs["top_p"] = getattr(args, "top_p", 1.0)
        kwargs["top_k"] = getattr(args, "top_k", -1)
        kwargs["repetition_penalty"] = getattr(args, "repetition_penalty", 1.0)
        if getattr(args, "presence_penalty", None) is not None:
            kwargs["presence_penalty"] = args.presence_penalty
        if getattr(args, "limit_mm_per_prompt", None) is not None:
            kwargs["limit_mm_per_prompt"] = {"image": args.limit_mm_per_prompt}
        if getattr(args, "max_num_batched_tokens", None) is not None:
            kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
        if getattr(args, "no_thinking", False):
            kwargs["enable_thinking"] = False
        model_modality = _resolve_model_modality(args, provider=provider)
        if model_modality is not None:
            kwargs["modality"] = model_modality

    return load_model(PROVIDER_TO_REGISTRY[provider], **kwargs)


def _build_single_model(args: argparse.Namespace) -> Tuple[str, Any]:
    provider = args.provider
    model_id = args.model_id or DEFAULT_MODEL_IDS[provider]
    if provider == "custom" and getattr(args, "custom_entry", None):
        name = f"custom:{args.custom_entry}"
    else:
        name = model_id
    return name, _build_model_from_parts(provider, model_id, args)


def _parse_model_spec(spec: str) -> Tuple[str, str, str]:
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


def _build_models(args: argparse.Namespace) -> Dict[str, Any]:
    """Build the ``{name -> model}`` dict requested by the user."""
    if getattr(args, "stub_model", False):
        return {"stub": _make_stub_model()}
    if getattr(args, "multi_models", None):
        models: Dict[str, Any] = {}
        for spec in args.multi_models:
            name, provider, model_id = _parse_model_spec(spec)
            models[name] = _build_model_from_parts(provider, model_id, args)
        return models
    if getattr(args, "provider", None):
        name, model = _build_single_model(args)
        return {name: model}
    raise SystemExit("--provider, --multi-models, or --stub-model required")


# ----------------------------------------------------------------------------
# Preflight
# ----------------------------------------------------------------------------


def _collect_preflight_warnings(
    registry: BenchmarkRegistry,
    benchmark_ids: List[str],
    models: Dict[str, Any],
) -> List[str]:
    from gdb.models.base import Modality

    def _supports_image_output(model: Any, modality: Any) -> bool:
        return bool(
            getattr(
                model,
                "supports_image_output",
                modality in {Modality.IMAGE_GENERATION, Modality.ANY},
            )
        )

    def _supports_video_output(model: Any) -> bool:
        return bool(getattr(model, "supports_video_output", False))

    def _supports_image_input(model: Any, modality: Any) -> bool:
        return bool(
            getattr(
                model,
                "supports_image_input",
                modality in {Modality.TEXT_AND_IMAGE, Modality.ANY},
            )
        )

    def _supports_mask_editing(model: Any) -> bool:
        return bool(getattr(model, "supports_mask_editing", False))

    image_tokens = (
        "input image",
        "layout image",
        "source image",
        "source composite image",
        "rendered image",
        "reference image",
        "component asset",
        "component assets",
        "visual component",
        "visual components",
        "mask",
        "masked",
    )
    visual_tokens = image_tokens + ("video",)

    warnings: List[str] = []
    seen: Set[str] = set()
    for bid in benchmark_ids:
        bench = registry.get(bid)
        inp = str(bench.meta.input_spec or "").lower()
        out = str(bench.meta.output_spec or "").lower()
        needs_visual_input = any(t in inp for t in visual_tokens)
        needs_image_output = any(t in out for t in ("image", "png", "jpg", "jpeg"))
        needs_video_output = any(t in out for t in ("video", "mp4"))
        needs_image_conditioning = needs_image_output and any(
            t in inp for t in image_tokens
        )
        needs_mask_editing = needs_image_output and any(
            t in inp for t in ("mask", "masked", "editable")
        )

        for name, model in models.items():
            modality = getattr(model, "modality", None)

            def _add(msg: str) -> None:
                if msg not in seen:
                    seen.add(msg)
                    warnings.append(msg)

            if needs_visual_input and modality == Modality.TEXT:
                _add(
                    f"{bid} expects visual input ({bench.meta.input_spec}); "
                    f"model '{name}' is text-only."
                )
            if needs_image_output and not _supports_image_output(model, modality):
                _add(
                    f"{bid} expects image output ({bench.meta.output_spec}); "
                    f"model '{name}' may need an image-generation capable wrapper."
                )
            if needs_video_output and not _supports_video_output(model):
                _add(
                    f"{bid} expects video output ({bench.meta.output_spec}); "
                    f"model '{name}' does not advertise video-generation support."
                )
            if needs_image_conditioning and not _supports_image_input(model, modality):
                _add(
                    f"{bid} uses source/reference images ({bench.meta.input_spec}); "
                    f"model '{name}' may ignore those visual inputs."
                )
            if needs_mask_editing and not _supports_mask_editing(model):
                _add(
                    f"{bid} is a masked image-editing task ({bench.meta.input_spec}); "
                    f"model '{name}' does not advertise mask/inpainting support."
                )
    return warnings


# ----------------------------------------------------------------------------
# Shared argument groups
# ----------------------------------------------------------------------------


def _add_selection_arguments(p: argparse.ArgumentParser) -> None:
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--suite",
        metavar="NAME",
        help=f"Named suite. Choices: {', '.join(list_suites())}",
    )
    group.add_argument(
        "--benchmarks",
        nargs="+",
        metavar="ID",
        help="Explicit benchmark IDs (e.g. layout-4 svg-1).",
    )
    p.add_argument(
        "--data",
        default=None,
        help="Override data directory for the task(s). Rarely needed when using --dataset-root.",
    )
    p.add_argument(
        "--dataset-root",
        default=None,
        help="Local Lica bundle root (lica-data/ + benchmarks/). "
        "When omitted, data is loaded from the HuggingFace Hub.",
    )
    p.add_argument(
        "--n",
        type=int,
        default=None,
        help="Limit number of samples per task (default: all).",
    )


def _add_model_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--stub-model",
        action="store_true",
        help="Use the built-in stub model (no API keys). Primarily for smoke tests.",
    )
    p.add_argument("--provider", choices=list(PROVIDER_TO_REGISTRY.keys()))
    p.add_argument("--model-id", default=None)
    p.add_argument(
        "--multi-models",
        nargs="+",
        metavar="SPEC",
        default=None,
        help="Run multiple models in one pass. Format: provider:model_id or alias=provider:model_id",
    )
    p.add_argument("--credentials", default=None)
    p.add_argument(
        "--custom-entry",
        default=None,
        help="Importable Python entrypoint for --provider custom: module.path:attr",
    )
    p.add_argument("--custom-init-kwargs", default=None)
    p.add_argument(
        "--custom-modality",
        choices=_MODALITY_CHOICES,
        default="any",
    )
    p.add_argument(
        "--model-modality",
        choices=_MODALITY_CHOICES,
        default=None,
        help="Override modality declaration for local providers (hf/vllm).",
    )
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=-1)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--presence-penalty", type=float, default=0.0)
    p.add_argument("--device", default="auto", help="HF device (auto/cpu/cuda/mps)")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--limit-mm-per-prompt", type=int, default=None)
    p.add_argument("--max-num-batched-tokens", type=int, default=None)
    p.add_argument("--no-thinking", action="store_true")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument(
        "--input-modality",
        choices=["text", "image", "both"],
        default=None,
        help="Override template-task input modality (text/image/both).",
    )


# ----------------------------------------------------------------------------
# Command handlers
# ----------------------------------------------------------------------------


def cmd_list(args: argparse.Namespace) -> int:
    registry = _build_registry()
    task_type = TaskType(args.task_type) if args.task_type else None
    benches = registry.list(domain=args.domain, task_type=task_type)
    if not benches:
        print("No benchmarks matched the given filters.")
        return 0

    runnable = {b.meta.id for b in registry.list() if _benchmark_pipeline_ready(b)}
    print(f"{'ID':<18} {'Type':<14} {'Domain':<14} {'Pipeline':<9} Name")
    print("-" * 90)
    for b in sorted(benches, key=lambda x: x.meta.id):
        ready = "ready" if b.meta.id in runnable else "-"
        print(
            f"{b.meta.id:<18} {b.meta.task_type.value:<14} "
            f"{b.meta.domain:<14} {ready:<9} {b.meta.name}"
        )
    total_runnable = sum(1 for b in benches if b.meta.id in runnable)
    print(
        f"\n{len(benches)} benchmark(s); "
        f"{total_runnable} have a runnable inference pipeline."
    )
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    registry = _build_registry()
    try:
        b = registry.get(args.benchmark_id)
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    m = b.meta
    print(f"ID:          {m.id}")
    print(f"Name:        {m.name}")
    print(f"Task type:   {m.task_type.value}")
    print(f"Domain:      {m.domain}")
    print(f"Description: {m.description}")
    if m.input_spec:
        print(f"Input:       {m.input_spec}")
    if getattr(m, "output_spec", None):
        print(f"Output:      {m.output_spec}")
    if m.metrics:
        print(f"Metrics:     {', '.join(m.metrics)}")
    if m.tags:
        print(f"Tags:        {', '.join(m.tags)}")
    print(f"Pipeline:    {'ready' if _benchmark_pipeline_ready(b) else 'not implemented'}")
    return 0


def cmd_suites(args: argparse.Namespace) -> int:
    registry = _build_registry()
    if args.name:
        info = describe_suite(args.name, registry)
        print(f"Suite: {info['name']}  ({info['kind']}, {info['n_tasks']} tasks)")
        for tid in info["task_ids"]:
            print(f"  - {tid}")
        return 0

    print(f"{'Suite':<24} {'Kind':<8} {'Tasks':>5}")
    print("-" * 40)
    for name in list_suites():
        info = describe_suite(name, registry)
        print(f"{name:<24} {info['kind']:<8} {info['n_tasks']:>5}")
    return 0


def _run_online(
    registry: BenchmarkRegistry,
    benchmark_ids: List[str],
    models: Dict[str, Any],
    args: argparse.Namespace,
) -> bool:
    input_modality = None
    if args.input_modality:
        from gdb.models.base import Modality

        input_modality = {
            "text": Modality.TEXT,
            "image": Modality.IMAGE,
            "both": Modality.TEXT_AND_IMAGE,
        }[args.input_modality]

    out_root = Path(args.output_dir) if args.output_dir else _default_output_root()
    save_dir: Optional[Path] = None
    if args.save_images:
        save_dir = (
            Path(args.images_dir) if args.images_dir else out_root / "generated-images"
        )
        save_dir.mkdir(parents=True, exist_ok=True)

    runner = BenchmarkRunner(registry)
    combined = RunReport()
    all_ok = True

    for bid in benchmark_ids:
        bench = registry.get(bid)
        print(f"\n[{bid}] {bench.meta.name}")
        try:
            if args.data:
                data_display = args.data
            elif args.dataset_root:
                data_display = str(bench.resolve_data_dir(args.dataset_root))
            else:
                data_display = "HuggingFace Hub"
        except FileNotFoundError as exc:
            print(f"  FAILED: {exc}")
            all_ok = False
            continue
        print(f"  data: {data_display}")

        t0 = time.time()
        try:
            report = runner.run(
                benchmark_ids=[bid],
                models=models,
                data_dir=args.data,
                dataset_root=args.dataset_root,
                n=args.n,
                batch_size=args.batch_size,
                prediction_save_dir=save_dir,
                input_modality=input_modality,
            )
        except Exception as exc:  # noqa: BLE001 — user-visible runtime failure
            print(f"  FAILED: {exc}")
            all_ok = False
            continue

        for name, result in sorted(report.results[bid].items()):
            scores = ", ".join(
                f"{k}={v:.4f}" for k, v in sorted(result.scores.items())
            )
            print(
                f"  {name}: {scores}  "
                f"(n={result.count}, ok={result.success_count}, "
                f"fail={result.failure_count}, "
                f"fail_rate={result.failure_rate:.1%}, "
                f"{time.time() - t0:.1f}s)"
            )
        combined.results[bid] = report.results[bid]

    if combined.results:
        if args.output:
            combined.save(args.output)
            print(f"\nSaved report to {args.output}")
        else:
            out_root.mkdir(parents=True, exist_ok=True)
            for bid in combined.results:
                RunReport(results={bid: combined.results[bid]}).save(
                    str(out_root / f"{bid}.csv")
                )
            print(f"\nSaved per-task CSVs to {out_root}/")

    if not args.no_log and len(runner.tracker) > 0:
        out_root.mkdir(parents=True, exist_ok=True)
        log_path = out_root / "tracker.jsonl"
        runner.tracker.save(str(log_path))
        print(f"Tracker log: {log_path}")

    if save_dir is not None:
        print(f"Generated images: {save_dir}")

    return all_ok


def cmd_eval(args: argparse.Namespace) -> int:
    registry = _build_registry()
    benchmark_ids = _resolve_benchmark_ids(args, registry)
    try:
        models = _build_models(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if not args.dataset_root and not args.data:
        print("[info] No --dataset-root provided; loading data from HuggingFace Hub.")

    warnings = _collect_preflight_warnings(registry, benchmark_ids, models)
    if warnings:
        print("\n[preflight] Potential model/task compatibility issues:")
        for msg in warnings:
            print(f"  - {msg}")
        print("  Continue with caution; some tasks may require a different model.\n")

    ok = _run_online(registry, benchmark_ids, models, args)
    return 0 if ok else 1


def _bundled_verify_root() -> Optional[Path]:
    """Return the path to the bundled ``_verify_data`` fixture, if shipped."""
    here = Path(__file__).resolve().parent
    candidate = here / "_verify_data"
    if candidate.is_dir() and (candidate / "benchmarks").is_dir():
        return candidate
    return None


def cmd_verify(args: argparse.Namespace) -> int:
    """Smoke test: run the stub model against the requested suite (default smoke)."""
    registry = _build_registry()
    suite_name = args.suite or "v0-smoke"
    try:
        benchmark_ids = resolve_suite(suite_name, registry)
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.benchmarks:
        benchmark_ids = list(args.benchmarks)

    # Default to the bundled fixture so `gdb verify` needs no downloads.
    dataset_root = args.dataset_root
    using_bundled = False
    if not dataset_root and not args.data:
        bundled = _bundled_verify_root()
        if bundled is not None:
            dataset_root = str(bundled)
            using_bundled = True

    source_desc = (
        "bundled fixture"
        if using_bundled
        else (f"dataset-root={dataset_root}" if dataset_root else "HuggingFace Hub")
    )
    print(
        f"Verifying install with stub model on {len(benchmark_ids)} task(s) "
        f"(suite={suite_name}, n={args.n or 'all'}, data={source_desc}).\n"
        "Scores will be ~0 — this only checks that inference & scoring run end-to-end."
    )

    verify_args = argparse.Namespace(
        **{k: v for k, v in vars(args).items() if k != "dataset_root"},
        dataset_root=dataset_root,
        provider=None,
        model_id=None,
        multi_models=None,
        stub_model=True,
        custom_entry=None,
        custom_init_kwargs=None,
        custom_modality="any",
        model_modality=None,
        credentials=None,
        temperature=0.0,
        max_tokens=None,
        top_p=1.0,
        top_k=-1,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        device="auto",
        tensor_parallel_size=1,
        resolution=1024,
        limit_mm_per_prompt=None,
        max_num_batched_tokens=None,
        no_thinking=False,
        batch_size=None,
        input_modality=None,
        save_images=False,
        images_dir=None,
    )
    models = {"stub": _make_stub_model()}
    ok = _run_online(registry, benchmark_ids, models, verify_args)
    if ok:
        print("\n[verify] OK — install is functional.")
    else:
        print("\n[verify] FAILED — see errors above.")
    return 0 if ok else 1


def cmd_submit(args: argparse.Namespace) -> int:
    from gdb.inference import BATCH_PROVIDERS, make_batch_runner, save_job_manifest

    if not args.provider:
        print("--provider is required for `gdb submit`.", file=sys.stderr)
        return 2
    if args.provider not in BATCH_PROVIDERS:
        print(
            f"`gdb submit` requires one of: {', '.join(sorted(BATCH_PROVIDERS))}",
            file=sys.stderr,
        )
        return 2

    registry = _build_registry()
    benchmark_ids = _resolve_benchmark_ids(args, registry)
    if len(benchmark_ids) != 1:
        print(
            "`gdb submit` currently supports exactly one benchmark per job. "
            "Loop over tasks in a shell for now.",
            file=sys.stderr,
        )
        return 2
    bid = benchmark_ids[0]
    bench = registry.get(bid)

    model_id = args.model_id or DEFAULT_MODEL_IDS[args.provider]
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

    if args.data:
        data_display = args.data
    elif args.dataset_root:
        data_display = str(bench.resolve_data_dir(args.dataset_root))
    else:
        data_display = "HuggingFace Hub"
    print(f"\n[{bid}] {bench.meta.name}")
    print(f"  data: {data_display}")
    print(f"  provider: {args.provider} / {model_id}")

    manifest_data = runner.submit(
        bid,
        batch_runner,
        data_dir=args.data,
        dataset_root=args.dataset_root,
        n=args.n,
    )
    extra = {"benchmark_id": bid}
    if args.provider == "gemini" and hasattr(batch_runner, "_last_submit_meta"):
        extra["job_prefix"] = batch_runner._last_submit_meta["job_prefix"]

    jobs_dir = Path(args.jobs_dir) if args.jobs_dir else _default_jobs_root()
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
    print(f"  Manifest:      {manifest_path}")
    print(f"\n  To collect: gdb collect {manifest_path}")
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    from gdb.inference import load_job_manifest, make_batch_runner

    manifest = load_job_manifest(args.manifest)
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
    job_prefix = manifest.get("job_prefix")
    if provider == "gemini" and job_prefix:
        collect_kwargs["job_prefix"] = job_prefix

    if benchmark_id:
        registry = _build_registry()
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
        scores = ", ".join(f"{k}={v:.4f}" for k, v in sorted(result.scores.items()))
        print(
            f"\n  [{benchmark_id}] {scores}  "
            f"(n={result.count}, ok={result.success_count}, "
            f"fail={result.failure_count}, "
            f"fail_rate={result.failure_rate:.1%})"
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
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    """Score a precomputed CSV of model outputs (no model inference)."""
    registry = _build_registry()
    runner = BenchmarkRunner(registry)
    report = runner.run_from_csv(args.csv_path)
    print(report.summary())
    if args.output:
        report.save(args.output)
        print(f"\nResults saved to {args.output}")
    return 0


def _render_markdown_report(report_dict: Dict[str, Any]) -> str:
    """Minimal markdown renderer for a ``RunReport`` JSON dump."""
    lines: List[str] = ["# GDB run report", ""]
    meta = report_dict.get("metadata", {})
    if meta:
        lines.append("## Metadata")
        lines.append("")
        for k, v in meta.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")

    lines.append("## Results")
    lines.append("")
    results = report_dict.get("results", {})
    if not results:
        lines.append("_(empty)_")
        return "\n".join(lines)

    metric_cols: List[str] = []
    seen_cols: Set[str] = set()
    for models in results.values():
        for res in models.values():
            for m in (res.get("scores") or {}):
                if m not in seen_cols:
                    seen_cols.add(m)
                    metric_cols.append(m)

    header = ["Benchmark", "Model", "n", "fail_rate", *metric_cols]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for bid, models in sorted(results.items()):
        for model_name, res in sorted(models.items()):
            row = [
                bid,
                model_name,
                str(res.get("count", "")),
                f"{res.get('failure_rate', 0):.1%}",
            ]
            for col in metric_cols:
                val = (res.get("scores") or {}).get(col)
                row.append(f"{val:.4f}" if isinstance(val, (int, float)) else "—")
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def cmd_report(args: argparse.Namespace) -> int:
    path = Path(args.report_path)
    if not path.is_file():
        print(f"No such file: {path}", file=sys.stderr)
        return 1
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    md = _render_markdown_report(data)
    if args.output:
        Path(args.output).write_text(md, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(md)
    return 0


# ----------------------------------------------------------------------------
# Argument parser
# ----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gdb",
        description="GDB — GraphicDesignBench CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print the installed lica-gdb version and exit.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging."
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    p_list = sub.add_parser("list", help="List registered benchmarks.")
    p_list.add_argument("--domain", help="Filter by domain (e.g. svg, layout).")
    p_list.add_argument(
        "--task-type",
        choices=["understanding", "generation"],
        help="Filter by task type.",
    )

    p_info = sub.add_parser("info", help="Show details for a single benchmark.")
    p_info.add_argument("benchmark_id", help="Benchmark ID (e.g. svg-1).")

    p_suites = sub.add_parser("suites", help="List named suites (or expand one).")
    p_suites.add_argument(
        "name", nargs="?", help="If given, print the task IDs in this suite."
    )

    p_eval = sub.add_parser(
        "eval",
        help="Run online inference against one or more benchmarks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_selection_arguments(p_eval)
    _add_model_arguments(p_eval)
    p_eval.add_argument("--output", "-o", default=None, help="Save report (.json or .csv).")
    p_eval.add_argument(
        "--output-dir",
        default=None,
        help="Directory for per-task CSVs / tracker log (default: ./outputs).",
    )
    p_eval.add_argument("--save-images", action="store_true")
    p_eval.add_argument("--images-dir", default=None)
    p_eval.add_argument("--no-log", action="store_true")

    p_verify = sub.add_parser(
        "verify",
        help="Smoke-test the install with the stub model (no API keys).",
    )
    p_verify.add_argument("--suite", default=None, help="Defaults to v0-smoke.")
    p_verify.add_argument("--benchmarks", nargs="+", metavar="ID")
    p_verify.add_argument("--dataset-root", default=None)
    p_verify.add_argument("--data", default=None)
    p_verify.add_argument("--n", type=int, default=2)
    p_verify.add_argument("--output", "-o", default=None)
    p_verify.add_argument("--output-dir", default=None)
    p_verify.add_argument("--no-log", action="store_true")

    p_submit = sub.add_parser(
        "submit", help="Submit a batch-API job for a single benchmark."
    )
    _add_selection_arguments(p_submit)
    _add_model_arguments(p_submit)
    p_submit.add_argument("--bucket", default=None, help="GCS bucket for Gemini batch.")
    p_submit.add_argument("--poll-interval", type=int, default=30)
    p_submit.add_argument(
        "--jobs-dir", default=None, help="Where to write the job manifest (default: ./jobs)."
    )

    p_collect = sub.add_parser(
        "collect", help="Collect results from a previous `gdb submit` manifest."
    )
    p_collect.add_argument("manifest", help="Path to job manifest JSON.")
    p_collect.add_argument("--credentials", default=None)
    p_collect.add_argument("--bucket", default=None)
    p_collect.add_argument("--poll-interval", type=int, default=30)
    p_collect.add_argument("--output", "-o", default=None)

    p_score = sub.add_parser(
        "score", help="Re-score a precomputed CSV of model outputs."
    )
    p_score.add_argument("csv_path", help="CSV with columns: task, expected_output, <model>_output.")
    p_score.add_argument("--output", "-o", default=None)

    p_report = sub.add_parser(
        "report", help="Render a run-report JSON as markdown."
    )
    p_report.add_argument("report_path", help="Path to a run-report JSON.")
    p_report.add_argument("--output", "-o", default=None, help="Write markdown to this file.")

    return parser


_DISPATCH: Dict[str, Callable[[argparse.Namespace], int]] = {
    "list": cmd_list,
    "info": cmd_info,
    "suites": cmd_suites,
    "eval": cmd_eval,
    "verify": cmd_verify,
    "submit": cmd_submit,
    "collect": cmd_collect,
    "score": cmd_score,
    "report": cmd_report,
}


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "version", False):
        from . import __version__

        print(f"lica-gdb {__version__}")
        return

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")

    if not args.command:
        parser.print_help()
        sys.exit(0)

    handler = _DISPATCH[args.command]
    sys.exit(handler(args))


if __name__ == "__main__":
    main()
