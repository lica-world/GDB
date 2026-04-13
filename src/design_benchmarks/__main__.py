"""CLI entry point: ``python -m design_benchmarks``."""

import argparse
import sys

from .base import TaskType
from .registry import BenchmarkRegistry
from .runner import BenchmarkRunner


def _build_registry() -> BenchmarkRegistry:
    registry = BenchmarkRegistry()
    registry.discover()
    return registry


def cmd_list(args: argparse.Namespace) -> None:
    registry = _build_registry()

    task_type = None
    if args.task_type:
        task_type = TaskType(args.task_type)

    benchmarks = registry.list(
        domain=args.domain,
        task_type=task_type,
    )

    if not benchmarks:
        print("No benchmarks matched the given filters.")
        return

    print(f"{'ID':<20} {'Type':<15} {'Domain':<16} {'Name'}")
    print("-" * 80)
    for b in benchmarks:
        print(
            f"{b.meta.id:<20} {b.meta.task_type.value:<15} "
            f"{b.meta.domain:<16} {b.meta.name}"
        )
    print(f"\n{len(benchmarks)} benchmark(s) found.")


def cmd_info(args: argparse.Namespace) -> None:
    registry = _build_registry()
    try:
        b = registry.get(args.benchmark_id)
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    m = b.meta
    print(f"ID:          {m.id}")
    print(f"Name:        {m.name}")
    print(f"Task type:   {m.task_type.value}")
    print(f"Domain:      {m.domain}")
    print(f"Description: {m.description}")
    if m.input_spec:
        print(f"Input:       {m.input_spec}")
    if m.metrics:
        print(f"Metrics:     {', '.join(m.metrics)}")
    if m.tags:
        print(f"Tags:        {', '.join(m.tags)}")


def cmd_run(args: argparse.Namespace) -> None:
    registry = _build_registry()
    runner = BenchmarkRunner(registry)
    report = runner.run_from_csv(args.csv_path)
    print(report.summary())

    if args.output:
        report.save(args.output)
        print(f"\nResults saved to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m design_benchmarks",
        description="Design benchmark framework",
    )
    sub = parser.add_subparsers(dest="command")

    p_list = sub.add_parser("list", help="List registered benchmarks")
    p_list.add_argument("--domain", help="Filter by domain (e.g. svg, temporal)")
    p_list.add_argument(
        "--task-type",
        choices=["understanding", "generation"],
        help="Filter by task type",
    )
    p_info = sub.add_parser("info", help="Show details for a benchmark")
    p_info.add_argument("benchmark_id", help="Benchmark ID (e.g. svg-1, layout-1)")

    p_run = sub.add_parser("run", help="Run benchmarks from a CSV file")
    p_run.add_argument("csv_path", help="Path to CSV with model outputs")
    p_run.add_argument("--output", "-o", help="Save results to JSON file")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    {"list": cmd_list, "info": cmd_info, "run": cmd_run}[args.command](args)


if __name__ == "__main__":
    main()
