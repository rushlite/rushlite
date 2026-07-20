"""CLI entrypoint for the fusion training benchmarks.

Usage:
    python -m benchmarks.fusion.run --tags short --output results.csv

Parses arguments, builds a RunConfig, invokes the runner, writes results,
and prints the summary. Nothing else lives here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .report import print_summary, write_results
from .runner import FusionRunner
from .spec import MODES, WORKLOAD_NAMES, RunConfig, TAG_SIZES
from .workloads import GRUWorkload, LSTMWorkload


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=("Recurrent-cell training benchmarks: Rushlite eager vs fused."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--workloads",
        nargs="+",
        choices=list(WORKLOAD_NAMES),
        default=list(WORKLOAD_NAMES),
        help="Cell workloads to run.",
    )
    p.add_argument(
        "--modes",
        nargs="+",
        choices=list(MODES),
        default=list(MODES),
        help="Execution modes to measure.",
    )
    p.add_argument(
        "--tags",
        nargs="+",
        choices=list(TAG_SIZES),
        default=["short"],
        help="Predefined size sets to run.",
    )
    p.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1],
        dest="batch_sizes",
        help="Batch sizes (only 1 is supported today).",
    )
    p.add_argument("--seed", type=int, default=0, help="Host data seed.")
    p.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup training computations per case and mode.",
    )
    p.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Timed samples per case and mode.",
    )
    p.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Iterations per sample; default calibrates once per case.",
    )
    p.add_argument(
        "--output",
        default=None,
        help="CSV output path. Default: no file, summary only.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    config = RunConfig(
        workloads=tuple(args.workloads),
        modes=tuple(args.modes),
        tags=tuple(args.tags),
        batch_sizes=tuple(args.batch_sizes),
        seed=args.seed,
        warmup=args.warmup,
        samples=args.samples,
        iterations=args.iterations,
        output=args.output,
    )

    workloads = {w.name: w for w in (GRUWorkload(), LSTMWorkload())}
    rows = FusionRunner(config, workloads).run()

    if config.output is not None:
        write_results(Path(config.output), rows)
        print(f"wrote {len(rows)} rows to {config.output}\n")

    print_summary(rows)


if __name__ == "__main__":
    main()
