"""CLI entrypoint for the common benchmark harness.

Usage:
    python -m benchmarks.common.run --backends rushlite torch --tag short --device cpu

Backends are imported lazily so --backends rushlite never imports torch and vice versa.
Backend modules are expected at benchmarks/backends/<name>_backend.py and must call
register_backend() at module level.
"""

from __future__ import annotations

import argparse
import importlib
import sys

_BACKEND_MODULE_MAP = {
    # logical name -> importable module path
    "torch": "benchmarks.backends.torch_backend",
    "rushlite": "benchmarks.backends.rushlite_backend",
}


def _load_backend(name: str) -> None:
    """Import the backend module for name, triggering its register_backend() call."""
    module_path = _BACKEND_MODULE_MAP.get(name)
    if module_path is None:
        # allow ad-hoc dotted module paths as well
        module_path = name

    try:
        importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        print(
            f"error: could not load backend '{name}' "
            f"(tried module '{module_path}'): {exc}",
            file=sys.stderr,
        )
        sys.exit(1)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Framework-agnostic operator microbenchmarks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--backends",
        nargs="+",
        default=["rushlite", "torch"],
        help="Backend names to benchmark. Each name must have a corresponding "
        "module in benchmarks/backends/.",
    )
    p.add_argument(
        "--tag",
        choices=["short", "long", "all"],
        default="short",
        help="Select which tagged cases to run.",
    )
    p.add_argument(
        "--device",
        nargs="+",
        default=None,
        help="Restrict to specific devices (e.g. cpu cuda). Default: all.",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=100,
        help="Number of warmup iterations (discarded).",
    )
    p.add_argument(
        "--min-time-per-test",
        type=float,
        default=1.0,
        dest="min_time_per_test",
        help="Minimum cumulative seconds per test before stopping.",
    )
    p.add_argument(
        "--num-runs",
        type=int,
        default=1,
        dest="num_runs",
        help="Number of independent measurement runs; report median.",
    )
    p.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Force a fixed iteration count (disables adaptive loop).",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        dest="output_dir",
        help="Directory for CSV output files. Default: benchmarks/results/<hostname>/.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print each result as it is measured.",
    )
    p.add_argument(
        "--list-cases",
        action="store_true",
        default=False,
        dest="list_cases",
        help="Print the cross-product of cases that would run and exit.",
    )
    return p.parse_args(argv)


def _list_cases(args: argparse.Namespace) -> None:
    """Print the benchmark cases that match the current filter and exit."""
    from .spec import CATALOG, resolve_cases

    def tag_ok(tag: str) -> bool:
        return args.tag == "all" or tag == args.tag

    count = 0
    for entry in CATALOG:
        for case in resolve_cases(entry):
            if not tag_ok(case.tag):
                continue
            if args.device and case.device not in args.device:
                continue
            for direction in ("forward", "backward"):
                print(
                    f"{entry.name:6s}  {entry.category:9s}  {direction:8s}"
                    f"  {case.device:4s}  {case.dtype:7s}  {case.shape:20s}"
                    f"  {case.tag}"
                )
                count += 1
    print(f"\ntotal: {count} cases")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if args.list_cases:
        _list_cases(args)
        return

    # load backend modules lazily - each registers itself via register_backend()
    for name in args.backends:
        _load_backend(name)

    from .backend import BACKENDS
    from .runner import RunConfig, run_all

    if not BACKENDS:
        print(
            "error: no backends are registered after loading requested modules.",
            file=sys.stderr,
        )
        sys.exit(1)

    cfg = RunConfig(
        tag=args.tag,
        devices=args.device,
        backends=args.backends,
        warmup=args.warmup,
        min_time_per_test=args.min_time_per_test,
        num_runs=args.num_runs,
        iterations=args.iterations,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )

    run_all(cfg)


if __name__ == "__main__":
    main()
