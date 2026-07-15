"""Runner: builds the op x backend x config x direction cross product and drives timing.

forward:  run_batch(n) -> for _ in range(n): backend.ops[name](*inputs)
backward: build forward output once outside the loop (requires_grad=True inputs),
          then run_batch(n) -> for _ in range(n): backend.backward(out)
          backward is seeded by ones_like(out) inside backend.backward().
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .backend import BACKENDS
from .report import append_row, csv_path
from .spec import CATALOG, OpEntry, resolve_cases
from .timer import measure

if TYPE_CHECKING:
    from .backend import Backend


@dataclass
class RunConfig:
    """Parameters passed from the CLI to the runner."""
    tag: str = "short"          # "short" | "long" | "all"
    devices: list[str] | None = None  # None = all
    backends: list[str] | None = None  # None = all registered
    warmup: int = 100
    min_time_per_test: float = 1.0
    num_runs: int = 1
    iterations: int | None = None   # force fixed count (disables adaptive)
    output_dir: str | None = None
    verbose: bool = False


def _tag_selected(case_tag: str, filter_tag: str) -> bool:
    if filter_tag == "all":
        return True
    return case_tag == filter_tag


def _make_run_batch_forward(backend: "Backend", op_name: str, call_args: list):
    fn = backend.ops[op_name]

    def run_batch(n: int) -> None:
        for _ in range(n):
            fn(*call_args)

    return run_batch


def _make_run_batch_backward(backend: "Backend", out) -> callable:
    def run_batch(n: int) -> None:
        for _ in range(n):
            backend.backward(out)

    return run_batch


def _make_sync(backend: "Backend", device: str):
    def sync():
        backend.sync(device)

    return sync


def _resolve_input_shapes(entry: OpEntry, raw: dict) -> list[list[int]]:
    """Return the list of tensor shapes for a given op and raw config row."""
    if entry.category == "binary":
        return [list(raw["in_one"]), list(raw["in_two"])]
    if entry.category == "unary":
        return [list(raw["in_one"])]
    # reduction: shape depends on dim
    R, V, dim = raw["R"], raw["V"], raw["dim"]
    if dim == 0:
        shape = [R, V]
    else:
        shape = [V, R]
    return [shape]


def run_all(cfg: RunConfig) -> None:
    """Execute the full benchmark cross product and write results."""
    active_backends: dict[str, Backend] = {}
    if cfg.backends is None:
        active_backends = BACKENDS
    else:
        for name in cfg.backends:
            if name not in BACKENDS:
                raise KeyError(
                    f"Backend '{name}' not registered. "
                    f"Available: {list(BACKENDS.keys())}"
                )
            active_backends[name] = BACKENDS[name]

    for entry in CATALOG:
        cases = resolve_cases(entry)
        for backend in active_backends.values():
            if entry.name not in backend.ops:
                continue

            out_path = csv_path(backend.name, cfg.output_dir)

            for case in cases:
                if not _tag_selected(case.tag, cfg.tag):
                    continue

                if cfg.devices is not None and case.device not in cfg.devices:
                    continue

                if case.device == "cuda" and not backend.cuda_available():
                    continue

                shapes = _resolve_input_shapes(entry, case.raw)
                sync_fn = _make_sync(backend, case.device)

                for direction in ("forward", "backward"):
                    requires_grad = direction == "backward"

                    inputs = [
                        backend.make_input(
                            shape, case.device, case.dtype, requires_grad
                        )
                        for shape in shapes
                    ]

                    # Reduction ops take the axis as a trailing call arg; it varies
                    # per case (dim 0 vs dim 1) so it cannot be baked into ops.
                    if entry.category == "reduction":
                        call_args = inputs + [case.raw["dim"]]
                    else:
                        call_args = inputs

                    if direction == "forward":
                        run_batch = _make_run_batch_forward(
                            backend, entry.name, call_args
                        )
                    else:
                        # build output once outside the timed loop
                        out = backend.ops[entry.name](*call_args)
                        run_batch = _make_run_batch_backward(backend, out)

                    iters = cfg.iterations if cfg.iterations is not None else 100
                    times = measure(
                        run_batch,
                        sync_fn,
                        iters=iters,
                        warmup=cfg.warmup,
                        min_time_per_test=cfg.min_time_per_test,
                        num_runs=cfg.num_runs,
                    )

                    median_us = statistics.median(times)

                    append_row(
                        out_path,
                        framework=backend.name,
                        op=entry.name,
                        category=entry.category,
                        direction=direction,
                        device=case.device,
                        dtype=case.dtype,
                        shape=case.shape,
                        tag=case.tag,
                        iters=iters,
                        runs=cfg.num_runs,
                        time_us=median_us,
                    )

                    if cfg.verbose:
                        print(
                            f"{backend.name:12s}  {entry.name:6s}  {direction:8s}"
                            f"  {case.device:4s}  {case.shape:20s}"
                            f"  {case.tag:5s}  {median_us:10.3f} us"
                        )
