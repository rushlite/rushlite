"""Case catalog and run configuration for the fusion training benchmarks.

The benchmark compares Rushlite against itself (capture/fusion off vs on) on
one full recurrent-cell training step: zero grads, forward, scalar loss,
backward. CUDA and float32 only.

Batch size is a real configuration axis so the schema can grow, but only
batch size one is supported today: the fused code generator does not index
broadcast leaves correctly, and a textbook [1, H] bias would broadcast over
[B, H]. resolve_cases() rejects anything else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Mode = Literal["eager", "fused"]

MODES: tuple[Mode, ...] = ("eager", "fused")
WORKLOAD_NAMES: tuple[str, ...] = ("gru", "lstm")
SUPPORTED_BATCH_SIZES: tuple[int, ...] = (1,)

DEVICE = "cuda"
DTYPE = "float32"

# hidden sizes per tag; input size is fixed equal to hidden size for now
TAG_SIZES: dict[str, tuple[int, ...]] = {
    "short": (256,),
    "long": (64, 128, 256, 512, 1024),
}


@dataclass(frozen=True)
class FusionCase:
    """One fully-resolved benchmark case (mode is not part of the case)."""

    workload: str
    input_size: int
    hidden_size: int
    batch_size: int
    device: str
    dtype: str
    tag: str


@dataclass(frozen=True)
class RunConfig:
    """Parameters passed from the CLI to the runner."""

    workloads: tuple[str, ...] = ("gru", "lstm")
    modes: tuple[Mode, ...] = ("eager", "fused")
    tags: tuple[str, ...] = ("short",)
    batch_sizes: tuple[int, ...] = (1,)
    seed: int = 0
    warmup: int = 10
    samples: int = 10
    iterations: int | None = None
    output: str | None = None


def resolve_cases(config: RunConfig) -> list[FusionCase]:
    """Expand a RunConfig into concrete cases, validating every axis."""
    for name in config.workloads:
        if name not in WORKLOAD_NAMES:
            raise ValueError(
                f"unknown workload '{name}'; expected one of {WORKLOAD_NAMES}"
            )
    for mode in config.modes:
        if mode not in MODES:
            raise ValueError(f"unknown mode '{mode}'; expected one of {MODES}")
    for tag in config.tags:
        if tag not in TAG_SIZES:
            raise ValueError(f"unknown tag '{tag}'; expected one of {tuple(TAG_SIZES)}")
    for batch_size in config.batch_sizes:
        if batch_size not in SUPPORTED_BATCH_SIZES:
            raise ValueError(
                f"batch size {batch_size} is not supported yet; "
                f"only {SUPPORTED_BATCH_SIZES} (broadcast leaves are not "
                "handled by the fused code generator)"
            )

    cases: list[FusionCase] = []
    for tag in config.tags:
        for hidden_size in TAG_SIZES[tag]:
            for workload in config.workloads:
                for batch_size in config.batch_sizes:
                    cases.append(
                        FusionCase(
                            workload=workload,
                            input_size=hidden_size,
                            hidden_size=hidden_size,
                            batch_size=batch_size,
                            device=DEVICE,
                            dtype=DTYPE,
                            tag=tag,
                        )
                    )
    return cases
