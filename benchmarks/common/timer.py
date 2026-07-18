"""Adaptive timing loop - framework-agnostic.

Algorithm mirrors pytorch's BenchmarkRunner._measure_metrics:
  - warmup: sync, run_batch(warmup), sync
  - per measured round: sync, t0=perf_counter, run_batch(iters), sync, record us/iter
  - grow iters *= multiplier until (batch_dt > min_secs or iters > max_iters)
    AND cumulative_time > min_time_per_test
  - report MEDIAN across rounds; repeat num_runs times
"""

import math
import statistics
from dataclasses import dataclass
from time import perf_counter
from typing import Callable


def measure(
    run_batch,
    sync,
    *,
    iters: int = 100,
    warmup: int = 100,
    min_time_per_test: float = 1.0,
    min_secs: float = 1.0,
    max_iters: float = 1e6,
    multiplier: int = 2,
    num_runs: int = 1,
) -> list[float]:
    """Run run_batch adaptively and return per-iteration time in microseconds.

    Args:
        run_batch: callable(n) - invokes the op n times without syncing.
        sync: callable() - blocks until device work completes; no-op on cpu.
        iters: starting iteration count.
        warmup: number of warmup iterations (results discarded).
        min_time_per_test: minimum cumulative seconds before stopping.
        min_secs: minimum seconds for a single batch to count as significant.
        max_iters: hard cap on iters before forcing exit.
        multiplier: factor by which iters grows each round.
        num_runs: how many independent runs to perform; returns list of medians.

    Returns:
        List of length num_runs, each entry is the median us/iter across rounds.
    """
    results = []
    for _ in range(num_runs):
        # warmup - absorbs JIT/codegen so it never lands in a timed batch
        sync()
        run_batch(warmup)
        sync()

        cur_iters = iters
        cumulative_time = 0.0
        time_trace: list[float] = []

        while True:
            # leading sync drains any prior async tail
            sync()
            t0 = perf_counter()
            run_batch(cur_iters)
            sync()
            dt = perf_counter() - t0

            us_per_iter = 1e6 * dt / cur_iters
            time_trace.append(us_per_iter)
            cumulative_time += dt

            significant = (
                dt > min_secs or cur_iters > max_iters
            ) and cumulative_time > min_time_per_test

            if significant:
                break

            cur_iters = int(cur_iters * multiplier)

        results.append(statistics.median(time_trace))

    return results


@dataclass(frozen=True)
class Measurement:
    """Fixed-sample timing result: per-iteration latencies in microseconds."""

    iterations_per_sample: int
    samples_us: tuple[float, ...]

    @property
    def median_us(self) -> float:
        return statistics.median(self.samples_us)

    @property
    def min_us(self) -> float:
        return min(self.samples_us)


def measure_samples(
    run_once: Callable[[], None],
    synchronize: Callable[[], None],
    *,
    warmup: int,
    samples: int,
    iterations: int | None = None,
    min_sample_secs: float = 0.05,
) -> Measurement:
    """Time run_once with a fixed number of samples and iterations per sample.

    Unlike measure(), the iteration count never changes between samples: it is
    calibrated once (unless supplied) and every sample runs exactly that many
    iterations, so samples are comparable to each other.

    Steps:
      1. synchronize, run warmup iterations, synchronize;
      2. calibrate iterations per sample once (skipped when iterations given);
      3. for each sample: synchronize, time `iterations` calls to run_once,
         synchronize, record us per iteration.

    run_once must enqueue one fresh unit of work and must not synchronize or
    transfer results itself; device completion is owned by `synchronize`.
    """
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    if samples < 1:
        raise ValueError("samples must be >= 1")
    if iterations is not None and iterations < 1:
        raise ValueError("iterations must be >= 1 when supplied")

    synchronize()
    for _ in range(warmup):
        run_once()
    synchronize()

    if iterations is None:
        t0 = perf_counter()
        run_once()
        synchronize()
        once = perf_counter() - t0
        if once >= min_sample_secs:
            iterations = 1
        else:
            iterations = math.ceil(min_sample_secs / max(once, 1e-9))

    samples_us: list[float] = []
    for _ in range(samples):
        synchronize()
        t0 = perf_counter()
        for _ in range(iterations):
            run_once()
        synchronize()
        dt = perf_counter() - t0
        samples_us.append(1e6 * dt / iterations)

    return Measurement(iterations_per_sample=iterations, samples_us=tuple(samples_us))
