"""Drives one-step recurrent-cell training benchmarks, eager vs fused.

Lifecycle per timed iteration (fresh graphs every time, unlike the generic
operator runner's backward-on-one-graph loop):

    prepared.zero_grad()
    with rushlite.capture_on(enable=mode == "fused"):
        states = workload.forward(prepared)
        loss = squared_error(states.hidden, prepared.target)
        loss.backward()

Nothing is realized to host inside the timed region. The scalar loss
reduction forces the forward graph, and gradient accumulation into the
leaves forces every backward branch, so eager and fused iterations do the
same work; mode is the only behavioral difference between paired calls.
Device completion is owned by the timer's synchronize callback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

import rushlite

from benchmarks.common.timer import measure_samples

from .spec import FusionCase, Mode, RunConfig, resolve_cases
from .workloads.base import CellStates, CellWorkload, PreparedCell

# forward tolerances cover the composite exp-based activations in float32;
# gradient tolerances are looser because error compounds through the
# backward graph
FORWARD_RTOL = 1e-4
FORWARD_ATOL = 1e-6
GRAD_RTOL = 1e-3
GRAD_ATOL = 1e-5


class CorrectnessError(AssertionError):
    """Eager and fused training computations disagreed on a case."""


@dataclass(frozen=True)
class StepArtifacts:
    states: CellStates
    loss: rushlite.Variable


@dataclass(frozen=True)
class ResultRow:
    workload: str
    mode: Mode
    device: str
    dtype: str
    input_size: int
    hidden_size: int
    batch_size: int
    sequence_steps: int
    tag: str
    seed: int
    iterations_per_sample: int
    samples: int
    median_us: float
    min_us: float


def squared_error(
    prediction: rushlite.Variable, target: rushlite.Variable
) -> rushlite.Variable:
    diff = prediction - target
    return (diff * diff).sum(1).sum(0)


def variable_array(variable: rushlite.Variable) -> np.ndarray:
    """Realize a Variable's data on host (never inside timing)."""
    return np.asarray(variable.tolist(), dtype=np.float64).reshape(variable.shape)


def grad_array(variable: rushlite.Variable) -> np.ndarray:
    """Realize a Variable's gradient on host (never inside timing)."""
    grad = variable.grad
    return np.asarray(grad.tolist(), dtype=np.float64).reshape(grad.shape)


def _compare(
    label: str,
    eager: np.ndarray,
    fused: np.ndarray,
    *,
    rtol: float,
    atol: float,
    errors: list[str],
) -> None:
    if eager.shape != fused.shape:
        errors.append(
            f"{label}: shape mismatch eager={eager.shape} fused={fused.shape}"
        )
        return
    if not np.allclose(eager, fused, rtol=rtol, atol=atol):
        worst = float(np.max(np.abs(eager - fused)))
        errors.append(f"{label}: max abs difference {worst:.3e}")


class FusionRunner:
    def __init__(
        self, config: RunConfig, workloads: Mapping[str, CellWorkload]
    ) -> None:
        self.config = config
        self.workloads = dict(workloads)

    def run(self) -> list[ResultRow]:
        rows: list[ResultRow] = []
        for case in resolve_cases(self.config):
            workload = self.workloads[case.workload]
            self.validate_case(case, workload)
            for mode in self.config.modes:
                rows.append(self.measure_case(case, workload, mode))
        return rows

    def run_training_step(
        self, workload: CellWorkload, prepared: PreparedCell, mode: Mode
    ) -> StepArtifacts:
        prepared.zero_grad()
        with rushlite.capture_on(enable=mode == "fused"):
            states = workload.forward(prepared)
            loss = squared_error(states.hidden, prepared.target)
            loss.backward()
        return StepArtifacts(states=states, loss=loss)

    def validate_case(self, case: FusionCase, workload: CellWorkload) -> None:
        """Reject the case unless eager and fused agree on loss, states, and
        every named gradient."""
        results: dict[Mode, tuple[PreparedCell, StepArtifacts]] = {}
        for mode in ("eager", "fused"):
            prepared = workload.prepare(case, self.config.seed)
            artifacts = self.run_training_step(workload, prepared, mode)
            results[mode] = (prepared, artifacts)
        rushlite.cuda.sync()

        eager_prepared, eager_step = results["eager"]
        fused_prepared, fused_step = results["fused"]

        errors: list[str] = []
        _compare(
            "loss",
            variable_array(eager_step.loss),
            variable_array(fused_step.loss),
            rtol=FORWARD_RTOL,
            atol=FORWARD_ATOL,
            errors=errors,
        )
        eager_states = eager_step.states.named_states()
        fused_states = fused_step.states.named_states()
        for name in eager_states:
            _compare(
                name,
                variable_array(eager_states[name]),
                variable_array(fused_states[name]),
                rtol=FORWARD_RTOL,
                atol=FORWARD_ATOL,
                errors=errors,
            )
        eager_leaves = eager_prepared.grad_leaves()
        fused_leaves = fused_prepared.grad_leaves()
        for name in eager_leaves:
            _compare(
                f"grad[{name}]",
                grad_array(eager_leaves[name]),
                grad_array(fused_leaves[name]),
                rtol=GRAD_RTOL,
                atol=GRAD_ATOL,
                errors=errors,
            )

        if errors:
            raise CorrectnessError(
                f"eager/fused mismatch for {case}: " + "; ".join(errors)
            )

    def measure_case(
        self, case: FusionCase, workload: CellWorkload, mode: Mode
    ) -> ResultRow:
        prepared = workload.prepare(case, self.config.seed)
        run_once = self._make_run_once(workload, prepared, mode)

        measurement = measure_samples(
            run_once,
            rushlite.cuda.sync,
            warmup=self.config.warmup,
            samples=self.config.samples,
            iterations=self.config.iterations,
        )

        return ResultRow(
            workload=case.workload,
            mode=mode,
            device=case.device,
            dtype=case.dtype,
            input_size=case.input_size,
            hidden_size=case.hidden_size,
            batch_size=case.batch_size,
            sequence_steps=1,
            tag=case.tag,
            seed=self.config.seed,
            iterations_per_sample=measurement.iterations_per_sample,
            samples=len(measurement.samples_us),
            median_us=measurement.median_us,
            min_us=measurement.min_us,
        )

    def _make_run_once(
        self, workload: CellWorkload, prepared: PreparedCell, mode: Mode
    ) -> Callable[[], None]:
        def run_once() -> None:
            self.run_training_step(workload, prepared, mode)

        return run_once
