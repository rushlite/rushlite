"""Shared workload plumbing: prepared leaves, states, and activations.

Workloads own data preparation and cell equations only. Capture, timing,
synchronization, correctness policy, and reporting live in the runner.

Rushlite has no native sigmoid/tanh, so both are composed from exp:

    sigmoid(x) = 1 / (1 + exp(-x))
    tanh(x)    = 2 * sigmoid(2 * x) - 1

The literals 1 and 2 are materialized as [batch, hidden] constant tensors
(PreparedCell.constants) instead of Python scalars: a scalar operand becomes
a shape-[1] tensor, and ElementwiseBinaryFn::is_fusible() requires equal
input shapes, so every scalar op would fall back to eager and break the
fused graph apart. Same-shape constants keep the math identical and every
pointwise op fusible, and they also avoid the fused code generator's known
broadcast-leaf indexing limitation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

import rushlite

from ..spec import FusionCase

# input and parameter values are small and uniform so the composite
# activations stay far from saturation and exp() cannot overflow
INIT_SCALE = 0.1

_DEVICES = {
    "cpu": rushlite.device.cpu,
    "cuda": rushlite.device.cuda,
}

_DTYPES = {
    "float32": rushlite.dtype.float32,
    "float64": rushlite.dtype.float64,
}


def make_variable(
    values: np.ndarray, *, device: str, dtype: str, requires_grad: bool
) -> rushlite.Variable:
    """Build a rushlite Variable from host values (outside any timing)."""
    return rushlite.Variable(
        values.tolist(),
        requires_grad=requires_grad,
        device=_DEVICES[device],
        dtype=_DTYPES[dtype],
    )


@dataclass
class PreparedCell:
    """Leaves for one case: everything a training step reads.

    inputs and parameters require gradients; constants and target do not.
    """

    inputs: dict[str, rushlite.Variable]
    parameters: dict[str, rushlite.Variable]
    target: rushlite.Variable
    constants: dict[str, rushlite.Variable] = field(default_factory=dict)

    def grad_leaves(self) -> dict[str, rushlite.Variable]:
        return {**self.inputs, **self.parameters}

    def zero_grad(self) -> None:
        for leaf in self.grad_leaves().values():
            leaf.zero_grad()


@dataclass(frozen=True)
class CellStates:
    hidden: rushlite.Variable
    cell: rushlite.Variable | None = None

    def named_states(self) -> dict[str, rushlite.Variable]:
        states = {"h_t": self.hidden}
        if self.cell is not None:
            states["c_t"] = self.cell
        return states


class CellWorkload(Protocol):
    name: str

    def prepare(self, case: FusionCase, seed: int) -> PreparedCell: ...

    def forward(self, prepared: PreparedCell) -> CellStates: ...


def sigmoid(x: rushlite.Variable, one: rushlite.Variable) -> rushlite.Variable:
    return one / (one + (-x).exp())


def tanh(
    x: rushlite.Variable, one: rushlite.Variable, two: rushlite.Variable
) -> rushlite.Variable:
    return two * sigmoid(two * x, one) - one


def uniform_values(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    return rng.uniform(-INIT_SCALE, INIT_SCALE, size=shape).astype(np.float32)


def prepare_from_arrays(
    case: FusionCase,
    *,
    inputs: dict[str, np.ndarray],
    parameters: dict[str, np.ndarray],
    target: np.ndarray,
) -> PreparedCell:
    """Turn seeded host arrays into a PreparedCell on the case's device."""

    def build(values: np.ndarray, requires_grad: bool) -> rushlite.Variable:
        return make_variable(
            values,
            device=case.device,
            dtype=case.dtype,
            requires_grad=requires_grad,
        )

    state_shape = (case.batch_size, case.hidden_size)
    return PreparedCell(
        inputs={k: build(v, True) for k, v in inputs.items()},
        parameters={k: build(v, True) for k, v in parameters.items()},
        target=build(target, False),
        constants={
            "one": build(np.ones(state_shape, dtype=np.float32), False),
            "two": build(np.full(state_shape, 2.0, dtype=np.float32), False),
        },
    )
