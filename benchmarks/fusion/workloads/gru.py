"""Textbook GRU cell, reset-before-recurrent-matmul formulation:

    r_t = sigmoid(x_t W_xr + h_prev W_hr + b_r)
    z_t = sigmoid(x_t W_xz + h_prev W_hz + b_z)
    n_t = tanh(x_t W_xn + (r_t * h_prev) W_hn + b_n)
    h_t = (1 - z_t) * n_t + z_t * h_prev

Gates use separate weights; nothing is packed or sliced.
"""

from __future__ import annotations

import numpy as np

from ..spec import FusionCase
from .base import (
    CellStates,
    PreparedCell,
    prepare_from_arrays,
    sigmoid,
    tanh,
    uniform_values,
)

GATES = ("r", "z", "n")


class GRUWorkload:
    name = "gru"

    def prepare(self, case: FusionCase, seed: int) -> PreparedCell:
        rng = np.random.default_rng(seed)
        input_shape = (case.batch_size, case.input_size)
        state_shape = (case.batch_size, case.hidden_size)

        inputs = {
            "x": uniform_values(rng, input_shape),
            "h_prev": uniform_values(rng, state_shape),
        }
        parameters: dict[str, np.ndarray] = {}
        for gate in GATES:
            parameters[f"w_x{gate}"] = uniform_values(
                rng, (case.input_size, case.hidden_size)
            )
            parameters[f"w_h{gate}"] = uniform_values(
                rng, (case.hidden_size, case.hidden_size)
            )
            parameters[f"b_{gate}"] = uniform_values(rng, state_shape)
        target = uniform_values(rng, state_shape)

        return prepare_from_arrays(
            case, inputs=inputs, parameters=parameters, target=target
        )

    def forward(self, prepared: PreparedCell) -> CellStates:
        x = prepared.inputs["x"]
        h_prev = prepared.inputs["h_prev"]
        p = prepared.parameters
        one = prepared.constants["one"]
        two = prepared.constants["two"]

        r = sigmoid(x @ p["w_xr"] + h_prev @ p["w_hr"] + p["b_r"], one)
        z = sigmoid(x @ p["w_xz"] + h_prev @ p["w_hz"] + p["b_z"], one)
        n = tanh(x @ p["w_xn"] + (r * h_prev) @ p["w_hn"] + p["b_n"], one, two)
        h = (one - z) * n + z * h_prev
        return CellStates(hidden=h)
