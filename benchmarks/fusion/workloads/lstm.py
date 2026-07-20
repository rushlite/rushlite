"""Textbook LSTM cell, conventional four-gate formulation:

    i_t = sigmoid(x_t W_xi + h_prev W_hi + b_i)
    f_t = sigmoid(x_t W_xf + h_prev W_hf + b_f)
    g_t = tanh(x_t W_xg + h_prev W_hg + b_g)
    o_t = sigmoid(x_t W_xo + h_prev W_ho + b_o)
    c_t = f_t * c_prev + i_t * g_t
    h_t = o_t * tanh(c_t)

Gates use separate weights; nothing is packed or sliced. Both c_t and h_t
are returned so the runner can validate them.
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

GATES = ("i", "f", "g", "o")


class LSTMWorkload:
    name = "lstm"

    def prepare(self, case: FusionCase, seed: int) -> PreparedCell:
        rng = np.random.default_rng(seed)
        input_shape = (case.batch_size, case.input_size)
        state_shape = (case.batch_size, case.hidden_size)

        inputs = {
            "x": uniform_values(rng, input_shape),
            "h_prev": uniform_values(rng, state_shape),
            "c_prev": uniform_values(rng, state_shape),
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
        c_prev = prepared.inputs["c_prev"]
        p = prepared.parameters
        one = prepared.constants["one"]
        two = prepared.constants["two"]

        i = sigmoid(x @ p["w_xi"] + h_prev @ p["w_hi"] + p["b_i"], one)
        f = sigmoid(x @ p["w_xf"] + h_prev @ p["w_hf"] + p["b_f"], one)
        g = tanh(x @ p["w_xg"] + h_prev @ p["w_hg"] + p["b_g"], one, two)
        o = sigmoid(x @ p["w_xo"] + h_prev @ p["w_ho"] + p["b_o"], one)
        c = f * c_prev + i * g
        h = o * tanh(c, one, two)
        return CellStates(hidden=h, cell=c)
