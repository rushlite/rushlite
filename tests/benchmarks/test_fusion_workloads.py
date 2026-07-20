"""Workload tests: shapes, parameter names, cell equations against a NumPy
reference, and the eager/fused correctness gate. CUDA only; skipped when no
device is available."""

import numpy as np
import pytest

import rushlite

from benchmarks.fusion.runner import (
    CorrectnessError,
    FusionRunner,
    variable_array,
    grad_array,
)
from benchmarks.fusion.spec import FusionCase, RunConfig
from benchmarks.fusion.workloads import GRUWorkload, LSTMWorkload

pytestmark = pytest.mark.skipif(
    not rushlite.cuda.is_available(), reason="CUDA is not available"
)

WORKLOADS = {"gru": GRUWorkload(), "lstm": LSTMWorkload()}


def small_case(workload: str, hidden: int = 16) -> FusionCase:
    return FusionCase(
        workload=workload,
        input_size=hidden,
        hidden_size=hidden,
        batch_size=1,
        device="cuda",
        dtype="float32",
        tag="short",
    )


def np_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def leaves_as_arrays(prepared) -> dict[str, np.ndarray]:
    return {
        name: variable_array(v)
        for name, v in {**prepared.inputs, **prepared.parameters}.items()
    }


# ---------------------------------------------------------------------------
# shapes and names
# ---------------------------------------------------------------------------


def test_gru_shapes_and_parameter_names():
    case = small_case("gru")
    prepared = WORKLOADS["gru"].prepare(case, seed=0)

    assert set(prepared.inputs) == {"x", "h_prev"}
    assert set(prepared.parameters) == {
        "w_xr",
        "w_hr",
        "b_r",
        "w_xz",
        "w_hz",
        "b_z",
        "w_xn",
        "w_hn",
        "b_n",
    }
    assert list(prepared.inputs["x"].shape) == [1, 16]
    assert list(prepared.parameters["w_xr"].shape) == [16, 16]
    assert list(prepared.parameters["b_r"].shape) == [1, 16]
    assert list(prepared.target.shape) == [1, 16]

    assert all(v.requires_grad for v in prepared.grad_leaves().values())
    assert not prepared.target.requires_grad
    assert not any(v.requires_grad for v in prepared.constants.values())

    states = WORKLOADS["gru"].forward(prepared)
    assert list(states.hidden.shape) == [1, 16]
    assert states.cell is None
    assert list(states.named_states()) == ["h_t"]


def test_lstm_shapes_and_parameter_names():
    case = small_case("lstm")
    prepared = WORKLOADS["lstm"].prepare(case, seed=0)

    assert set(prepared.inputs) == {"x", "h_prev", "c_prev"}
    assert set(prepared.parameters) == {
        "w_xi",
        "w_hi",
        "b_i",
        "w_xf",
        "w_hf",
        "b_f",
        "w_xg",
        "w_hg",
        "b_g",
        "w_xo",
        "w_ho",
        "b_o",
    }

    states = WORKLOADS["lstm"].forward(prepared)
    assert list(states.hidden.shape) == [1, 16]
    assert list(states.cell.shape) == [1, 16]
    assert list(states.named_states()) == ["h_t", "c_t"]


def test_prepare_is_deterministic_per_seed():
    case = small_case("gru")
    a = WORKLOADS["gru"].prepare(case, seed=7)
    b = WORKLOADS["gru"].prepare(case, seed=7)
    c = WORKLOADS["gru"].prepare(case, seed=8)

    np.testing.assert_array_equal(
        variable_array(a.inputs["x"]), variable_array(b.inputs["x"])
    )
    assert not np.array_equal(
        variable_array(a.inputs["x"]), variable_array(c.inputs["x"])
    )


# ---------------------------------------------------------------------------
# cell equations vs NumPy reference
# ---------------------------------------------------------------------------


def test_gru_equations_match_numpy():
    case = small_case("gru")
    prepared = WORKLOADS["gru"].prepare(case, seed=1)
    states = WORKLOADS["gru"].forward(prepared)

    v = leaves_as_arrays(prepared)
    r = np_sigmoid(v["x"] @ v["w_xr"] + v["h_prev"] @ v["w_hr"] + v["b_r"])
    z = np_sigmoid(v["x"] @ v["w_xz"] + v["h_prev"] @ v["w_hz"] + v["b_z"])
    n = np.tanh(v["x"] @ v["w_xn"] + (r * v["h_prev"]) @ v["w_hn"] + v["b_n"])
    h = (1.0 - z) * n + z * v["h_prev"]

    np.testing.assert_allclose(variable_array(states.hidden), h, rtol=1e-4, atol=1e-5)


def test_lstm_equations_match_numpy():
    case = small_case("lstm")
    prepared = WORKLOADS["lstm"].prepare(case, seed=1)
    states = WORKLOADS["lstm"].forward(prepared)

    v = leaves_as_arrays(prepared)
    i = np_sigmoid(v["x"] @ v["w_xi"] + v["h_prev"] @ v["w_hi"] + v["b_i"])
    f = np_sigmoid(v["x"] @ v["w_xf"] + v["h_prev"] @ v["w_hf"] + v["b_f"])
    g = np.tanh(v["x"] @ v["w_xg"] + v["h_prev"] @ v["w_hg"] + v["b_g"])
    o = np_sigmoid(v["x"] @ v["w_xo"] + v["h_prev"] @ v["w_ho"] + v["b_o"])
    c = f * v["c_prev"] + i * g
    h = o * np.tanh(c)

    np.testing.assert_allclose(variable_array(states.cell), c, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(variable_array(states.hidden), h, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# eager/fused correctness gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("workload", ["gru", "lstm"])
def test_eager_and_fused_agree(workload):
    runner = FusionRunner(RunConfig(seed=3), WORKLOADS)
    runner.validate_case(small_case(workload, hidden=32), WORKLOADS[workload])


@pytest.mark.parametrize("workload", ["gru", "lstm"])
def test_training_step_populates_all_gradients(workload):
    runner = FusionRunner(RunConfig(seed=3), WORKLOADS)
    case = small_case(workload)
    prepared = WORKLOADS[workload].prepare(case, seed=3)
    runner.run_training_step(WORKLOADS[workload], prepared, "fused")
    rushlite.cuda.sync()

    for name, leaf in prepared.grad_leaves().items():
        grad = grad_array(leaf)
        assert grad.shape == tuple(leaf.shape)
        assert np.any(grad != 0.0), f"gradient for {name} is all zeros"


def test_validate_case_detects_divergence():
    """The gate must actually fail when eager and fused disagree."""

    class BrokenWorkload(GRUWorkload):
        def forward(self, prepared):
            states = super().forward(prepared)
            if rushlite._C.is_capture_enabled():
                two = prepared.constants["two"]
                return type(states)(hidden=states.hidden * two)
            return states

    runner = FusionRunner(RunConfig(seed=0), WORKLOADS)
    with pytest.raises(CorrectnessError):
        runner.validate_case(small_case("gru"), BrokenWorkload())
