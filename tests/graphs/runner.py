"""Build a template in torch and rushlite from one fill, then diff fwd + bwd.

Mirrors the comparison the single-op stress suite does, generalized to an
N-input graph driven by a template ``build`` function. Tolerances scale with
chain depth since float error compounds down a chain.
"""

import torch
import rushlite

from ops import OpSet, row_normalize, BACKEND_TORCH, BACKEND_LAMP

EPSILON = 1e-10
TORCH_DTYPE = torch.float64

# Base tolerances, multiplied by chain depth; backward is allowed to be looser.
BASE_ATOL = 1e-5
BASE_RTOL = 1e-4
BACKWARD_MULT = 2.0


def _atol(pred, true):
    return float(torch.max(torch.abs(torch.tensor(pred) - torch.tensor(true))))


def _rtol(pred, true):
    return float(torch.max(torch.abs(torch.tensor(pred) - torch.tensor(true)))) / (
        float(torch.max(torch.tensor(true))) + EPSILON
    )


def _from_row_major(flat, like):
    """Reshape rushlite's flat (row-major) output to match a torch-shaped value."""
    return torch.tensor(flat).reshape(torch.tensor(like).shape).tolist()


def _to_lamp(arr, device):
    return rushlite.Variable(
        arr.tolist(), requires_grad=True, device=device, dtype=rushlite.dtype.float64
    )


def run_once(template, fills, device, rng):
    """Draw inputs, run both backends, assert forward value + per-input grads match."""
    arrs = [row_normalize(rng.uniform(-1.0, 1.0, s)) for s in template.input_shapes]

    torch_vars = [torch.tensor(a, dtype=TORCH_DTYPE, requires_grad=True) for a in arrs]
    torch_out = template.build(OpSet(BACKEND_TORCH, fills), torch_vars)
    torch_out.backward(torch.ones_like(torch_out))

    lamp_vars = [_to_lamp(a, device) for a in arrs]
    lamp_out = template.build(OpSet(BACKEND_LAMP, fills), lamp_vars)
    lamp_out.backward()

    depth = len(template.slots)
    fwd_atol = BASE_ATOL * depth
    fwd_rtol = BASE_RTOL * depth
    bwd_atol = fwd_atol * BACKWARD_MULT
    bwd_rtol = fwd_rtol * BACKWARD_MULT

    true_out = torch_out.detach().tolist()
    pred_out = _from_row_major(lamp_out.tolist(), true_out)
    assert _atol(pred_out, true_out) <= fwd_atol, f"{template.name}: forward atol"
    assert _rtol(pred_out, true_out) <= fwd_rtol, f"{template.name}: forward rtol"

    for i, (tv, lv, a) in enumerate(zip(torch_vars, lamp_vars, arrs)):
        true_grad = tv.grad.tolist()
        pred_grad = _from_row_major(lv.grad.tolist(), a.tolist())
        assert (
            _atol(pred_grad, true_grad) <= bwd_atol
        ), f"{template.name}: grad[{i}] atol"
        assert (
            _rtol(pred_grad, true_grad) <= bwd_rtol
        ), f"{template.name}: grad[{i}] rtol"
