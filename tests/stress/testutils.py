import numpy as np
import torch
import rushlite

MATRIX_DIMS = (16, 16)


def sample_from_intervals(intervals, shape):
    m = len(intervals)
    n = np.prod(shape)

    def interval_size(interval):
        assert interval[1] >= interval[0]
        return interval[1] - interval[0]

    tot_sz = sum([interval_size(i) for i in intervals])
    p = [interval_size(i) / tot_sz for i in intervals]

    interval_idx = np.random.choice(m, size=n, p=p)
    out = np.empty(n)

    for i, (lo, hi) in enumerate(intervals):
        mask = interval_idx == i
        out[mask] = np.random.uniform(lo, hi, mask.sum())
    return out.reshape(shape).tolist()


def sample_matrices(n, ranges, shape=MATRIX_DIMS):
    return [sample_from_intervals(ranges, shape) for _ in range(n)]


def from_row_major(flat, like):
    t = torch.tensor(flat).reshape(torch.tensor(like).shape)
    return t.tolist()


def to_rushlite_var(mat, device, requires_grad=True):
    return rushlite.Variable(
        mat, requires_grad=requires_grad, device=device, dtype=rushlite.dtype.float64
    )


def _atol(pred, true):
    return float(torch.max(torch.abs(torch.tensor(pred) - torch.tensor(true))))


def _rtol(pred, true, epsilon):
    return float(torch.max(torch.abs(torch.tensor(pred) - torch.tensor(true)))) / (
        float(torch.max(torch.tensor(true))) + epsilon
    )


def calculate_pair_tolerances(cg, tg, epsilon):
    return _atol(cg, tg), _rtol(cg, tg, epsilon)
