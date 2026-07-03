import sys

from rushlite import _C
from rushlite._C import Variable

from ..Module import Module


class ReLU(Module):
    def forward(self, x: Variable) -> Variable:
        return _C.clamp(x, 0, sys.float_info.max)


class Sigmoid(Module):
    def forward(self, x: Variable) -> Variable:
        return 1 / (1 + _C.exp(-x))


class Tanh(Module):
    def forward(self, x: Variable) -> Variable:
        exp_x = _C.exp(x)
        nexp_x = _C.exp(-x)
        return (exp_x - nexp_x) / (exp_x + nexp_x)


class Softmax(Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Variable) -> Variable:
        dim_idx = x.ndim + self.dim if self.dim < 0 else self.dim
        if not 0 <= dim_idx < x.ndim:
            raise ValueError("Dim not in bounds")

        max_vals = _C.max(x, dim_idx)
        x_shifted = x - max_vals

        exp_x = _C.exp(x_shifted)
        return exp_x / _C.sum(exp_x, dim_idx)
