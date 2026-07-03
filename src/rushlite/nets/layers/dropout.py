from rushlite import _C
from rushlite._C import Variable

from ..Module import Module


class Dropout(Module):
    def __init__(self, p: float) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Variable) -> Variable:
        mask = _C.rand(x.shape)
        return (mask < self.p) * x
