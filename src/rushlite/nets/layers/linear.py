from rushlite import _C
from rushlite._C import Variable

from ..Module import Module


class Linear(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: "_C.device | None" = None,
        dtype: "_C.dtype | None" = None,
    ) -> None:
        super().__init__()
        # None falls through to the C++ DEFAULT_DEVICE/DEFAULT_DTYPE
        kwargs = {}
        if device is not None:
            kwargs["device"] = device
        if dtype is not None:
            kwargs["dtype"] = dtype

        self.requires_bias = bias
        self.weights = _C.randn(
            0, 1, [in_features, out_features], requires_grad=True, **kwargs
        )
        if bias:
            self.bias = _C.randn(0, 1, [out_features], requires_grad=True, **kwargs)

    def forward(self, x: Variable) -> Variable:
        if not self.requires_bias:  # (batch x in) x (in x out)
            return _C.matmul(x, self.weights)
        return _C.matmul(x, self.weights) + self.bias


class Flatten(Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        if start_dim < 0:
            raise ValueError("Start dim must be non-negative")
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: Variable) -> Variable:
        end_dim_idx = x.ndim + self.end_dim if self.end_dim < 0 else self.end_dim
        if not self.start_dim < end_dim_idx:
            raise ValueError("Start and end dims not valid")

        flattened_dims = 1
        new_shape = []

        for i in range(x.ndim):
            if i < self.start_dim or i > end_dim_idx:
                new_shape.append(x.shape[i])
                continue
            flattened_dims *= x.shape[i]
            if i == end_dim_idx:
                new_shape.append(flattened_dims)

        return _C.reshape(x, new_shape)
