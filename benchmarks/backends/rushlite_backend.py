"""Rushlite backend - the framework under test.

rushlite is this repo's own package, so this backend is always importable. It
registers itself on import. Inputs are uniform rand in [0, 1) (see make_input);
that distribution is timing-equivalent to randn while keeping every op's inputs
valid (log/sqrt/pow stay finite), and both backends use the same recipe so the
comparison stays fair.
"""

from __future__ import annotations

import rushlite

from benchmarks.common.backend import Backend, register_backend

_DTYPES = {
    "float32": rushlite.dtype.float32,
    "float64": rushlite.dtype.float64,
}

_DEVICES = {
    "cpu": rushlite.device.cpu,
    "cuda": rushlite.device.cuda,
}


def _reduction(name: str):
    """Wrap a rushlite reduction so the runner can call it as fn(tensor, axis)."""
    fn = getattr(rushlite, name)
    return lambda x, dim: fn(x, dim)


class RushliteBackend(Backend):
    name = "rushlite"

    ops = {
        "add": rushlite.add,
        "sub": rushlite.sub,
        "mul": rushlite.mul,
        "div": rushlite.div,
        "pow": rushlite.pow,
        "abs": rushlite.abs,
        # clamp bounds are constant across all cases, so bake them in here.
        "clamp": lambda x: rushlite.clamp(x, 0.25, 0.75),
        "cos": rushlite.cos,
        "exp": rushlite.exp,
        "log": rushlite.log,
        "neg": rushlite.neg,
        "sin": rushlite.sin,
        "sqrt": rushlite.sqrt,
        "tan": rushlite.tan,
        "sum": _reduction("sum"),
        "min": _reduction("min"),
        "max": _reduction("max"),
        "prod": _reduction("prod"),
    }

    def make_input(self, shape, device, dtype, requires_grad):
        return rushlite.rand(
            list(shape),
            requires_grad=requires_grad,
            device=_DEVICES[device],
            dtype=_DTYPES[dtype],
        )

    def sync(self, device):
        if device == "cuda":
            rushlite.cuda.sync()

    def cuda_available(self) -> bool:
        return rushlite.cuda.is_available()

    def backward(self, out) -> None:
        # Variable.backward() seeds ones_like(out) internally; repeated calls
        # accumulate grad, which is fine for timing.
        out.backward()


register_backend(RushliteBackend())
