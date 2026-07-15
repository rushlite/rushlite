"""PyTorch backend - optional reference baseline.

Imported only when the user asks for it (--backends ... torch), so a rushlite-only
run never needs torch installed. Registers itself on import. Inputs use the same
uniform rand [0, 1) recipe as the rushlite backend so the two are comparable.
"""

from __future__ import annotations

import torch

from benchmarks.common.backend import Backend, register_backend

_DTYPES = {
    "float32": torch.float32,
    "float64": torch.float64,
}


class TorchBackend(Backend):
    name = "torch"

    ops = {
        "add": torch.add,
        "sub": torch.sub,
        "mul": torch.mul,
        "div": torch.div,
        "pow": torch.pow,
        "abs": torch.abs,
        "clamp": lambda x: torch.clamp(x, min=0.25, max=0.75),
        "cos": torch.cos,
        "exp": torch.exp,
        "log": torch.log,
        "neg": torch.neg,
        "sin": torch.sin,
        "sqrt": torch.sqrt,
        "tan": torch.tan,
        "sum": lambda x, dim: torch.sum(x, dim),
        "min": lambda x, dim: torch.min(x, dim).values,
        "max": lambda x, dim: torch.max(x, dim).values,
        "prod": lambda x, dim: torch.prod(x, dim),
    }

    def make_input(self, shape, device, dtype, requires_grad):
        return torch.rand(
            list(shape),
            device=device,
            dtype=_DTYPES[dtype],
            requires_grad=requires_grad,
        )

    def sync(self, device):
        if device == "cuda":
            torch.cuda.synchronize()

    def cuda_available(self) -> bool:
        return torch.cuda.is_available()

    def backward(self, out) -> None:
        # Seed grad with ones_like (not out.mean()) to match the rushlite harness,
        # and retain the graph so the same output can be re-differentiated each
        # timed iteration. Grad accumulates across calls, which is fine for timing.
        out.backward(torch.ones_like(out), retain_graph=True)


register_backend(TorchBackend())
