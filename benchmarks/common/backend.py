"""Backend ABC and registry.

A backend module (e.g. benchmarks/backends/torch_backend.py) registers itself
by calling register_backend(MyBackend()) at module level. The runner then
dispatches to BACKENDS by name.
"""

from abc import ABC, abstractmethod
from typing import Callable


class Backend(ABC):
    """Abstract base for a compute backend (torch, rushlite, etc.)."""

    name: str
    ops: dict[str, Callable]  # op_name -> fn(*tensors) -> output tensor

    @abstractmethod
    def make_input(
        self, shape: list[int], device: str, dtype: str, requires_grad: bool
    ):
        """Allocate a random input tensor on the given device/dtype."""
        ...

    @abstractmethod
    def sync(self, device: str) -> None:
        """Block until all pending device work completes. No-op on cpu."""
        ...

    @abstractmethod
    def cuda_available(self) -> bool:
        """Return True if CUDA is accessible through this backend."""
        ...

    @abstractmethod
    def backward(self, out) -> None:
        """Seed grad with ones_like(out) and run backward.

        Accumulating gradients across repeated calls is acceptable for timing;
        correctness of the accumulated grad is not measured here.
        """
        ...


# Module-level registry - populated by register_backend() calls in backend modules.
BACKENDS: dict[str, "Backend"] = {}


def register_backend(b: "Backend") -> None:
    """Add b to BACKENDS keyed by b.name."""
    BACKENDS[b.name] = b
