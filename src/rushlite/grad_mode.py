from contextlib import contextmanager
from rushlite._C import set_grad_enabled, is_grad_enabled


@contextmanager
def no_grad():
    """Disable gradient recording for ops run inside the block.

    Only op *recording* is gated: results of ops come out with
    requires_grad=False and no grad_fn. Explicitly constructing a leaf with
    Variable(data, requires_grad=True) still honors the flag, and calling
    backward() on a graph built outside the block still works.

    Usable as a context manager (`with no_grad():`) or a decorator
    (`@no_grad()`), since @contextmanager objects double as decorators.
    """
    prev = is_grad_enabled()
    try:
        set_grad_enabled(False)
        yield
    finally:
        set_grad_enabled(prev)
