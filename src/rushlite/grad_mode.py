from contextlib import contextmanager
from rushlite._C import set_grad_enabled, is_grad_enabled


@contextmanager
def no_grad():
    prev = is_grad_enabled()
    try:
        set_grad_enabled(False)
        yield
    finally:
        set_grad_enabled(prev)
