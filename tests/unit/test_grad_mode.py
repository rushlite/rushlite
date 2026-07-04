"""Tests for rushlite.no_grad and the underlying grad-mode toggles.

no_grad gates op *recording* only: ops run inside come out with
requires_grad=False and no grad_fn, while explicit leaf creation and
backward() over a pre-built graph are unaffected.
"""

import pytest
import rushlite as rsl
from rushlite._C import is_grad_enabled, set_grad_enabled


@pytest.fixture(autouse=True)
def _grad_enabled_by_default():
    assert is_grad_enabled()
    yield
    set_grad_enabled(True)


def _var(data, requires_grad=False):
    return rsl.Variable(data, requires_grad=requires_grad)


def test_op_inside_no_grad_does_not_require_grad():
    a = _var([1.0, 2.0], requires_grad=True)
    b = _var([3.0, 4.0], requires_grad=True)
    with rsl.no_grad():
        c = a * b
    assert not c.requires_grad
    assert c.tolist() == [3.0, 8.0]


def test_op_outside_no_grad_records():
    a = _var([1.0, 2.0], requires_grad=True)
    c = a * 2.0
    assert c.requires_grad


def test_explicit_leaf_creation_honored_inside_no_grad():
    with rsl.no_grad():
        leaf = _var([1.0], requires_grad=True)
    assert leaf.requires_grad


def test_backward_of_prebuilt_graph_works_inside_no_grad():
    a = _var([1.0, 2.0], requires_grad=True)
    b = _var([3.0, 4.0], requires_grad=True)
    loss = (a * b).sum(0)
    with rsl.no_grad():
        loss.backward()
    assert a.grad.tolist() == [3.0, 4.0]
    assert b.grad.tolist() == [1.0, 2.0]


def test_no_grad_restores_previous_mode():
    with rsl.no_grad():
        assert not is_grad_enabled()
        with rsl.no_grad():
            assert not is_grad_enabled()
        assert not is_grad_enabled()
    assert is_grad_enabled()


def test_no_grad_restores_on_exception():
    with pytest.raises(RuntimeError):
        with rsl.no_grad():
            raise RuntimeError("boom")
    assert is_grad_enabled()


def test_no_grad_as_decorator():
    @rsl.no_grad()
    def fn(a, b):
        return a + b

    a = _var([1.0], requires_grad=True)
    out = fn(a, a)
    assert not out.requires_grad
    assert is_grad_enabled()
