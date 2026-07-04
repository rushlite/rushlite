"""Tests for Module parameter storage and the functional update path.

Parameters live ONLY in _params_dict (single source of truth); attribute
access resolves them via __getattr__, and the ``params`` view supports
functional updates by dotted path, promoting values assigned under
no_grad() back to grad-requiring leaves.
"""

import pytest
import rushlite as rsl
from rushlite.nets import layers


class Scale(rsl.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = rsl.Variable([1.0, 2.0], requires_grad=True)

    def forward(self, x):
        return x * self.w


class Affine(rsl.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = Scale()
        self.b = rsl.Variable([0.5, 0.5], requires_grad=True)

    def forward(self, x):
        return self.scale(x) + self.b


class TestParamStorage:
    def test_params_live_only_in_params_dict(self):
        m = Scale()
        assert "w" not in vars(m)
        assert "_params_dict" in vars(m)
        assert hasattr(m, "w")
        assert m.w is m._params_dict["w"]

    def test_submodules_live_only_in_params_dict(self):
        m = Affine()
        assert "scale" not in vars(m)
        assert m.scale is m._params_dict["scale"]

    def test_forward_reads_params_dict(self):
        m = Scale()
        m._params_dict["w"] = rsl.Variable([10.0, 10.0], requires_grad=True)
        x = rsl.Variable([1.0, 1.0])
        assert m(x).tolist() == [10.0, 10.0]

    def test_non_variable_assignment_evicts_shadowed_param(self):
        m = Scale()
        assert len(list(m.parameters())) == 1
        m.w = 3
        assert list(m.parameters()) == []
        assert m.w == 3
        assert vars(m)["w"] == 3

    def test_delattr_removes_param(self):
        m = Scale()
        del m.w
        assert not hasattr(m, "w")
        assert list(m.parameters()) == []

    def test_delattr_falls_through_for_plain_attributes(self):
        m = Scale()
        m.tag = "x"
        del m.tag
        assert not hasattr(m, "tag")
        with pytest.raises(AttributeError):
            del m.missing

    def test_missing_attribute_raises(self):
        m = Scale()
        with pytest.raises(AttributeError, match="nope"):
            m.nope

    def test_assign_param_before_init_raises(self):
        class Bad(rsl.Module):
            def __init__(self):
                self.w = rsl.Variable([1.0], requires_grad=True)

            def forward(self):
                pass

        with pytest.raises(AttributeError, match=r"super\(\).__init__\(\)"):
            Bad()

    def test_access_param_before_init_raises(self):
        class Empty(rsl.Module):
            def __init__(self):
                pass

            def forward(self):
                pass

        with pytest.raises(AttributeError, match=r"super\(\).__init__\(\)"):
            Empty().w

    def test_plain_setattr_does_not_promote(self):
        m = Scale()
        m.w = rsl.Variable([1.0, 1.0], requires_grad=False)
        assert not m.w.requires_grad


class TestParamsView:
    def test_getitem_by_dotted_path(self):
        m = Affine()
        assert m.params["b"] is m.b
        assert m.params["scale.w"] is m.scale.w

    def test_iteration_matches_named_parameters(self):
        m = Affine()
        assert list(m.params) == [name for name, _ in m.named_parameters()]
        assert list(m.params.keys()) == ["scale.w", "b"]
        assert [name for name, _ in m.params.items()] == ["scale.w", "b"]
        assert len(m.params) == 2
        assert "scale.w" in m.params
        assert "scale.nope" not in m.params

    def test_getitem_missing_raises_keyerror(self):
        m = Affine()
        with pytest.raises(KeyError):
            m.params["nope"]
        with pytest.raises(KeyError):
            m.params["nope.w"]
        with pytest.raises(KeyError):
            m.params["b.w"]  # 'b' is a Variable, not a submodule

    def test_setitem_rejects_non_variable(self):
        m = Scale()
        with pytest.raises(TypeError):
            m.params["w"] = 5

    def test_setitem_assigns_on_owning_module(self):
        m = Affine()
        new_w = rsl.Variable([7.0, 7.0], requires_grad=True)
        m.params["scale.w"] = new_w
        assert m.scale.w is new_w
        x = rsl.Variable([1.0, 1.0])
        assert m(x).tolist() == [7.5, 7.5]

    def test_setitem_promotes_to_requires_grad(self):
        m = Scale()
        m.params["w"] = rsl.Variable([3.0, 3.0], requires_grad=False)
        assert m.w.requires_grad
        assert m.w.tolist() == [3.0, 3.0]


class TestFunctionalUpdates:
    def _loss(self, model, x, y):
        err = model(x) - y
        return (err * err).sum(0)

    def test_update_spelling_a_no_grad(self):
        model = Scale()
        x = rsl.Variable([1.0, 1.0])
        y = rsl.Variable([3.0, 5.0])
        lr = 0.1

        self._loss(model, x, y).backward()
        p = model.params["w"]
        with rsl.no_grad():
            model.params["w"] = p - lr * rsl.Variable(p.grad)

        new_p = model.params["w"]
        assert new_p is not p
        assert new_p.requires_grad
        # w=[1,2], dL/dw = 2*(w-y)*x = [-4,-6]; w' = w + 0.1*[4,6]
        assert new_p.tolist() == pytest.approx([1.4, 2.6])

    def test_update_spelling_b_pure_tensor(self):
        model = Scale()
        x = rsl.Variable([1.0, 1.0])
        y = rsl.Variable([3.0, 5.0])
        lr = 0.1

        self._loss(model, x, y).backward()
        p = model.params["w"]
        model.params["w"] = rsl.Variable(p.data - lr * p.grad, requires_grad=True)

        new_p = model.params["w"]
        assert new_p.requires_grad
        assert new_p.tolist() == pytest.approx([1.4, 2.6])

    def test_training_loop_decreases_loss(self):
        model = Affine()
        x = rsl.Variable([1.0, 1.0])
        y = rsl.Variable([3.0, 5.0])
        lr = 0.05

        losses = []
        for _ in range(10):
            loss = self._loss(model, x, y)
            loss.backward()
            losses.append(loss.tolist()[0])
            with rsl.no_grad():
                for name, p in list(model.named_parameters()):
                    model.params[name] = p - lr * rsl.Variable(p.grad)
        assert losses[-1] < losses[0]

    def test_zero_grad(self):
        model = Scale()
        x = rsl.Variable([1.0, 1.0])
        y = rsl.Variable([3.0, 5.0])
        self._loss(model, x, y).backward()
        assert any(v != 0.0 for v in model.w.grad.tolist())
        model.zero_grad()
        assert model.w.grad.tolist() == [0.0, 0.0]


class TestNetsLayers:
    def test_sequential_named_parameters_and_update(self):
        model = layers.Sequential([layers.Linear(2, 2), layers.ReLU()])
        names = [name for name, _ in model.named_parameters()]
        assert names == ["0.weights", "0.bias"]

        new_bias = rsl.Variable([1.0, 2.0], requires_grad=True)
        model.params["0.bias"] = new_bias
        assert model.layers[0].bias is new_bias

    def test_linear_forward_uses_params_dict(self):
        lin = layers.Linear(2, 2, bias=False)
        assert "weights" not in vars(lin)
        eye = rsl.Variable([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
        lin.params["weights"] = eye
        x = rsl.Variable([[3.0, 4.0]])
        assert lin(x).tolist() == [3.0, 4.0]
