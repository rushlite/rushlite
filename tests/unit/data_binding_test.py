import pytest

import rushlite as rsl


class TestTensorOperators:
    def test_binary_ops(self):
        a = rsl.Variable([1.0, 2.0, 3.0]).data
        b = rsl.Variable([4.0, 10.0, 18.0]).data
        assert (a + b).tolist() == pytest.approx([5.0, 12.0, 21.0])
        assert (b - a).tolist() == pytest.approx([3.0, 8.0, 15.0])
        assert (a * b).tolist() == pytest.approx([4.0, 20.0, 54.0])
        assert (b / a).tolist() == pytest.approx([4.0, 5.0, 6.0])

    def test_scalar_ops(self):
        a = rsl.Variable([1.0, 2.0, 3.0]).data
        assert (a * 2.0).tolist() == pytest.approx([2.0, 4.0, 6.0])
        assert (2.0 * a).tolist() == pytest.approx([2.0, 4.0, 6.0])
        assert (a - 1.0).tolist() == pytest.approx([0.0, 1.0, 2.0])
        assert (1.0 - a).tolist() == pytest.approx([0.0, -1.0, -2.0])
        assert (-a).tolist() == pytest.approx([-1.0, -2.0, -3.0])

    def test_inplace_ops_write_through_shared_storage(self):
        w = rsl.Variable([1.0, 2.0])
        alias = w.data
        alias += 1.0
        assert w.tolist() == pytest.approx([2.0, 3.0])
        alias -= rsl.Variable([1.0, 1.0]).data
        assert w.tolist() == pytest.approx([1.0, 2.0])

    def test_inplace_op_rejects_broadcast(self):
        a = rsl.Variable([1.0]).data
        b = rsl.Variable([1.0, 2.0, 3.0]).data
        with pytest.raises(ValueError):
            a += b


class TestVariableDataSetter:
    def test_data_is_writable(self):
        w = rsl.Variable([1.0, 2.0], requires_grad=True)
        w.data = rsl.Variable([5.0, 6.0]).data
        assert w.tolist() == pytest.approx([5.0, 6.0])

    def test_data_setter_rejects_shape_mismatch(self):
        w = rsl.Variable([1.0, 2.0], requires_grad=True)
        with pytest.raises(RuntimeError):
            w.data = rsl.Variable([1.0, 2.0, 3.0]).data

    def test_sgd_step(self):
        w = rsl.Variable([1.0, 2.0, 3.0], requires_grad=True)
        loss = (w * w).sum(0)
        loss.backward()
        assert w.grad.tolist() == pytest.approx([2.0, 4.0, 6.0])

        w.data -= 0.5 * w.grad
        assert w.tolist() == pytest.approx([0.0, 0.0, 0.0])


class TestModuleParameterUpdates:
    class TinyModel(rsl.Module):
        def __init__(self):
            super().__init__()
            self.w = rsl.Variable([1.0, 2.0, 3.0], requires_grad=True)

        def forward(self, x):
            return (x * self.w).sum(0)

    def test_named_parameters_update_is_visible_to_model(self):
        model = self.TinyModel()
        x = rsl.Variable([1.0, 1.0, 1.0])
        loss = model(x)
        loss.backward()

        for _, p in model.named_parameters():
            p.data -= 0.5 * p.grad

        assert model.w.tolist() == pytest.approx([0.5, 1.5, 2.5])

    def test_parameters_update_is_visible_to_model(self):
        model = self.TinyModel()
        x = rsl.Variable([2.0, 2.0, 2.0])
        loss = model(x)
        loss.backward()

        for p in model.parameters():
            p.data -= 1.0 * p.grad

        assert model.w.tolist() == pytest.approx([-1.0, 0.0, 1.0])
