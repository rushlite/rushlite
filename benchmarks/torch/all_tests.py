"""
python -m pt.all_tests
"""

import operator_benchmark as op_bench

import torch

from benchmark_pytorch import PyTorchOperatorTestCase


# operator_benchmark times backward as `output.mean().backward()`, so its timed
# region includes MeanBackward on top of the op's own backward. The Lamp3
# harness measures a bare `out.backward()` (grad seeded with ones_like), so we
# match it here: seed ones_like(output) once, then backward straight from output
# with no reduction in the timed loop. grad magnitude differs from mean by 1/N,
# which is irrelevant since we only measure time.
def _seed_ones(self):
    self._grad = torch.ones_like(self.output)


def _run_backward_bare(
    self, num_runs, print_per_iter=False, gpu_sync=False, cuda_sync=False
):
    for _ in range(num_runs):
        self.output.backward(self._grad, retain_graph=True)
    if gpu_sync or cuda_sync:
        if hasattr(torch, "accelerator"):
            torch.accelerator.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()


PyTorchOperatorTestCase._output_mean = _seed_ones
PyTorchOperatorTestCase.run_backward = _run_backward_bare

"""Microbenchmarks for all operators."""

cross_product_config = {
    "device": ["cpu", "cuda"],
    "dtype": [torch.float],
    "tags": ["short", "long"],
}


binary_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["add", torch.add],
        ["sub", torch.sub],
        ["div", torch.div],
        ["mul", torch.mul],
        ["pow", torch.pow],
    ],
)

binary_configs_broadcast = op_bench.config_list(
    attr_names=["in_one", "in_two"],
    attrs=[
        [[64, 1, 64], [1, 64, 1]],
    ],
    cross_product_configs=cross_product_config,
    tags=["long"],
)
binary_configs_same_size = op_bench.config_list(
    attr_names=["in_one", "in_two"],
    attrs=[
        [[128, 128], [128, 128]],
        [[256, 256], [256, 256]],
        [[512, 512], [512, 512]],
        [[1024, 1024], [1024, 1024]],
    ],
    cross_product_configs=cross_product_config,
    tags=["long"],
)


class BinaryOpBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, in_one, in_two, dtype, device, op_func):
        self.inputs = {
            "in_one": torch.randn(in_one, device=device, requires_grad=True).to(
                dtype=dtype
            ),
            "in_two": torch.randn(in_two, device=device, requires_grad=True).to(
                dtype=dtype
            ),
        }
        self.op_func = op_func

    def forward(self, in_one, in_two):
        return self.op_func(in_one, in_two)


op_bench.generate_pt_tests_from_op_list(
    binary_ops_list, binary_configs_same_size, BinaryOpBenchmark
)
op_bench.generate_pt_gradient_tests_from_op_list(
    binary_ops_list, binary_configs_same_size, BinaryOpBenchmark
)
op_bench.generate_pt_tests_from_op_list(
    binary_ops_list, binary_configs_broadcast, BinaryOpBenchmark
)
op_bench.generate_pt_gradient_tests_from_op_list(
    binary_ops_list, binary_configs_broadcast, BinaryOpBenchmark
)


# ---

unary_ops_configs = op_bench.config_list(
    attr_names=["in_one"],
    attrs=[[[128, 128]], [[256, 256]], [[512, 512]], [[1024, 1024]]],
    cross_product_configs=cross_product_config,
    tags=["long"],
)


class UnaryOpBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, in_one, device, dtype, op_func):
        self.inputs = {
            "input": torch.rand(in_one, device=device, requires_grad=True).to(
                dtype=dtype
            )
        }
        self.op_func = op_func

    def forward(self, input):
        return self.op_func(input)


def clamp(input):
    return torch.clamp(input, min=0.25, max=0.75)


unary_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["abs", torch.abs],
        ["clamp", clamp],
        ["cos", torch.cos],
        ["exp", torch.exp],
        ["log", torch.log],
        ["neg", torch.neg],
        ["sin", torch.sin],
        ["sqrt", torch.sqrt],
        ["tan", torch.tan],
    ],
)


op_bench.generate_pt_tests_from_op_list(
    unary_ops_list, unary_ops_configs, UnaryOpBenchmark
)
op_bench.generate_pt_gradient_tests_from_op_list(
    unary_ops_list, unary_ops_configs, UnaryOpBenchmark
)


reduction_configs = op_bench.cross_product_configs(
    R=[64, 256],
    V=[32, 512],
    dim=[0, 1],
    device=["cpu", "cuda"],
    tags=["short", "long"],
)


class ReductionBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, R, V, dim, device, op_func):
        shape = (R, V) if dim == 0 else (V, R)
        self.input_tensor = torch.rand(shape, device=device, requires_grad=True)
        self.inputs = {"input_tensor": self.input_tensor, "dim": dim}
        self.op_func = op_func

    def forward(self, input_tensor, dim: int):
        return self.op_func(input_tensor, dim)


reduction_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["sum", lambda x, dim: x.sum(dim=dim)],
        ["min", lambda x, dim: x.min(dim=dim).values],
        ["max", lambda x, dim: x.max(dim=dim).values],
        ["prod", lambda x, dim: x.prod(dim=dim)],
    ],
)

op_bench.generate_pt_tests_from_op_list(
    reduction_ops_list, reduction_configs, ReductionBenchmark
)
op_bench.generate_pt_gradient_tests_from_op_list(
    reduction_ops_list, reduction_configs, ReductionBenchmark
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
