#pragma once

#include <memory>
#include <string>
#include <vector>

#include "lamp3/lamp3.hpp"
#include "reg_defs.hpp"

template <size_t N>
struct OperatorConfig {
  std::array<std::vector<size_t>, N> shapes;
  lmp::tensor::DeviceType device;
  lmp::tensor::DataType dtype;
};

class OperatorBase {
 public:
  virtual ~OperatorBase() = default;
  virtual std::string name() const = 0;
};

inline std::string to_string(const std::vector<size_t>& shape) {
  std::string str;
  for (size_t i = 0; i < shape.size(); i++) {
    if (i) {
      str += "x";
    }
    str += std::to_string(shape[i]);
  }
  return str;
}

template <size_t N>
std::string to_string(const std::array<std::vector<size_t>, N>& shapes) {
  std::string str;
  for (size_t i = 0; i < shapes.size(); i++) {
    if (i) {
      str += "_";
    }
    str += to_string(shapes[i]);
  }
  return str;
}

inline std::string to_string(lmp::tensor::DataType dtype) {
  switch (dtype) {
    case lmp::tensor::DataType::Bool:
      return "Bool";
    case lmp::tensor::DataType::Int16:
      return "Int16";
    case lmp::tensor::DataType::Int32:
      return "Int32";
    case lmp::tensor::DataType::Int64:
      return "Int64";
    case lmp::tensor::DataType::Float32:
      return "Float32";
    case lmp::tensor::DataType::Float64:
      return "Float64";
    default:
      return "Unknown";
  }
}

class BinaryOperatorBase : public OperatorBase {
 public:
  void register_benchmarks(const OperatorConfig<2>& config) {
    auto op_fn = [this](const std::array<lmp::autograd::Variable, 2>& inputs)
        -> lmp::autograd::Variable {
      return apply_operation(inputs[0], inputs[1]);
    };
    auto init_fn =
        // NOLINTNEXTLINE(bugprone-exception-escape)
        [config](bool requires_grad) -> std::array<lmp::autograd::Variable, 2> {
      return {lmp::autograd::randn(0, 1, config.shapes[0], requires_grad,
                                   config.device, config.dtype),
              lmp::autograd::randn(0, 1, config.shapes[1], requires_grad,
                                   config.device, config.dtype)};
    };

    bool is_cuda = config.device == lmp::tensor::DeviceType::CUDA;
    std::string bench_name = name() + "_" + to_string(config.shapes) + "_" +
                             to_string(config.dtype) + "_" +
                             (is_cuda ? "CUDA" : "CPU");
    register_forward<2>(bench_name, op_fn, init_fn, is_cuda);
    register_backward<2>(bench_name, op_fn, init_fn, is_cuda);
  }

 protected:
  virtual lmp::autograd::Variable apply_operation(
      const lmp::autograd::Variable& a, const lmp::autograd::Variable& b) = 0;
};

class UnaryOperatorBase : public OperatorBase {
 public:
  void register_benchmarks(const OperatorConfig<1>& config) {
    auto op_fn = [this](const std::array<lmp::autograd::Variable, 1>& inputs)
        -> lmp::autograd::Variable { return apply_operation(inputs[0]); };
    auto init_fn =
        // NOLINTNEXTLINE(bugprone-exception-escape)
        [config](bool requires_grad) -> std::array<lmp::autograd::Variable, 1> {
      return {lmp::autograd::rand(config.shapes[0], requires_grad,
                                  config.device, config.dtype)};
    };

    bool is_cuda = config.device == lmp::tensor::DeviceType::CUDA;
    std::string bench_name = name() + "_" + to_string(config.shapes) + "_" +
                             to_string(config.dtype) + "_" +
                             (is_cuda ? "CUDA" : "CPU");
    register_forward<1>(bench_name, op_fn, init_fn, is_cuda);
    register_backward<1>(bench_name, op_fn, init_fn, is_cuda);
  }

 protected:
  virtual lmp::autograd::Variable apply_operation(
      const lmp::autograd::Variable& a) = 0;
};

class ReductOperatorBase : public OperatorBase {
 public:
  const std::vector<size_t> axes = {0U, 1U};
  void register_benchmarks(const OperatorConfig<1>& config) {
    auto init_fn =
        // NOLINTNEXTLINE(bugprone-exception-escape)
        [config](bool requires_grad) -> std::array<lmp::autograd::Variable, 1> {
      return {lmp::autograd::rand(config.shapes[0], requires_grad,
                                  config.device, config.dtype)};
    };

    for (size_t axis : axes) {
      auto op_fn = [this,
                    axis](const std::array<lmp::autograd::Variable, 1>& inputs)
          -> lmp::autograd::Variable {
        return apply_operation(inputs[0], axis);
      };

      bool is_cuda = config.device == lmp::tensor::DeviceType::CUDA;
      std::string bench_name = name() + "_axis" + std::to_string(axis) + "_" +
                               to_string(config.shapes) + "_" +
                               to_string(config.dtype) + "_" +
                               (is_cuda ? "CUDA" : "CPU");
      register_forward<1>(bench_name, op_fn, init_fn, is_cuda);
      register_backward<1>(bench_name, op_fn, init_fn, is_cuda);
    }
  }

 protected:
  virtual lmp::autograd::Variable apply_operation(
      const lmp::autograd::Variable& a, size_t axis) = 0;
};

class AddOp : public BinaryOperatorBase {
 public:
  std::string name() const override { return "Add"; }  // NOLINT

 protected:
  lmp::autograd::Variable apply_operation(
      const lmp::autograd::Variable& a,
      const lmp::autograd::Variable& b) override {
    return a + b;
  }
};

class SubOp : public BinaryOperatorBase {
 public:
  std::string name() const override { return "Sub"; }  // NOLINT

 protected:
  lmp::autograd::Variable apply_operation(
      const lmp::autograd::Variable& a,
      const lmp::autograd::Variable& b) override {
    return a - b;
  }
};

class MulOp : public BinaryOperatorBase {
 public:
  std::string name() const override { return "Mul"; }  // NOLINT

 protected:
  lmp::autograd::Variable apply_operation(
      const lmp::autograd::Variable& a,
      const lmp::autograd::Variable& b) override {
    return a * b;
  }
};

class DivOp : public BinaryOperatorBase {
 public:
  std::string name() const override { return "Div"; }  // NOLINT

 protected:
  lmp::autograd::Variable apply_operation(
      const lmp::autograd::Variable& a,
      const lmp::autograd::Variable& b) override {
    return a / b;
  }
};

class PowOp : public BinaryOperatorBase {
 public:
  std::string name() const override { return "Pow"; }  // NOLINT

 protected:
  lmp::autograd::Variable apply_operation(
      const lmp::autograd::Variable& a,
      const lmp::autograd::Variable& b) override {
    return lmp::autograd::ops::pow(a, b);
  }
};

class NegOp : public UnaryOperatorBase {
 public:
  std::string name() const override { return "Neg"; }  // NOLINT

 protected:
  lmp::autograd::Variable apply_operation(
      const lmp::autograd::Variable& a) override {
    return lmp::autograd::ops::neg(a);
  }
};

class LogOp : public UnaryOperatorBase {
 public:
  std::string name() const override { return "Log"; }  // NOLINT

 protected:
  lmp::autograd::Variable apply_operation(
      const lmp::autograd::Variable& a) override {
    return lmp::autograd::ops::log(a);
  }
};

class ExpOp : public UnaryOperatorBase {
 public:
  std::string name() const override { return "Exp"; }  // NOLINT

 protected:
  lmp::autograd::Variable apply_operation(
      const lmp::autograd::Variable& a) override {
    return lmp::autograd::ops::exp(a);
  }
};

class SqrtOp : public UnaryOperatorBase {
 public:
  std::string name() const override { return "Sqrt"; }  // NOLINT

 protected:
  lmp::autograd::Variable apply_operation(
      const lmp::autograd::Variable& a) override {
    return lmp::autograd::ops::sqrt(a);
  }
};

class AbsOp : public UnaryOperatorBase {
 public:
  std::string name() const override { return "Abs"; }  // NOLINT

 protected:
  lmp::autograd::Variable apply_operation(
      const lmp::autograd::Variable& a) override {
    return lmp::autograd::ops::abs(a);
  }
};

class SinOp : public UnaryOperatorBase {
 public:
  std::string name() const override { return "Sin"; }  // NOLINT

 protected:
  lmp::autograd::Variable apply_operation(
      const lmp::autograd::Variable& a) override {
    return lmp::autograd::ops::sin(a);
  }
};

class CosOp : public UnaryOperatorBase {
 public:
  std::string name() const override { return "Cos"; }  // NOLINT

 protected:
  lmp::autograd::Variable apply_operation(
      const lmp::autograd::Variable& a) override {
    return lmp::autograd::ops::cos(a);
  }
};

class TanOp : public UnaryOperatorBase {
 public:
  std::string name() const override { return "Tan"; }  // NOLINT

 protected:
  lmp::autograd::Variable apply_operation(
      const lmp::autograd::Variable& a) override {
    return lmp::autograd::ops::tan(a);
  }
};

class ClampOp : public UnaryOperatorBase {
 public:
  std::string name() const override { return "Clamp"; }  // NOLINT

 protected:
  lmp::autograd::Variable apply_operation(
      const lmp::autograd::Variable& a) override {
    return lmp::autograd::ops::clamp(a, 0.25F, 0.75F);
  }
};

class MatmulOp {};  // stub

class TransposeOp {};  // stub

class SumOp : public ReductOperatorBase {
 public:
  std::string name() const override { return "Sum"; }  // NOLINT

 protected:
  lmp::autograd::Variable apply_operation(const lmp::autograd::Variable& a,
                                          size_t axis) override {
    return lmp::autograd::ops::sum(a, axis);
  }
};

class MinOp : public ReductOperatorBase {
 public:
  std::string name() const override { return "Min"; }  // NOLINT

 protected:
  lmp::autograd::Variable apply_operation(const lmp::autograd::Variable& a,
                                          size_t axis) override {
    return lmp::autograd::ops::min(a, axis);
  }
};

class MaxOp : public ReductOperatorBase {
 public:
  std::string name() const override { return "Max"; }  // NOLINT

 protected:
  lmp::autograd::Variable apply_operation(const lmp::autograd::Variable& a,
                                          size_t axis) override {
    return lmp::autograd::ops::max(a, axis);
  }
};

class ProdOp : public ReductOperatorBase {
 public:
  std::string name() const override { return "Prod"; }  // NOLINT

 protected:
  lmp::autograd::Variable apply_operation(const lmp::autograd::Variable& a,
                                          size_t axis) override {
    return lmp::autograd::ops::prod(a, axis);
  }
};