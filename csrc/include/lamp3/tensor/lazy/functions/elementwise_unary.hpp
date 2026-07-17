#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "lamp3/tensor/data_type.hpp"
#include "lamp3/tensor/lazy/lazy_function.hpp"
#include "lamp3/tensor/native/unary_ops.hpp"
#include "lamp3/tensor/storage.hpp"
#include "lamp3/tensor/tensor_impl.hpp"

namespace lmp::tensor::lazy {

struct ElementwiseUnaryFn : LazyFunction {
  using LazyFunction::LazyFunction;
  std::shared_ptr<TensorImpl> infer_output() const override;
  bool is_fusible() const override { return true; }
};

struct NegFn : ElementwiseUnaryFn {
  using ElementwiseUnaryFn::ElementwiseUnaryFn;
  void run_eager(TensorImpl& out) override;
  std::string codegen_expr() const override { return "-({0})"; }
};

struct ExpFn : ElementwiseUnaryFn {
  using ElementwiseUnaryFn::ElementwiseUnaryFn;
  void run_eager(TensorImpl& out) override;
  std::string codegen_expr() const override { return "exp({0})"; }
};

struct LogFn : ElementwiseUnaryFn {
  using ElementwiseUnaryFn::ElementwiseUnaryFn;
  void run_eager(TensorImpl& out) override;
  std::string codegen_expr() const override { return "log({0})"; }
};

struct SqrtFn : ElementwiseUnaryFn {
  using ElementwiseUnaryFn::ElementwiseUnaryFn;
  void run_eager(TensorImpl& out) override;
  std::string codegen_expr() const override { return "sqrt({0})"; }
};

struct AbsFn : ElementwiseUnaryFn {
  using ElementwiseUnaryFn::ElementwiseUnaryFn;
  void run_eager(TensorImpl& out) override;
  std::string codegen_expr() const override {
    return "(({0}) < 0 ? -({0}) : ({0}))";
  }
};

struct SinFn : ElementwiseUnaryFn {
  using ElementwiseUnaryFn::ElementwiseUnaryFn;
  void run_eager(TensorImpl& out) override;
  std::string codegen_expr() const override { return "sin({0})"; }
};

struct CosFn : ElementwiseUnaryFn {
  using ElementwiseUnaryFn::ElementwiseUnaryFn;
  void run_eager(TensorImpl& out) override;
  std::string codegen_expr() const override { return "cos({0})"; }
};

struct TanFn : ElementwiseUnaryFn {
  using ElementwiseUnaryFn::ElementwiseUnaryFn;
  void run_eager(TensorImpl& out) override;
  std::string codegen_expr() const override { return "tan({0})"; }
};

struct ClampFn : ElementwiseUnaryFn {
  ClampFn(std::vector<std::shared_ptr<TensorImpl>> inputs, Scalar min_val,
          Scalar max_val)
      : ElementwiseUnaryFn(std::move(inputs)),
        min_val_(min_val),
        max_val_(max_val) {}

  void run_eager(TensorImpl& out) override;
  std::string codegen_expr() const override;

 private:
  Scalar min_val_;
  Scalar max_val_;
};

}  // namespace lmp::tensor::lazy
