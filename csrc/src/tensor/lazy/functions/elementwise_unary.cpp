#include "lamp3/tensor/lazy/functions/elementwise_unary.hpp"

#include <iomanip>
#include <limits>
#include <sstream>

namespace lmp::tensor::lazy {
namespace {

std::string scalar_literal(Scalar value) {
  std::ostringstream os;
  os << std::setprecision(std::numeric_limits<Scalar>::max_digits10) << value;
  return os.str();
}

}  // namespace

std::shared_ptr<TensorImpl> ElementwiseUnaryFn::infer_output() const {
  Storage empty(0, inputs[0]->device());
  return std::make_shared<TensorImpl>(empty, inputs[0]->shape(),
                                      inputs[0]->type());
}

#define LMP_DEFINE_UNARY_RUN_EAGER(name, stub)                               \
  void name##Fn::run_eager(TensorImpl& out) {                                \
    TensorImpl result = ops::stub##_stub()(inputs[0]->device(), *inputs[0]); \
    out.set_realized(result.storage());                                      \
  }

LMP_DEFINE_UNARY_RUN_EAGER(Neg, neg)
LMP_DEFINE_UNARY_RUN_EAGER(Exp, exp)
LMP_DEFINE_UNARY_RUN_EAGER(Log, log)
LMP_DEFINE_UNARY_RUN_EAGER(Sqrt, sqrt)
LMP_DEFINE_UNARY_RUN_EAGER(Abs, abs)
LMP_DEFINE_UNARY_RUN_EAGER(Sin, sin)
LMP_DEFINE_UNARY_RUN_EAGER(Cos, cos)
LMP_DEFINE_UNARY_RUN_EAGER(Tan, tan)

#undef LMP_DEFINE_UNARY_RUN_EAGER

void ClampFn::run_eager(TensorImpl& out) {
  TensorImpl result =
      ops::clamp_stub()(inputs[0]->device(), *inputs[0], min_val_, max_val_);
  out.set_realized(result.storage());
}

std::string ClampFn::codegen_expr() const {
  const std::string min = scalar_literal(min_val_);
  const std::string max = scalar_literal(max_val_);
  std::ostringstream os;
  os << "(({0}) < (" << min << ") ? (" << min << ") : " << "(({0}) > (" << max
     << ") ? (" << max << ") : ({0})))";
  return os.str();
}

}  // namespace lmp::tensor::lazy
