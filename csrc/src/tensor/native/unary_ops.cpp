#include "lamp3/tensor/native/unary_ops.hpp"

#include "lamp3/tensor/lazy/capture_mode.hpp"
#include "lamp3/tensor/lazy/functions/elementwise_unary.hpp"
#include "lamp3/tensor/lazy/record.hpp"
#include "lamp3/tensor/tensor.hpp"
#include "lamp3/tensor/tensor_impl.hpp"

namespace lmp::tensor::ops {

LMP_DEFINE_DISPATCH(neg_fn, neg_stub);
LMP_DEFINE_DISPATCH(log_fn, log_stub);
LMP_DEFINE_DISPATCH(exp_fn, exp_stub);
LMP_DEFINE_DISPATCH(sqrt_fn, sqrt_stub);
LMP_DEFINE_DISPATCH(abs_fn, abs_stub);
LMP_DEFINE_DISPATCH(sin_fn, sin_stub);
LMP_DEFINE_DISPATCH(cos_fn, cos_stub);
LMP_DEFINE_DISPATCH(tan_fn, tan_stub);
LMP_DEFINE_DISPATCH(clamp_fn, clamp_stub);
LMP_DEFINE_DISPATCH(abs_backward_fn, abs_backward_stub);
LMP_DEFINE_DISPATCH(clamp_backward_fn, clamp_backward_stub);

#define LMP_DEFINE_UNARY_OP(name, lazy_fn)                                     \
  Tensor name(const Tensor& a) {                                               \
    std::shared_ptr<TensorImpl> ai = detail::UnsafeTensorAccessor::getImpl(a); \
    if (lazy::should_capture(a.device())) {                                    \
      return detail::UnsafeTensorAccessor::fromImpl(                           \
          lazy::record(std::make_shared<lazy::lazy_fn>(                        \
              std::vector<std::shared_ptr<TensorImpl>>{ai})));                 \
    }                                                                          \
    return detail::UnsafeTensorAccessor::fromImpl(                             \
        std::make_shared<TensorImpl>(name##_stub()(a.device(), *ai)));         \
  }

LMP_DEFINE_UNARY_OP(neg, NegFn)
LMP_DEFINE_UNARY_OP(exp, ExpFn)
LMP_DEFINE_UNARY_OP(log, LogFn)
LMP_DEFINE_UNARY_OP(sqrt, SqrtFn)
LMP_DEFINE_UNARY_OP(abs, AbsFn)
LMP_DEFINE_UNARY_OP(sin, SinFn)
LMP_DEFINE_UNARY_OP(cos, CosFn)
LMP_DEFINE_UNARY_OP(tan, TanFn)

#undef LMP_DEFINE_UNARY_OP

Tensor clamp(const Tensor& a, Scalar min_val, Scalar max_val) {
  std::shared_ptr<TensorImpl> ai = detail::UnsafeTensorAccessor::getImpl(a);
  if (lazy::should_capture(a.device())) {
    return detail::UnsafeTensorAccessor::fromImpl(
        lazy::record(std::make_shared<lazy::ClampFn>(
            std::vector<std::shared_ptr<TensorImpl>>{ai}, min_val, max_val)));
  }
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      clamp_stub()(a.device(), *ai, min_val, max_val)));
}

Tensor abs_backward(const Tensor& input, const Tensor& grad_output) {
  LMP_CHECK(input.device() == grad_output.device())
      << "abs_backward requires tensors on the same device";
  LMP_CHECK(input.shape() == grad_output.shape())
      << "abs_backward requires tensors with identical shapes";
  LMP_CHECK(input.type() == grad_output.type())
      << "abs_backward requires tensors with identical dtypes";
  return detail::UnsafeTensorAccessor::fromImpl(
      std::make_shared<TensorImpl>(abs_backward_stub()(
          input.device(), *detail::UnsafeTensorAccessor::getImpl(input),
          *detail::UnsafeTensorAccessor::getImpl(grad_output))));
}

Tensor clamp_backward(const Tensor& input, const Tensor& grad_output,
                      Scalar min_val, Scalar max_val) {
  LMP_CHECK(input.device() == grad_output.device())
      << "clamp_backward requires tensors on the same device";
  LMP_CHECK(input.shape() == grad_output.shape())
      << "clamp_backward requires tensors with identical shapes";
  LMP_CHECK(input.type() == grad_output.type())
      << "clamp_backward requires tensors with identical dtypes";
  return detail::UnsafeTensorAccessor::fromImpl(
      std::make_shared<TensorImpl>(clamp_backward_stub()(
          input.device(), *detail::UnsafeTensorAccessor::getImpl(input),
          *detail::UnsafeTensorAccessor::getImpl(grad_output), min_val,
          max_val)));
}

}  // namespace lmp::tensor::ops
