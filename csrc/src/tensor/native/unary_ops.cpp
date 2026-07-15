#include "lamp3/tensor/native/unary_ops.hpp"

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

Tensor neg(const Tensor& a) {
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      neg_stub()(a.device(), *detail::UnsafeTensorAccessor::getImpl(a))));
}
Tensor exp(const Tensor& a) {
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      exp_stub()(a.device(), *detail::UnsafeTensorAccessor::getImpl(a))));
}
Tensor log(const Tensor& a) {
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      log_stub()(a.device(), *detail::UnsafeTensorAccessor::getImpl(a))));
}
Tensor sqrt(const Tensor& a) {
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      sqrt_stub()(a.device(), *detail::UnsafeTensorAccessor::getImpl(a))));
}
Tensor abs(const Tensor& a) {
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      abs_stub()(a.device(), *detail::UnsafeTensorAccessor::getImpl(a))));
}
Tensor sin(const Tensor& a) {
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      sin_stub()(a.device(), *detail::UnsafeTensorAccessor::getImpl(a))));
}
Tensor cos(const Tensor& a) {
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      cos_stub()(a.device(), *detail::UnsafeTensorAccessor::getImpl(a))));
}
Tensor tan(const Tensor& a) {
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      tan_stub()(a.device(), *detail::UnsafeTensorAccessor::getImpl(a))));
}
Tensor clamp(const Tensor& a, Scalar min_val, Scalar max_val) {
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      clamp_stub()(a.device(), *detail::UnsafeTensorAccessor::getImpl(a),
                   min_val, max_val)));
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
