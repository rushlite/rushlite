#include "lamp3/tensor/native/expand_ops.hpp"

#include <memory>
#include <vector>

#include "lamp3/common/assert.hpp"
#include "lamp3/tensor/lazy/capture_mode.hpp"
#include "lamp3/tensor/lazy/functions/elementwise_binary.hpp"
#include "lamp3/tensor/lazy/lazy_backend.hpp"
#include "lamp3/tensor/lazy/lazy_function.hpp"
#include "lamp3/tensor/lazy/record.hpp"
#include "lamp3/tensor/tensor.hpp"

namespace lmp::tensor::ops {

LMP_DEFINE_DISPATCH(add_fn, add_stub);
LMP_DEFINE_DISPATCH(sub_fn, sub_stub);
LMP_DEFINE_DISPATCH(mul_fn, mul_stub);
LMP_DEFINE_DISPATCH(div_fn, div_stub);
LMP_DEFINE_DISPATCH(pow_fn, pow_stub);
LMP_DEFINE_DISPATCH(eq_fn, eq_stub);
LMP_DEFINE_DISPATCH(ne_fn, ne_stub);
LMP_DEFINE_DISPATCH(ge_fn, ge_stub);
LMP_DEFINE_DISPATCH(le_fn, le_stub);
LMP_DEFINE_DISPATCH(gt_fn, gt_stub);
LMP_DEFINE_DISPATCH(lt_fn, lt_stub);

Tensor add(const Tensor& a, const Tensor& b) {
  LMP_CHECK(a.device() == b.device()) << "Tensors are on different devices";
  std::shared_ptr<TensorImpl> ai = detail::UnsafeTensorAccessor::getImpl(a);
  std::shared_ptr<TensorImpl> bi = detail::UnsafeTensorAccessor::getImpl(b);
  if (lazy::should_capture(a.device())) {
    return detail::UnsafeTensorAccessor::fromImpl(
        lazy::record(std::make_shared<lazy::AddFn>(
            std::vector<std::shared_ptr<TensorImpl>>{ai, bi})));
  }
  return detail::UnsafeTensorAccessor::fromImpl(
      std::make_shared<TensorImpl>(add_stub()(a.device(), *ai, *bi)));
}
Tensor sub(const Tensor& a, const Tensor& b) {
  LMP_CHECK(a.device() == b.device()) << "Tensors are on different devices";
  std::shared_ptr<TensorImpl> ai = detail::UnsafeTensorAccessor::getImpl(a);
  std::shared_ptr<TensorImpl> bi = detail::UnsafeTensorAccessor::getImpl(b);
  if (lazy::should_capture(a.device())) {
    return detail::UnsafeTensorAccessor::fromImpl(
        lazy::record(std::make_shared<lazy::SubFn>(
            std::vector<std::shared_ptr<TensorImpl>>{ai, bi})));
  }
  return detail::UnsafeTensorAccessor::fromImpl(
      std::make_shared<TensorImpl>(sub_stub()(a.device(), *ai, *bi)));
}
Tensor mul(const Tensor& a, const Tensor& b) {
  LMP_CHECK(a.device() == b.device()) << "Tensors are on different devices";
  std::shared_ptr<TensorImpl> ai = detail::UnsafeTensorAccessor::getImpl(a);
  std::shared_ptr<TensorImpl> bi = detail::UnsafeTensorAccessor::getImpl(b);
  if (lazy::should_capture(a.device())) {
    return detail::UnsafeTensorAccessor::fromImpl(
        lazy::record(std::make_shared<lazy::MulFn>(
            std::vector<std::shared_ptr<TensorImpl>>{ai, bi})));
  }
  return detail::UnsafeTensorAccessor::fromImpl(
      std::make_shared<TensorImpl>(mul_stub()(a.device(), *ai, *bi)));
}
Tensor div(const Tensor& a, const Tensor& b) {
  LMP_CHECK(a.device() == b.device()) << "Tensors are on different devices";
  std::shared_ptr<TensorImpl> ai = detail::UnsafeTensorAccessor::getImpl(a);
  std::shared_ptr<TensorImpl> bi = detail::UnsafeTensorAccessor::getImpl(b);
  if (lazy::should_capture(a.device())) {
    return detail::UnsafeTensorAccessor::fromImpl(
        lazy::record(std::make_shared<lazy::DivFn>(
            std::vector<std::shared_ptr<TensorImpl>>{ai, bi})));
  }
  return detail::UnsafeTensorAccessor::fromImpl(
      std::make_shared<TensorImpl>(div_stub()(a.device(), *ai, *bi)));
}
Tensor pow(const Tensor& a, const Tensor& b) {
  LMP_CHECK(a.device() == b.device()) << "Tensors are on different devices";
  std::shared_ptr<TensorImpl> ai = detail::UnsafeTensorAccessor::getImpl(a);
  std::shared_ptr<TensorImpl> bi = detail::UnsafeTensorAccessor::getImpl(b);
  if (lazy::should_capture(a.device())) {
    return detail::UnsafeTensorAccessor::fromImpl(
        lazy::record(std::make_shared<lazy::PowFn>(
            std::vector<std::shared_ptr<TensorImpl>>{ai, bi})));
  }
  return detail::UnsafeTensorAccessor::fromImpl(
      std::make_shared<TensorImpl>(pow_stub()(a.device(), *ai, *bi)));
}

Tensor eq(const Tensor& a, const Tensor& b) {
  LMP_CHECK(a.device() == b.device()) << "Tensors are on different devices";
  std::shared_ptr<TensorImpl> ai = detail::UnsafeTensorAccessor::getImpl(a);
  std::shared_ptr<TensorImpl> bi = detail::UnsafeTensorAccessor::getImpl(b);
  if (lazy::should_capture(a.device())) {
    return detail::UnsafeTensorAccessor::fromImpl(
        lazy::record(std::make_shared<lazy::EqFn>(
            std::vector<std::shared_ptr<TensorImpl>>{ai, bi})));
  }
  return detail::UnsafeTensorAccessor::fromImpl(
      std::make_shared<TensorImpl>(eq_stub()(a.device(), *ai, *bi)));
}
Tensor ne(const Tensor& a, const Tensor& b) {
  LMP_CHECK(a.device() == b.device()) << "Tensors are on different devices";
  std::shared_ptr<TensorImpl> ai = detail::UnsafeTensorAccessor::getImpl(a);
  std::shared_ptr<TensorImpl> bi = detail::UnsafeTensorAccessor::getImpl(b);
  if (lazy::should_capture(a.device())) {
    return detail::UnsafeTensorAccessor::fromImpl(
        lazy::record(std::make_shared<lazy::NeFn>(
            std::vector<std::shared_ptr<TensorImpl>>{ai, bi})));
  }
  return detail::UnsafeTensorAccessor::fromImpl(
      std::make_shared<TensorImpl>(ne_stub()(a.device(), *ai, *bi)));
}
Tensor ge(const Tensor& a, const Tensor& b) {
  LMP_CHECK(a.device() == b.device()) << "Tensors are on different devices";
  std::shared_ptr<TensorImpl> ai = detail::UnsafeTensorAccessor::getImpl(a);
  std::shared_ptr<TensorImpl> bi = detail::UnsafeTensorAccessor::getImpl(b);
  if (lazy::should_capture(a.device())) {
    return detail::UnsafeTensorAccessor::fromImpl(
        lazy::record(std::make_shared<lazy::GeFn>(
            std::vector<std::shared_ptr<TensorImpl>>{ai, bi})));
  }
  return detail::UnsafeTensorAccessor::fromImpl(
      std::make_shared<TensorImpl>(ge_stub()(a.device(), *ai, *bi)));
}
Tensor le(const Tensor& a, const Tensor& b) {
  LMP_CHECK(a.device() == b.device()) << "Tensors are on different devices";
  std::shared_ptr<TensorImpl> ai = detail::UnsafeTensorAccessor::getImpl(a);
  std::shared_ptr<TensorImpl> bi = detail::UnsafeTensorAccessor::getImpl(b);
  if (lazy::should_capture(a.device())) {
    return detail::UnsafeTensorAccessor::fromImpl(
        lazy::record(std::make_shared<lazy::LeFn>(
            std::vector<std::shared_ptr<TensorImpl>>{ai, bi})));
  }
  return detail::UnsafeTensorAccessor::fromImpl(
      std::make_shared<TensorImpl>(le_stub()(a.device(), *ai, *bi)));
}
Tensor gt(const Tensor& a, const Tensor& b) {
  LMP_CHECK(a.device() == b.device()) << "Tensors are on different devices";
  std::shared_ptr<TensorImpl> ai = detail::UnsafeTensorAccessor::getImpl(a);
  std::shared_ptr<TensorImpl> bi = detail::UnsafeTensorAccessor::getImpl(b);
  if (lazy::should_capture(a.device())) {
    return detail::UnsafeTensorAccessor::fromImpl(
        lazy::record(std::make_shared<lazy::GtFn>(
            std::vector<std::shared_ptr<TensorImpl>>{ai, bi})));
  }
  return detail::UnsafeTensorAccessor::fromImpl(
      std::make_shared<TensorImpl>(gt_stub()(a.device(), *ai, *bi)));
}
Tensor lt(const Tensor& a, const Tensor& b) {
  LMP_CHECK(a.device() == b.device()) << "Tensors are on different devices";
  std::shared_ptr<TensorImpl> ai = detail::UnsafeTensorAccessor::getImpl(a);
  std::shared_ptr<TensorImpl> bi = detail::UnsafeTensorAccessor::getImpl(b);
  if (lazy::should_capture(a.device())) {
    return detail::UnsafeTensorAccessor::fromImpl(
        lazy::record(std::make_shared<lazy::LtFn>(
            std::vector<std::shared_ptr<TensorImpl>>{ai, bi})));
  }
  return detail::UnsafeTensorAccessor::fromImpl(
      std::make_shared<TensorImpl>(lt_stub()(a.device(), *ai, *bi)));
}

}  // namespace lmp::tensor::ops