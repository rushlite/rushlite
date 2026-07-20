#include "lamp3/tensor/native/shape_ops.hpp"

#include "lamp3/tensor/native/memory_ops.hpp"
#include "lamp3/tensor/tensor.hpp"
#include "lamp3/tensor/tensor_impl.hpp"

namespace lmp::tensor::ops {

Tensor reshape(const Tensor& a, std::vector<size_t> new_shape) {
  std::shared_ptr<TensorImpl> impl = detail::UnsafeTensorAccessor::getImpl(a);
  return detail::UnsafeTensorAccessor::fromImpl(
      std::make_shared<TensorImpl>(impl->reshape(std::move(new_shape))));
}

Tensor transpose(const Tensor& a, size_t dim0, size_t dim1) {
  auto impl = detail::UnsafeTensorAccessor::getImpl(a);
  return detail::UnsafeTensorAccessor::fromImpl(
      std::make_shared<TensorImpl>(impl->transpose(dim0, dim1)));
}

Tensor permute(const Tensor& a, const std::vector<size_t>& dims) {
  auto impl = detail::UnsafeTensorAccessor::getImpl(a);
  return detail::UnsafeTensorAccessor::fromImpl(
      std::make_shared<TensorImpl>(impl->permute(dims)));
}

Tensor squeeze(const Tensor& a, size_t dim) {
  auto impl = detail::UnsafeTensorAccessor::getImpl(a);
  return detail::UnsafeTensorAccessor::fromImpl(
      std::make_shared<TensorImpl>(impl->squeeze(dim)));
}

Tensor expand_dims(const Tensor& a, size_t dim) {
  auto impl = detail::UnsafeTensorAccessor::getImpl(a);
  return detail::UnsafeTensorAccessor::fromImpl(
      std::make_shared<TensorImpl>(impl->expand_dims(dim)));
}

}  // namespace lmp::tensor::ops
