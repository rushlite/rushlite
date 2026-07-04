#include "lamp3/tensor/tensor.hpp"

#include <iostream>

#include "lamp3/tensor/lazy/realize.hpp"
#include "lamp3/tensor/native/shape_ops.hpp"

namespace lmp::tensor {

void* Tensor::data() { return impl_->data(); }
DataType Tensor::type() const noexcept { return impl_->type(); }
DeviceType Tensor::device() const noexcept { return impl_->device(); }
const std::vector<size_t>& Tensor::shape() const noexcept {
  return impl_->shape();
}
const std::vector<detail::stride_t>& Tensor::strides() const noexcept {
  return impl_->strides();
}
size_t Tensor::numel() const noexcept { return impl_->numel(); }

Tensor Tensor::reshape(std::vector<size_t> new_shape) const {
  return ops::reshape(*this, std::move(new_shape));
}

Tensor Tensor::squeeze(size_t dim) const { return ops::squeeze(*this, dim); }

Tensor Tensor::expand_dims(size_t dim) const {
  return ops::expand_dims(*this, dim);
}

Tensor Tensor::to(DeviceType device) const {
  return lmp::tensor::ops::to(*this, device);
}

Scalar Tensor::index(const std::vector<size_t>& idx) const {
  return impl_->index(idx);
}

void Tensor::copy(const Tensor& other) { impl_->copy(*other.impl_); }

void Tensor::fill(Scalar item) { impl_->fill(item); }

// call to data implicitly realizes lazy graph
void Tensor::realize() { impl_->data(); }

std::ostream& operator<<(std::ostream& os, const Tensor& obj) {
  obj.impl_->print(os);
  return os;
}

}  // namespace lmp::tensor
