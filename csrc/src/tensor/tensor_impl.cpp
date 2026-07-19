#include "lamp3/tensor/tensor_impl.hpp"

#include <algorithm>

#include "lamp3/common/assert.hpp"
#include "lamp3/tensor/data_type.hpp"
#include "lamp3/tensor/device_type.hpp"
#include "lamp3/tensor/dispatch_type.hpp"
#include "lamp3/tensor/lazy/realize.hpp"
#include "lamp3/tensor/native/memory_ops.hpp"
#include "lamp3/tensor/utils/align_utils.hpp"

namespace lmp::tensor {

TensorImpl::TensorImpl(Storage storage, const std::vector<size_t>& shape,
                       DataType dtype)
    : data_(std::move(storage)),
      shape_(shape),
      type_(dtype),
      strides_(std::vector<detail::stride_t>(shape.size())),
      numel_(shape.empty() ? 0
                           : std::accumulate(shape.begin(), shape.end(), 1,
                                             std::multiplies<>())) {
  LMP_DISPATCH_ALL_TYPES(dtype, [&] {
    LMP_CHECK(data_.byte_size() == 0 ||
              data_.byte_size() / sizeof(scalar_t) == numel_)
        << "Storage size mismatch: expected " << numel_ << " elements of type "
        << dtype << " (" << sizeof(scalar_t) << " bytes each), but storage has "
        << data_.byte_size() << " bytes (capacity for "
        << (data_.byte_size() / sizeof(scalar_t)) << " elements)";
  });
  set_contiguous_strides();
}

void* TensorImpl::data() {
  if (is_deferred()) lazy::realize(this);
  return data_.data();
}
DataType TensorImpl::type() const noexcept { return type_; }
DeviceType TensorImpl::device() const noexcept { return data_.device(); }
const std::vector<size_t>& TensorImpl::shape() const noexcept { return shape_; }
const std::vector<detail::stride_t>& TensorImpl::strides() const noexcept {
  return strides_;
}
size_t TensorImpl::numel() const noexcept { return numel_; }

bool TensorImpl::is_contiguous() const noexcept {
  if (numel_ == 0) return true;

  detail::stride_t expected_stride = 1;
  for (size_t i = shape_.size(); i > 0; --i) {
    const size_t dim = i - 1;
    if (shape_[dim] == 1) continue;
    if (strides_[dim] != expected_stride) return false;
    expected_stride *= static_cast<detail::stride_t>(shape_[dim]);
  }
  return true;
}

bool TensorImpl::is_deferred() const noexcept { return lazy_ != nullptr; }

const std::shared_ptr<lazy::LazyFunction>& TensorImpl::lazy_op()
    const noexcept {
  return lazy_;
}

void TensorImpl::set_realized(Storage storage) {
  LMP_DISPATCH_ALL_TYPES(type_, [&] {
    LMP_CHECK(storage.byte_size() / sizeof(scalar_t) == numel_)
        << "set_realized: storage size mismatch: expected " << numel_
        << " elements, got " << (storage.byte_size() / sizeof(scalar_t));
  });
  LMP_CHECK(storage.device() == data_.device())
      << "set_realized: device mismatch";
  data_ = std::move(storage);
  lazy_ = nullptr;
}

void TensorImpl::set_deferred(std::shared_ptr<lazy::LazyFunction> op) {
  LMP_CHECK(!is_deferred()) << "tensor already has a pending op";
  lazy_ = std::move(op);
}

Storage TensorImpl::storage() const noexcept { return data_; }

void TensorImpl::set_contiguous_strides() {
  detail::stride_t stride = 1;
  strides_.resize(shape_.size());
  for (size_t i = shape_.size(); i > 0; --i) {
    const size_t dim = i - 1;
    strides_[dim] = stride;
    stride *= static_cast<detail::stride_t>(shape_[dim]);
  }
}

TensorImpl TensorImpl::reshape(std::vector<size_t> new_shape) {
  size_t new_size = new_shape.empty()
                        ? 0
                        : std::accumulate(new_shape.begin(), new_shape.end(), 1,
                                          std::multiplies<>());
  LMP_CHECK(new_size == numel_) << "Cannot reshape tensor: total number of "
                                   "elements must remain the same.";
  LMP_CHECK(is_contiguous())
      << "Cannot reshape a non-contiguous tensor without materializing it";
  data();
  TensorImpl other(*this);
  other.shape_ = std::move(new_shape);
  other.set_contiguous_strides();
  return other;
}

TensorImpl TensorImpl::transpose(size_t dim0, size_t dim1) {
  LMP_CHECK(dim0 < shape_.size())
      << "Dimension " << dim0 << " out of range for transpose of rank "
      << shape_.size();
  LMP_CHECK(dim1 < shape_.size())
      << "Dimension " << dim1 << " out of range for transpose of rank "
      << shape_.size();

  data();
  TensorImpl other(*this);
  std::swap(other.shape_[dim0], other.shape_[dim1]);
  std::swap(other.strides_[dim0], other.strides_[dim1]);
  return other;
}

TensorImpl TensorImpl::permute(const std::vector<size_t>& dims) {
  LMP_CHECK(dims.size() == shape_.size())
      << "Permutation must contain exactly " << shape_.size()
      << " dimensions";

  std::vector<bool> seen(shape_.size(), false);
  for (size_t dim : dims) {
    LMP_CHECK(dim < shape_.size())
        << "Dimension " << dim << " out of range for permutation of rank "
        << shape_.size();
    LMP_CHECK(!seen[dim])
        << "Dimension " << dim << " appears more than once in permutation";
    seen[dim] = true;
  }

  data();
  TensorImpl other(*this);
  for (size_t i = 0; i < dims.size(); ++i) {
    other.shape_[i] = shape_[dims[i]];
    other.strides_[i] = strides_[dims[i]];
  }
  return other;
}

TensorImpl TensorImpl::squeeze(size_t dim) {
  LMP_CHECK(dim < shape_.size()) << "Dimension out of range for squeeze";
  LMP_CHECK(shape_[dim] == 1) << "Cannot squeeze dimension that is not size 1";
  data();
  TensorImpl other(*this);
  other.shape_.erase(other.shape_.begin() + dim);
  other.strides_.erase(other.strides_.begin() + dim);
  return other;
}

TensorImpl TensorImpl::expand_dims(size_t dim) {
  LMP_CHECK(dim <= shape_.size()) << "Dimension out of range for expand_dims";
  const detail::stride_t stride =
      dim < shape_.size()
          ? strides_[dim] * static_cast<detail::stride_t>(shape_[dim])
          : 1;
  data();
  TensorImpl other(*this);
  other.shape_.insert(other.shape_.begin() + dim, 1);
  other.strides_.insert(other.strides_.begin() + dim, stride);
  return other;
}

Scalar TensorImpl::index(const std::vector<size_t>& idx) {
  LMP_CHECK(idx.size() == shape_.size()) << "Indexing does not match shape";
  size_t at = 0;
  for (size_t i = 0; i < idx.size(); i++) {
    LMP_CHECK(idx[i] < shape_[i])
        << "Index " << idx[i] << " out of bounds for dimension " << i
        << " with size " << shape_[i];
    at += static_cast<size_t>(strides_[i]) * idx[i];
  }
  return LMP_DISPATCH_ALL_TYPES(type(), [&]() {
    scalar_t elem{};
    ops::copy_stub()(device(), DeviceType::CPU,
                     static_cast<scalar_t*>(data()) + at, &elem, 1, type(),
                     type());
    return static_cast<Scalar>(elem);
  });
}

// TODO(astronaut): this needs to be defined more clearly i.e. what happens if
// other is bigger/smaller, maybe default behavior should be to assign
// other.type, other.device, other.data COMPLETELY to this
void TensorImpl::copy(const TensorImpl& other) const {
  LMP_DISPATCH_ALL_TYPES(other.type(), [&] {
    ops::copy_stub()(other.device(), device(), other.data_.data(), data_.data(),
                     other.numel(), other.type(), type());
  });
}

void TensorImpl::fill(Scalar item) const {
  ops::fill_stub()(device(), data_.data(), numel(), item, type());
}

const size_t kMaxPrintElem = 2e1;
void TensorImpl::print(std::ostream& os) const {
  os << "Tensor(data=[";
  LMP_DISPATCH_ALL_TYPES(this->type_, [&] {
    size_t print_size = std::min(kMaxPrintElem, this->numel());
    auto* data_ptr = new scalar_t[print_size * sizeof(scalar_t)];
    ops::copy_stub()(this->device(), DeviceType::CPU, this->data_.data(),
                     static_cast<void*>(data_ptr), print_size, this->type_,
                     this->type_);
    for (size_t i = 0; i < print_size; i++) {
      os << data_ptr[i];
      if (i < print_size - 1) {
        os << ", ";
      } else if (print_size < this->numel()) {
        os << ",...";
      }
    }
  });
  os << "], shape=[";
  for (size_t i = 0; i < shape_.size(); i++) {
    os << shape_[i];
    if (i < shape_.size() - 1) {
      os << ", ";
    }
  }
  os << "], dtype=" << this->type_ << ", device=" << this->device() << ")";
}

}  // namespace lmp::tensor
