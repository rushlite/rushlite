#pragma once

#include <iostream>
#include <memory>
#include <utility>
#include <vector>
#include "data_type.hpp"
#include "device_type.hpp"
#include "dispatch_type.hpp"
#include "lamp3/common/config.hpp"
#include "lamp3/tensor/native/memory_ops.hpp"
#include "tensor_impl.hpp"
#include "utils/fill_like.hpp"  // NOLINT(misc-header-include-cycle)

namespace lmp::tensor {

namespace detail {
class UnsafeTensorAccessor;
}

/**
 * @brief  Main tensor object for Lamp3
 *
 * @details A thin, type-erased handle that shares ownership of an
 * underlying `TensorImpl`.  Multiple `Tensor` objects can
 * point to the same storage, so no deep copies occur. 
 *
 */
class Tensor {
 public:
  Tensor() = default;

  /**
  * @brief Construct a tensor from a vector
  * @param data   Flat vector in row-major order
  * @param shape  Dimensions, e.g. {28, 28} for MNIST
  * @param device Destination device (CPU/GPU)
  * @param dtype  Element type (defaults to float64)
  */
  template <typename T>
  explicit Tensor(const std::vector<T>& data, const std::vector<size_t>& shape,
                  DeviceType device = DEFAULT_DEVICE,
                  DataType dtype = DEFAULT_DTYPE)
      : impl_(std::make_shared<TensorImpl>(data, shape, device, dtype)) {}

  void* data();
  DataType type() const noexcept;
  DeviceType device() const noexcept;
  const std::vector<size_t>& shape() const noexcept;
  const std::vector<detail::stride_t>& strides() const noexcept;
  size_t numel() const noexcept;

  /// @note These functions return an object representing the underlying data
  template <typename T>
  std::vector<T> to_vector() const {
    std::vector<T> converted_data(impl_->numel());
    LMP_DISPATCH_ALL_TYPES(impl_->type(), [&] {
      std::unique_ptr<scalar_t[]> original_data =
          std::make_unique<scalar_t[]>(numel());
      ops::copy_stub()(device(), DeviceType::CPU, impl_->data(),
                       original_data.get(), numel(), type(), type());

      for (size_t i = 0; i < impl_->numel(); ++i) {
        converted_data[i] = static_cast<T>(original_data[i]);
      }
    });
    return converted_data;
  }
  Scalar index(const std::vector<size_t>& idx) const;

  /** 
  * @note These functions are similar to Pytorch in that they return a VIEW
  * i.e. don't change the underlying storage @see storage.hpp
  */
  Tensor reshape(std::vector<size_t> new_shape) const;
  Tensor squeeze(size_t dim) const;
  Tensor expand_dims(size_t dim) const;

  /**
  * @note `to` is similar to a unary operation because it returns a completely new 
  * object (NOT a view, which is what Pytorch does).
  */
  Tensor to(DeviceType device) const;

  /// @note These functions modify the actual data in-place.
  void copy(const Tensor& other);
  void fill(Scalar item);

  /**
  * @note This function guarantees the data is realized. it's a no-op
  * is already calculated
  */
  void realize();

  friend std::ostream& operator<<(std::ostream& os, const Tensor& obj);
  friend class TensorOpFact;
  friend class detail::UnsafeTensorAccessor;

 private:
  explicit Tensor(std::shared_ptr<TensorImpl> ptr) : impl_(std::move(ptr)) {}
  std::shared_ptr<TensorImpl> impl_;
};

namespace detail {
// @internal
/** @brief Unsafe accessor for Tensor, for getting and using impl
 * 
 * @details This class should only be used for operations which need access to the underlying Impl.
 * This class should be used SPARINGLY. 
 */
struct UnsafeTensorAccessor {
  static std::shared_ptr<TensorImpl> getImpl(const Tensor& ten) {
    return ten.impl_;
  }
  static Tensor fromImpl(std::shared_ptr<TensorImpl> ptr) {
    return Tensor(std::move(ptr));
  }
};
// @endinternal
}  // namespace detail

}  // namespace lmp::tensor
