#pragma once

#include <iostream>
#include <memory>
#include <numeric>
#include <vector>
#include "data_type.hpp"
#include "device_type.hpp"
#include "dispatch_type.hpp"
#include "lamp3/common/assert.hpp"
#include "lamp3/tensor/native/memory_ops.hpp"
#include "lamp3/tensor/storage.hpp"
#include "lamp3/tensor/utils/align_utils.hpp"

namespace lmp::tensor {

namespace lazy {
class LazyFunction;
}  // namespace lazy

/// @internal
/**
 * @brief  Main implementation class for Tensor object
 *
 * @details `TensorImpl` contains a few core members: `type_`, `shape_`, and `data_`
 * Note that similar to Pytorch, Tensor/TensorImpl is not responsible for the 
 * low-level data storage -- note that `TensorImpl` has no member called `device_`.
 * That is managed by `Storage`.
 *
 * @see Tensor, Storage
 */
class TensorImpl {
 public:
  /**
   * @brief Construct a TensorImpl from a vector of data
   * 
   * @tparam T The element type of the input data vector
   * @param data   Flat vector containing the tensor data in row-major order
   * @param shape  Dimensions of the tensor, e.g. {28, 28} for a 2D tensor
   * @param device Target device where the tensor will be stored (CPU/GPU)
   * @param dtype  Data type for the tensor elements (may differ from T)
   * 
   * @throws std::runtime_error if data.size() != product of shape dimensions
   * 
   * @details This constructor allocates storage on the specified device and
   * copies the input data.
   *
   * @note Note that the input data's type T does NOT have to be the same as dtype. 
   * i.e. inputting dtype = DataType::Float64, but data = std::vector<int>{...} is valid
   */
  template <typename T>
  explicit TensorImpl(const std::vector<T>& data,
                      const std::vector<size_t>& shape, DeviceType device,
                      DataType dtype)
      : data_(LMP_DISPATCH_ALL_TYPES(
            dtype,
            [&] { return Storage(data.size() * sizeof(scalar_t), device); })),
        shape_(shape),
        type_(dtype),
        strides_(std::vector<detail::stride_t>(shape.size())),
        numel_(shape.empty() ? 0
                             : std::accumulate(shape.begin(), shape.end(), 1,
                                               std::multiplies<>())) {
    LMP_CHECK(data.size() == numel_)
        << "Size mismatch, product of shape must equal num elements";
    DataType src_dtype = TypeMeta<T>::kValue;
    ops::copy_stub()(DeviceType::CPU, device, data.data(), data_.data(), numel_,
                     src_dtype, type_);
    set_contiguous_strides();
  }
  /// @internal
  /// @note: this should not be used by the user.
  explicit TensorImpl(Storage storage, const std::vector<size_t>& shape,
                      DataType dtype);
  /// @endinternal

  void* data();
  DataType type() const noexcept;
  DeviceType device() const noexcept;
  const std::vector<size_t>& shape() const noexcept;
  const std::vector<detail::stride_t>& strides() const noexcept;
  size_t numel() const noexcept;
  bool is_contiguous() const noexcept;
  bool is_deferred() const noexcept;
  const std::shared_ptr<lazy::LazyFunction>& lazy_op() const noexcept;
  void set_realized(Storage storage);
  void set_deferred(std::shared_ptr<lazy::LazyFunction> op);
  Storage storage() const noexcept;

  TensorImpl reshape(std::vector<size_t> new_shape);
  TensorImpl transpose(size_t dim0, size_t dim1);
  TensorImpl permute(const std::vector<size_t>& dims);
  TensorImpl squeeze(size_t dim);
  TensorImpl expand_dims(size_t dim);
  Scalar index(const std::vector<size_t>& idx);

  void copy(const TensorImpl& other) const;
  void fill(Scalar item) const;
  void print(std::ostream& os) const;

 private:
  friend class Tensor;

  /// Set canonical row-major strides for a tensor known to be contiguous.
  void set_contiguous_strides();

  DataType type_;
  Storage data_;
  size_t numel_;
  std::vector<size_t> shape_;
  std::vector<detail::stride_t> strides_;
  std::shared_ptr<lazy::LazyFunction> lazy_;  // pending op; null on the eager path
};
/// @endinternal

}  // namespace lmp::tensor
