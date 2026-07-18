#pragma once

#include <driver_types.h>

#include "lamp3/common/assert.hpp"
#include "lamp3/tensor/cuda/memory.cuh"

namespace lmp::tensor::detail::cuda {

/// @internal
/**
 * @brief A simple utility for managing a list of device pointers
 */
template <typename T>
class ListDevicePtr {
 private:
  DataPtr storage_;
  size_t size_ = 0;

 public:
  ListDevicePtr() = default;

  explicit ListDevicePtr(const T* obj_list, size_t size)
      : storage_(empty_cuda(sizeof(T) * size)), size_(size) {
    if (size == 0) {
      return;
    }
    LMP_CUDA_CHECK(
        cudaMemcpyAsync(storage_.data(), obj_list, sizeof(T) * size,
                        cudaMemcpyHostToDevice));
  }

  T* get() const noexcept { return static_cast<T*>(storage_.data()); }
  size_t size() const noexcept { return size_; }
};
/// @endinternal

}  // namespace lmp::tensor::detail::cuda
