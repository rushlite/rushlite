#pragma once

#include <driver_types.h>
#include <memory>
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
  std::shared_ptr<T[]> ptr_;
  size_t size_;

 public:
  ListDevicePtr() = default;
  explicit ListDevicePtr(const T* obj_list, size_t size) : size_(size) {
    T* raw = nullptr;
    LMP_CUDA_CHECK(cudaMallocAsync(&raw, sizeof(T) * size, 0));
    LMP_CUDA_CHECK(
        cudaMemcpyAsync(raw, obj_list, sizeof(T) * size, cudaMemcpyHostToDevice));
    ptr_ = std::shared_ptr<T[]>(raw, [size](T* p) {
      LMP_CUDA_CHECK(cudaFreeAsync(p, 0));
    });
  }

  T* get() const noexcept { return ptr_.get(); }
  size_t size() const noexcept { return size_; }
};
/// @endinternal

}  // namespace lmp::tensor::detail::cuda
