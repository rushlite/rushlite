#pragma once

#include <cstddef>

namespace lmp::tensor::detail::cuda {

/// @internal
// The owning device is recorded so deallocation does not depend on the
// caller's current CUDA device.
struct DeviceAllocation {
  void* pointer = nullptr;
  int device = 0;
};

DeviceAllocation cuda_pool_allocate(std::size_t byte_size);
void cuda_pool_deallocate(void* pointer, int device) noexcept;
/// @endinternal

}  // namespace lmp::tensor::detail::cuda
