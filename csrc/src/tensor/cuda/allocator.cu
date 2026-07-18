#include "allocator.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include "allocator_core.hpp"
#include "lamp3/common/assert.hpp"

namespace lmp::tensor::detail::cuda {
namespace {

constexpr std::size_t kAllocationAlignment = 256;
constexpr std::size_t kSegmentSize = 64ULL * 1024 * 1024;

using allocator::Address;
using allocator::align_up;
using allocator::BlockArena;
using allocator::ReleasedSegment;

Address to_address(void* pointer) { return reinterpret_cast<Address>(pointer); }

void* to_pointer(Address address) { return reinterpret_cast<void*>(address); }

struct DeviceState {
  DeviceState() : arena(kAllocationAlignment) {}

  std::mutex mutex;
  BlockArena arena;
};

// Host suballocator over a few long-lived CUDA segments. Reuse needs no event
// or sync because all work runs on the legacy null stream (see the allocator
// design doc for the stream-safety argument).
class CudaAllocator {
 public:
  static CudaAllocator& instance() {
    // Intentionally immortal: static DataPtr instances can outlive normal
    // static destruction and would otherwise deallocate into a dead singleton.
    static CudaAllocator* allocator = new CudaAllocator();
    return *allocator;
  }

  DeviceAllocation allocate(std::size_t byte_size) {
    if (byte_size == 0) {
      return {};
    }

    int device = 0;
    LMP_CUDA_CHECK(cudaGetDevice(&device));
    DeviceState& state = state_for(device);
    std::lock_guard<std::mutex> lock(state.mutex);

    if (auto allocation = state.arena.allocate(byte_size)) {
      return {to_pointer(allocation->address), device};
    }

    grow_locked(state, byte_size);

    auto allocation = state.arena.allocate(byte_size);
    LMP_INTERNAL_ASSERT(allocation.has_value())
        << "CUDA allocator could not use the segment it just acquired";
    return {to_pointer(allocation->address), device};
  }

  void deallocate(void* pointer, int device) noexcept {
    if (pointer == nullptr) {
      return;
    }

    DeviceState& state = state_for(device);
    std::lock_guard<std::mutex> lock(state.mutex);
    const bool freed = state.arena.deallocate(to_address(pointer));
    LMP_INTERNAL_ASSERT(freed)
        << "CUDA allocator received an unknown address for deallocation";
  }

 private:
  CudaAllocator() {
    int device_count = 0;
    LMP_CUDA_CHECK(cudaGetDeviceCount(&device_count));
    states_.reserve(static_cast<std::size_t>(device_count));
    for (int device = 0; device < device_count; ++device) {
      states_.push_back(std::make_unique<DeviceState>());
    }
  }

  DeviceState& state_for(int device) {
    LMP_INTERNAL_ASSERT(device >= 0 &&
                        static_cast<std::size_t>(device) < states_.size())
        << "CUDA allocator received an invalid device";
    return *states_[static_cast<std::size_t>(device)];
  }

  // Holds the device mutex across the blocking cudaMalloc so a concurrent
  // request cannot trigger duplicate growth.
  void grow_locked(DeviceState& state, std::size_t byte_size) {
    const std::size_t aligned_request =
        align_up(byte_size, kAllocationAlignment);
    const std::size_t segment_size = std::max(kSegmentSize, aligned_request);

    void* segment_pointer = nullptr;
    cudaError_t error = cudaMalloc(&segment_pointer, segment_size);
    if (error == cudaErrorMemoryAllocation) {
      release_free_segments_locked(state);
      error = cudaMalloc(&segment_pointer, segment_size);
    }
    LMP_CUDA_CHECK(error);

    state.arena.add_segment(to_address(segment_pointer), segment_size);
  }

  // Synchronizes first: draining queued work is what makes physically
  // releasing a segment safe (no kernel can still touch the address).
  void release_free_segments_locked(DeviceState& state) {
    LMP_CUDA_CHECK(cudaDeviceSynchronize());
    for (const ReleasedSegment& segment :
         state.arena.extract_fully_free_segments()) {
      LMP_CUDA_CHECK(cudaFree(to_pointer(segment.address)));
    }
  }

  std::vector<std::unique_ptr<DeviceState>> states_;
};

}  // namespace

DeviceAllocation cuda_pool_allocate(std::size_t byte_size) {
  return CudaAllocator::instance().allocate(byte_size);
}

void cuda_pool_deallocate(void* pointer, int device) noexcept {
  CudaAllocator::instance().deallocate(pointer, device);
}

}  // namespace lmp::tensor::detail::cuda
