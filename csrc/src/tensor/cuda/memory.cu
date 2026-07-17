#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <thrust/device_ptr.h>

#include <algorithm>
#include <cstdint>
#include <cuda/std/array>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "allocator_core.hpp"
#include "lamp3/common/assert.hpp"
#include "lamp3/common/macros.hpp"
#include "lamp3/tensor/cuda/memory.cuh"
#include "lamp3/tensor/cuda/vec.cuh"
#include "lamp3/tensor/data_type.hpp"
#include "lamp3/tensor/dispatch_type.hpp"
#include "lamp3/tensor/native/memory_ops.hpp"

namespace lmp::tensor::detail::cuda {
namespace {

constexpr std::size_t kAllocationAlignment = 256;
constexpr std::size_t kSegmentSize = 64ULL * 1024 * 1024;

using allocator::Address;
using allocator::align_up;
using allocator::Allocation;
using allocator::BlockArena;
using allocator::ReleasedSegment;

[[noreturn]] void throw_cuda_error(const char* operation, cudaError_t error) {
  throw std::runtime_error(std::string(operation) + ": " +
                           cudaGetErrorString(error));
}

void check_cuda(cudaError_t error, const char* operation) {
  if (error != cudaSuccess) {
    throw_cuda_error(operation, error);
  }
}

Address to_address(void* pointer) { return reinterpret_cast<Address>(pointer); }

void* to_pointer(Address address) { return reinterpret_cast<void*>(address); }

struct DeviceState {
  DeviceState() : arena(kAllocationAlignment) {}

  std::mutex mutex;
  BlockArena arena;
};

struct DeviceAllocation {
  void* pointer = nullptr;
  int device = 0;
};

class CudaAllocator {
 public:
  static CudaAllocator& instance() {
    static CudaAllocator* allocator = new CudaAllocator();
    return *allocator;
  }

  DeviceAllocation allocate(std::size_t byte_size) {
    if (byte_size == 0) {
      return {};
    }

    int device = 0;
    check_cuda(cudaGetDevice(&device),
               "CUDA allocator could not get the current device");
    DeviceState& state = state_for(device);
    std::lock_guard<std::mutex> lock(state.mutex);

    if (auto allocation = state.arena.allocate(byte_size)) {
      return {to_pointer(allocation->address), device};
    }

    const std::size_t aligned_request =
        align_up(byte_size, kAllocationAlignment);
    const std::size_t segment_size = std::max(kSegmentSize, aligned_request);

    void* segment_pointer = nullptr;
    cudaError_t error = cudaMalloc(&segment_pointer, segment_size);
    if (error == cudaErrorMemoryAllocation) {
      release_free_segments_locked(state);
      error = cudaMalloc(&segment_pointer, segment_size);
    }
    if (error != cudaSuccess) {
      throw std::runtime_error("CUDA allocator could not acquire a " +
                               std::to_string(segment_size) +
                               "-byte segment: " + cudaGetErrorString(error));
    }

    try {
      state.arena.add_segment(to_address(segment_pointer), segment_size);
    } catch (...) {
      const cudaError_t free_error = cudaFree(segment_pointer);
      if (free_error != cudaSuccess) {
        throw_cuda_error("CUDA allocator failed to release a rejected segment",
                         free_error);
      }
      throw;
    }

    const std::optional<Allocation> allocation =
        state.arena.allocate(byte_size);
    if (!allocation) {
      throw std::logic_error(
          "CUDA allocator could not use the segment it just acquired");
    }
    return {to_pointer(allocation->address), device};
  }

  void deallocate(void* pointer, int device) {
    if (pointer == nullptr) {
      return;
    }

    DeviceState& state = state_for(device);
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.arena.deallocate(to_address(pointer))) {
      throw std::runtime_error(
          "CUDA allocator received an unknown address for deallocation");
    }
  }

 private:
  CudaAllocator() {
    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count),
               "CUDA allocator could not get the device count");
    states_.reserve(static_cast<std::size_t>(device_count));
    for (int device = 0; device < device_count; ++device) {
      states_.push_back(std::make_unique<DeviceState>());
    }
  }

  DeviceState& state_for(int device) {
    if (device < 0 || static_cast<std::size_t>(device) >= states_.size()) {
      throw std::runtime_error("CUDA allocator received an invalid device");
    }
    return *states_[static_cast<std::size_t>(device)];
  }

  void release_free_segments_locked(DeviceState& state) {
    check_cuda(cudaDeviceSynchronize(),
               "CUDA allocator could not synchronize before OOM recovery");

    std::vector<ReleasedSegment> segments =
        state.arena.extract_fully_free_segments();
    for (std::size_t index = 0; index < segments.size(); ++index) {
      const cudaError_t error = cudaFree(to_pointer(segments[index].address));
      if (error == cudaSuccess) {
        continue;
      }

      for (std::size_t restore = index; restore < segments.size(); ++restore) {
        state.arena.add_segment(segments[restore].address,
                                segments[restore].size);
      }
      throw_cuda_error(
          "CUDA allocator could not release a segment during OOM recovery",
          error);
    }
  }

  std::vector<std::unique_ptr<DeviceState>> states_;
};

}  // namespace

template <typename T>
__global__ void addInplaceKernel(T* destination, const T* source,
                                 std::size_t n_vec, std::size_t size) {
  auto* dst_vec = reinterpret_cast<vec4_t<T>*>(destination);
  const auto* src_vec = reinterpret_cast<const vec4_t<T>*>(source);
  std::size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  std::size_t stride = gridDim.x * blockDim.x;
  for (std::size_t i = idx; i < n_vec; i += stride) {
    vec4_t<T> value = dst_vec[i];
    const vec4_t<T> addend = src_vec[i];
    value.x += addend.x;
    value.y += addend.y;
    value.z += addend.z;
    value.w += addend.w;
    dst_vec[i] = value;
  }
  for (std::size_t i = (n_vec * internal::kVecWidth) + idx; i < size;
       i += stride) {
    destination[i] += source[i];
  }
}

template <typename T>
void addInplace(T* destination, const T* source, std::size_t size) {
  using V = vec4_t<T>;
  const bool vectorized = internal::is_aligned(destination, alignof(V)) &&
                          internal::is_aligned(source, alignof(V));
  const std::size_t n_vec = vectorized ? size / internal::kVecWidth : 0;
  const std::size_t work_items = vectorized ? n_vec : size;
  addInplaceKernel<<<internal::elemwise_blocks(work_items),
                     internal::kElemwiseThreads>>>(destination, source, n_vec,
                                                   size);
}

DataPtr empty_cuda(std::size_t byte_size) {
  if (byte_size == 0) {
    return {};
  }

  const DeviceAllocation allocation =
      CudaAllocator::instance().allocate(byte_size);
  return DataPtr(allocation.pointer, [device = allocation.device](void* ptr) {
    CudaAllocator::instance().deallocate(ptr, device);
  });
}

void fill_cuda(void* ptr, std::size_t size, Scalar t, DataType type) {
  LMP_DISPATCH_ALL_TYPES(type, [&]() {
    cudaVecFill(size, static_cast<scalar_t*>(ptr), static_cast<scalar_t>(t));
    LMP_CUDA_INTERNAL_ASSERT(cudaGetLastError())
        << "fill_cuda: thrust::fill failed.";
  });
}

void add_inplace_cuda(void* destination, const void* source, std::size_t size,
                      DataType type) {
  LMP_DISPATCH_ALL_TYPES(type, [&]() {
    addInplace(static_cast<scalar_t*>(destination),
               static_cast<const scalar_t*>(source), size);
    LMP_CUDA_INTERNAL_ASSERT(cudaDeviceSynchronize())
        << "add_inplace_cuda: kernel failed.";
  });
}

void resize_cuda(DataPtr dptr, std::size_t old_byte_size,
                 std::size_t new_byte_size) {
  // TODO: resize_stub must take DataPtr by reference before this can update
  // StorageImpl. Keeping the temporary in this allocator avoids mixing
  // cudaMallocAsync ownership with the allocator's deleter.
  DataPtr replacement = empty_cuda(new_byte_size);
  LMP_CUDA_CHECK(cudaMemcpyAsync(replacement.data(), dptr.data(),
                                 std::min(old_byte_size, new_byte_size),
                                 cudaMemcpyDeviceToDevice));
  dptr = std::move(replacement);
}

LMP_REGISTER_DISPATCH(ops::empty_stub, DeviceType::CUDA, empty_cuda);
LMP_REGISTER_DISPATCH(ops::fill_stub, DeviceType::CUDA, fill_cuda);
LMP_REGISTER_DISPATCH(ops::resize_stub, DeviceType::CUDA, resize_cuda);
LMP_REGISTER_DISPATCH(ops::add_inplace_stub, DeviceType::CUDA,
                      add_inplace_cuda);

void vecCopyHostToDevice(const void* src, void* dest, std::size_t size,
                         DataType src_dtype, DataType dest_dtype) {
  LMP_DISPATCH_ALL_TYPES(src_dtype, [&] {
    using src_type = scalar_t;
    LMP_DISPATCH_ALL_TYPES(dest_dtype, [&] {
      using dest_type = scalar_t;

      DataPtr temporary = empty_cuda(size * sizeof(src_type));
      LMP_CUDA_CHECK(cudaMemcpyAsync(temporary.data(), src,
                                     size * sizeof(src_type),
                                     cudaMemcpyHostToDevice))
          << "copy_cpu to CUDA: cudaMemcpy HtoD for tmp failed.";

      cudaVecCopy<src_type, dest_type>(
          size, static_cast<const src_type*>(temporary.data()),
          static_cast<dest_type*>(dest));

      LMP_CUDA_INTERNAL_ASSERT(cudaGetLastError())
          << "copy_cpu to CUDA: vecCopy kernel failed.";
    });
  });
}

void copy_cuda(DeviceType to_device, const void* src, void* dest,
               std::size_t size, DataType src_dtype, DataType dest_dtype) {
  switch (to_device) {
    case DeviceType::CPU: {
      LMP_DISPATCH_ALL_TYPES(src_dtype, [&] {
        using src_type = scalar_t;
        LMP_DISPATCH_ALL_TYPES(dest_dtype, [&] {
          using dest_type = scalar_t;

          DataPtr temporary = empty_cuda(size * sizeof(dest_type));
          cudaVecCopy<src_type, dest_type>(
              size, static_cast<const src_type*>(src),
              static_cast<dest_type*>(temporary.data()));
          LMP_CUDA_INTERNAL_ASSERT(cudaGetLastError())
              << "copy_cuda to CPU: vecCopy kernel failed.";
          LMP_CUDA_CHECK(cudaMemcpyAsync(dest, temporary.data(),
                                         size * sizeof(dest_type),
                                         cudaMemcpyDeviceToHost))
              << "copy_cuda to CPU: cudaMemcpy DtoH failed.";
        });
      });
      break;
    }
    case DeviceType::CUDA: {
      LMP_DISPATCH_ALL_TYPES(src_dtype, [&] {
        using src_type = scalar_t;
        LMP_DISPATCH_ALL_TYPES(dest_dtype, [&] {
          using dest_type = scalar_t;

          DataPtr temporary = empty_cuda(size * sizeof(dest_type));
          cudaVecCopy<src_type, dest_type>(
              size, static_cast<const src_type*>(src),
              static_cast<dest_type*>(temporary.data()));

          LMP_CUDA_INTERNAL_ASSERT(cudaGetLastError())
              << "copy_cuda to CUDA: vecCopy kernel failed.";
          LMP_CUDA_CHECK(cudaMemcpyAsync(dest, temporary.data(),
                                         size * sizeof(dest_type),
                                         cudaMemcpyDeviceToDevice))
              << "copy_cuda to CUDA: cudaMemcpy DtoD failed.";
        });
      });
      break;
    }
    case DeviceType::Count:
      LMP_INTERNAL_ASSERT(false) << "DeviceType::Count is an internal utility.";
      break;
  }
}

template <typename U, typename V>
__global__ void cudaVecCopyKernel(std::size_t size, const U* in, V* out) {
  for (std::size_t i = (blockIdx.x * blockDim.x) + threadIdx.x; i < size;
       i += gridDim.x * blockDim.x) {
    out[i] = static_cast<V>(in[i]);
  }
}

template <typename U, typename V>
void cudaVecCopy(std::size_t size, const U* in, V* out) {
  std::size_t threads = 256;
  std::size_t blocks = std::min((size + threads - 1) / threads, 1024UL);
  cudaVecCopyKernel<U, V><<<blocks, threads>>>(size, in, out);
}

template <typename T>
__global__ void cudaVecFillKernel(std::size_t size, T* out, T value) {
  for (std::size_t i = (blockIdx.x * blockDim.x) + threadIdx.x; i < size;
       i += gridDim.x * blockDim.x) {
    out[i] = value;
  }
}

template <typename T>
void cudaVecFill(std::size_t size, T* out, T value) {
  std::size_t threads = 256;
  std::size_t blocks = std::min((size + threads - 1) / threads, 1024UL);
  cudaVecFillKernel<T><<<blocks, threads>>>(size, out, value);
}

#define INSTANTIATE_COPY(arg1_type, arg2_type)     \
  template void cudaVecCopy<arg1_type, arg2_type>( \
      std::size_t, const arg1_type*, arg2_type*);
#define INSTANTIATE_FILL(arg1_type) \
  template void cudaVecFill<arg1_type>(std::size_t, arg1_type*, arg1_type);

LMP_FOR_EACH_CARTESIAN_PRODUCT(INSTANTIATE_COPY, LMP_LIST_TYPES, LMP_LIST_TYPES)
LMP_FOR_EACH_CARTESIAN_PRODUCT(INSTANTIATE_FILL, LMP_LIST_TYPES)

#undef INSTANTIATE_COPY
#undef INSTANTIATE_FILL

LMP_REGISTER_DISPATCH(ops::copy_stub, DeviceType::CUDA, copy_cuda);

}  // namespace lmp::tensor::detail::cuda
