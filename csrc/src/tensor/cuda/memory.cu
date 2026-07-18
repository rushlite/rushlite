#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <thrust/device_ptr.h>

#include <algorithm>
#include <cstdint>
#include <cuda/std/array>
#include <utility>

#include "allocator.cuh"
#include "lamp3/common/assert.hpp"
#include "lamp3/common/macros.hpp"
#include "lamp3/tensor/cuda/memory.cuh"
#include "lamp3/tensor/cuda/vec.cuh"
#include "lamp3/tensor/data_type.hpp"
#include "lamp3/tensor/dispatch_type.hpp"
#include "lamp3/tensor/native/memory_ops.hpp"

namespace lmp::tensor::detail::cuda {

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

  const DeviceAllocation allocation = cuda_pool_allocate(byte_size);
  return DataPtr(allocation.pointer, [device = allocation.device](void* ptr) {
    cuda_pool_deallocate(ptr, device);
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
