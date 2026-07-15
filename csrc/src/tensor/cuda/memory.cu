#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <thrust/device_ptr.h>

#include <cuda/std/array>

#include "lamp3/common/assert.hpp"
#include "lamp3/common/macros.hpp"
#include "lamp3/tensor/cuda/memory.cuh"
#include "lamp3/tensor/data_type.hpp"
#include "lamp3/tensor/dispatch_type.hpp"
#include "lamp3/tensor/native/memory_ops.hpp"

namespace lmp::tensor::detail::cuda {

DataPtr empty_cuda(size_t byte_size) {
  void* raw = nullptr;
  LMP_CUDA_CHECK(cudaMallocAsync(&raw, byte_size, nullptr))
      << "empty_cuda: cudaMalloc failed.";
  return DataPtr(raw, [byte_size](void* ptr) {
    LMP_CUDA_CHECK(cudaFreeAsync(ptr, nullptr));
  });
}

void fill_cuda(void* ptr, size_t size, Scalar t, DataType type) {
  LMP_DISPATCH_ALL_TYPES(type, [&]() {
    cudaVecFill(size, static_cast<scalar_t*>(ptr), static_cast<scalar_t>(t));
    LMP_CUDA_INTERNAL_ASSERT(cudaGetLastError())
        << "fill_cuda: thrust::fill failed.";
  });
}

void resize_cuda(DataPtr dptr, size_t old_byte_size, size_t new_byte_size) {
  void* ptr = nullptr;
  LMP_CUDA_CHECK(cudaMallocAsync(&ptr, new_byte_size, nullptr));
  LMP_CUDA_CHECK(cudaMemcpyAsync(ptr, dptr.data(),
                                 std::min(old_byte_size, new_byte_size),
                                 cudaMemcpyDeviceToDevice));

  auto* deleter = std::get_deleter<std::function<void(void*)>>(dptr.ptr);
  dptr = DataPtr(ptr, *deleter);
}

LMP_REGISTER_DISPATCH(ops::empty_stub, DeviceType::CUDA, empty_cuda);
LMP_REGISTER_DISPATCH(ops::fill_stub, DeviceType::CUDA, fill_cuda);
LMP_REGISTER_DISPATCH(ops::resize_stub, DeviceType::CUDA, resize_cuda);

void vecCopyHostToDevice(const void* src, void* dest, size_t size,
                         DataType src_dtype, DataType dest_dtype) {
  LMP_DISPATCH_ALL_TYPES(src_dtype, [&] {
    using src_type = scalar_t;
    LMP_DISPATCH_ALL_TYPES(dest_dtype, [&] {
      using dest_type = scalar_t;

      void* tmp = nullptr;
      LMP_CUDA_CHECK(cudaMallocAsync(&tmp, size * sizeof(src_type), nullptr))
          << "copy_cpu to CUDA: cudaMalloc for tmp failed.";
      LMP_CUDA_CHECK(cudaMemcpyAsync(tmp, src, size * sizeof(src_type),
                                     cudaMemcpyHostToDevice))
          << "copy_cpu to CUDA: cudaMemcpy HtoD for tmp failed.";

      cudaVecCopy<src_type, dest_type>(size, static_cast<const src_type*>(tmp),
                                       static_cast<dest_type*>(dest));

      LMP_CUDA_INTERNAL_ASSERT(cudaGetLastError())
          << "copy_cpu to CUDA: vecCopy kernel failed.";
      LMP_CUDA_CHECK(cudaFreeAsync(tmp, nullptr))
          << "copy_cpu to CUDA: cudaFreeAsync for tmp failed.";
    });
  });
}

void copy_cuda(DeviceType to_device, const void* src, void* dest, size_t size,
               DataType src_dtype, DataType dest_dtype) {
  switch (to_device) {
    case DeviceType::CPU: {
      LMP_DISPATCH_ALL_TYPES(src_dtype, [&] {
        using src_type = scalar_t;
        LMP_DISPATCH_ALL_TYPES(dest_dtype, [&] {
          using dest_type = scalar_t;

          void* tmp = nullptr;
          LMP_CUDA_CHECK(
              cudaMallocAsync(&tmp, size * sizeof(dest_type), nullptr))
              << "copy_cuda to CPU: cudaMalloc for tmp failed.";

          cudaVecCopy<src_type, dest_type>(size,
                                           static_cast<const src_type*>(src),
                                           static_cast<dest_type*>(tmp));
          LMP_CUDA_INTERNAL_ASSERT(cudaGetLastError())
              << "copy_cuda to CPU: vecCopy kernel failed.";
          LMP_CUDA_CHECK(cudaMemcpyAsync(dest, tmp, size * sizeof(dest_type),
                                         cudaMemcpyDeviceToHost))
              << "copy_cuda to CPU: cudaMemcpy DtoH failed.";
          LMP_CUDA_CHECK(cudaFreeAsync(tmp, nullptr))
              << "copy_cuda to CPU: cudaFreeAsync for tmp failed.";
        });
      });
      break;
    }
    case DeviceType::CUDA: {
      LMP_DISPATCH_ALL_TYPES(src_dtype, [&] {
        using src_type = scalar_t;
        LMP_DISPATCH_ALL_TYPES(dest_dtype, [&] {
          using dest_type = scalar_t;

          void* tmp = nullptr;
          LMP_CUDA_CHECK(
              cudaMallocAsync(&tmp, size * sizeof(dest_type), nullptr))
              << "copy_cuda to CUDA: cudaMalloc for tmp failed.";

          cudaVecCopy<src_type, dest_type>(size,
                                           static_cast<const src_type*>(src),
                                           static_cast<dest_type*>(tmp));

          LMP_CUDA_INTERNAL_ASSERT(cudaGetLastError())
              << "copy_cuda to CUDA: vecCopy kernel failed.";
          LMP_CUDA_CHECK(cudaMemcpyAsync(dest, tmp, size * sizeof(dest_type),
                                         cudaMemcpyDeviceToDevice))
              << "copy_cuda to CUDA: cudaMemcpy DtoD failed.";
          LMP_CUDA_CHECK(cudaFreeAsync(tmp, nullptr))
              << "copy_cuda to CUDA: cudaFreeAsync for tmp failed.";
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
__global__ void cudaVecCopyKernel(size_t size, const U* in, V* out) {
  for (size_t i = (blockIdx.x * blockDim.x) + threadIdx.x; i < size;
       i += gridDim.x * blockDim.x) {
    out[i] = static_cast<V>(in[i]);
  }
}

template <typename U, typename V>
void cudaVecCopy(size_t size, const U* in, V* out) {
  size_t threads = 256;
  size_t blocks = std::min((size + threads - 1) / threads, 1024UL);
  cudaVecCopyKernel<U, V><<<blocks, threads>>>(size, in, out);
}

template <typename T>
__global__ void cudaVecFillKernel(size_t size, T* out, T value) {
  for (size_t i = (blockIdx.x * blockDim.x) + threadIdx.x; i < size;
       i += gridDim.x * blockDim.x) {
    out[i] = value;
  }
}

template <typename T>
void cudaVecFill(size_t size, T* out, T value) {
  size_t threads = 256;
  size_t blocks = std::min((size + threads - 1) / threads, 1024UL);
  cudaVecFillKernel<T><<<blocks, threads>>>(size, out, value);
}

#define INSTANTIATE_COPY(arg1_type, arg2_type)                              \
  template void cudaVecCopy<arg1_type, arg2_type>(size_t, const arg1_type*, \
                                                  arg2_type*);
#define INSTANTIATE_FILL(arg1_type) \
  template void cudaVecFill<arg1_type>(size_t, arg1_type*, arg1_type);

LMP_FOR_EACH_CARTESIAN_PRODUCT(INSTANTIATE_COPY, LMP_LIST_TYPES, LMP_LIST_TYPES)
LMP_FOR_EACH_CARTESIAN_PRODUCT(INSTANTIATE_FILL, LMP_LIST_TYPES)

#undef INSTANTIATE_COPY
#undef INSTANTIATE_FILL

LMP_REGISTER_DISPATCH(ops::copy_stub, DeviceType::CUDA, copy_cuda);

}  // namespace lmp::tensor::detail::cuda
