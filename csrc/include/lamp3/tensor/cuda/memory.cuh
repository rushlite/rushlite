#pragma once

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdatomic.h>
#include <atomic>
#include <cuda/std/array>
#include "lamp3/common/assert.hpp"
#include "lamp3/tensor/data_ptr.hpp"
#include "lamp3/tensor/data_type.hpp"
#include "lamp3/tensor/device_type.hpp"

namespace lmp::tensor::detail::cuda {

/// @internal
void copy_cuda(DeviceType to_device, const void* src, void* dest, size_t size,
               DataType src_dtype, DataType dest_dtype);
DataPtr empty_cuda(size_t byte_size);
void fill_cuda(void* ptr, size_t size, Scalar t, DataType type);
void resize_cuda(DataPtr dptr, size_t old_byte_size, size_t new_byte_size);

template <typename U, typename V>
__global__ void cudaVecCopyKernel(size_t size, const U* in, V* out);
template <typename U, typename V>
void cudaVecCopy(size_t size, const U* in, V* out);

template <typename T>
__global__ void cudaVecFillKernel(size_t size, T* out, T value);
template <typename T>
void cudaVecFill(size_t size, T* out, T value);

void vecCopyHostToDevice(const void* src, void* dest, size_t size,
                         DataType src_dtype, DataType dest_dtype);
/// @endinternal

}  // namespace lmp::tensor::detail::cuda
