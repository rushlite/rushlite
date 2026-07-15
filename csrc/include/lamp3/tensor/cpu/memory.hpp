#pragma once

#include <cstddef>
#include "lamp3/tensor/data_ptr.hpp"
#include "lamp3/tensor/data_type.hpp"
#include "lamp3/tensor/device_type.hpp"

namespace lmp::tensor::detail::cpu {

/// @internal
void copy_cpu(DeviceType to_device, const void* src, void* dest, size_t size,
              DataType src_dtype, DataType dest_dtype);
DataPtr empty_cpu(size_t byte_size);
void fill_cpu(void* ptr, size_t size, Scalar t, DataType type);
void resize_cpu(DataPtr dptr, size_t old_byte_size, size_t new_byte_size);
void add_inplace_cpu(void* destination, const void* source, size_t size,
                     DataType type);
/// @endinternal

/// @internal
template <typename U, typename V>
void vecCopy(size_t size, const U* in, V* out);
/// @endinternal

}  // namespace lmp::tensor::detail::cpu
