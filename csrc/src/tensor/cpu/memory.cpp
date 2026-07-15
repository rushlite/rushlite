#include "lamp3/tensor/cpu/memory.hpp"

#include <cstring>

#include "lamp3/common/macros.hpp"
#include "lamp3/tensor/data_type.hpp"
#include "lamp3/tensor/dispatch_type.hpp"
#include "lamp3/tensor/native/memory_ops.hpp"

#ifdef LMP_ENABLE_CUDA
#include "lamp3/tensor/cuda/memory.cuh"
#endif

namespace lmp::tensor::detail::cpu {

void fill_cpu(void* ptr, size_t size, Scalar t, DataType type) {
  LMP_DISPATCH_ALL_TYPES(type, [&]() {
    auto* data = static_cast<scalar_t*>(ptr);
    std::fill(data, data + size, static_cast<scalar_t>(t));
  });
}
void resize_cpu(DataPtr dptr, size_t old_byte_size, size_t new_byte_size) {
  void* ptr = ::operator new(new_byte_size);
  std::memcpy(ptr, dptr.data(), std::min(old_byte_size, new_byte_size));

  auto* deleter = std::get_deleter<std::function<void(void*)>>(dptr.ptr);
  dptr = DataPtr(ptr, *deleter);
}
DataPtr empty_cpu(size_t byte_size) {
  void* raw = static_cast<void*>(new char[byte_size]);
  return DataPtr(raw, [](void* ptr) { delete[] static_cast<char*>(ptr); });
}

template <typename T>
void addInplaceLoop(T* destination, const T* source, size_t size) {
#pragma omp parallel for simd
  for (size_t i = 0; i < size; ++i) {
    destination[i] += source[i];
  }
}

void add_inplace_cpu(void* destination, const void* source, size_t size,
                     DataType type) {
  LMP_DISPATCH_ALL_TYPES(type, [&]() {
    addInplaceLoop(static_cast<scalar_t*>(destination),
                   static_cast<const scalar_t*>(source), size);
  });
}

LMP_REGISTER_DISPATCH(ops::empty_stub, DeviceType::CPU, empty_cpu);
LMP_REGISTER_DISPATCH(ops::resize_stub, DeviceType::CPU, resize_cpu);
LMP_REGISTER_DISPATCH(ops::fill_stub, DeviceType::CPU, fill_cpu);
LMP_REGISTER_DISPATCH(ops::add_inplace_stub, DeviceType::CPU, add_inplace_cpu);

void copy_cpu(DeviceType to_device, const void* src, void* dest, size_t size,
              DataType src_dtype, DataType dest_dtype) {
  switch (to_device) {
    case DeviceType::CPU: {
      LMP_DISPATCH_ALL_TYPES(src_dtype, [&] {
        using src_type = scalar_t;
        LMP_DISPATCH_ALL_TYPES(dest_dtype, [&] {
          using dest_type = scalar_t;
          vecCopy(size, static_cast<const src_type*>(src),
                  static_cast<dest_type*>(dest));
        });
      });
      break;
    }
    case DeviceType::CUDA: {
#ifdef LMP_ENABLE_CUDA
      cuda::vecCopyHostToDevice(src, dest, size, src_dtype, dest_dtype);
#else
      LMP_INTERNAL_ASSERT(false) << "Enable CUDA is false";
#endif
      break;
    }
  }
}

/**
 * @brief Small parallized copy function using OMP
 *
 * @tparam U Input template type
 * @tparam V Output template type
 * @param size Size of the array being used
 * @param in Input array
 * @param out Output array
 */
template <typename U, typename V>
void vecCopy(size_t size, const U* in, V* out) {
#pragma omp parallel for simd
  for (size_t i = 0; i < size; i++) {
    out[i] = in[i];
  }
}

#define INSTANTIATE_COPY(arg1_type, arg2_type)                          \
  template void vecCopy<arg1_type, arg2_type>(size_t, const arg1_type*, \
                                              arg2_type(*));

LMP_FOR_EACH_CARTESIAN_PRODUCT(INSTANTIATE_COPY, LMP_LIST_TYPES, LMP_LIST_TYPES)
#undef INSTANTIATE_COPY

LMP_REGISTER_DISPATCH(ops::copy_stub, DeviceType::CPU, copy_cpu);

}  // namespace lmp::tensor::detail::cpu
