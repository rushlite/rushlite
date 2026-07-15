#pragma once

#include "lamp3/tensor/data_ptr.hpp"
#include "lamp3/tensor/device_type.hpp"
#include "lamp3/tensor/dispatch_stub.hpp"
#include "lamp3/tensor/data_type.hpp"

namespace lmp::tensor {
class Tensor;

namespace ops {

/// @internal
using copy_fn = void (*)(DeviceType, const void*, void*, size_t, DataType,
                         DataType);
using empty_fn = detail::DataPtr (*)(size_t);
using fill_fn = void (*)(void*, size_t, Scalar, DataType type);
using resize_fn = void (*)(detail::DataPtr, size_t, size_t);
using add_inplace_fn = void (*)(void*, const void*, size_t, DataType);

LMP_DECLARE_DISPATCH(copy_fn, copy_stub);
LMP_DECLARE_DISPATCH(empty_fn, empty_stub);
LMP_DECLARE_DISPATCH(fill_fn, fill_stub);
LMP_DECLARE_DISPATCH(resize_fn, resize_stub);
LMP_DECLARE_DISPATCH(add_inplace_fn, add_inplace_stub);
/// @endinternal

/**
 * @brief Convert a tensor to a different device
 * @param a The tensor to convert
 * @param to_device The device to convert the tensor to
 * @return A new tensor with the same data, but on the new device
 * @note unlike Pytorch, this function returns a new tensor, not a view.
 */
Tensor to(const Tensor& a, DeviceType to_device);

/// @internal
void add_inplace(Tensor& destination, const Tensor& source);
/// @endinternal

}  // namespace lmp::tensor::ops
}
