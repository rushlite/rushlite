#include "lamp3/tensor/native/memory_ops.hpp"

#include "lamp3/tensor/tensor.hpp"
#include "lamp3/tensor/tensor_impl.hpp"

namespace lmp::tensor::ops {

LMP_DEFINE_DISPATCH(copy_fn, copy_stub);
LMP_DEFINE_DISPATCH(empty_fn, empty_stub);
LMP_DEFINE_DISPATCH(fill_fn, fill_stub);
LMP_DEFINE_DISPATCH(resize_fn, resize_stub);
LMP_DEFINE_DISPATCH(add_inplace_fn, add_inplace_stub);

Tensor to(const Tensor& a, DeviceType to_device) {
  LMP_CHECK(a.device() != to_device)
      << "Device argument must be different from current device.";

  return LMP_DISPATCH_ALL_TYPES(a.type(), [&]() {
    Storage new_storage(a.numel() * sizeof(scalar_t), to_device);
    copy_stub()(a.device(), to_device, const_cast<Tensor&>(a).data(),
                new_storage.data(), a.numel(), a.type(), a.type());
    TensorImpl new_impl(new_storage, a.shape(), a.type());
    return detail::UnsafeTensorAccessor::fromImpl(
        std::make_shared<TensorImpl>(new_impl));
  });
}

void add_inplace(Tensor& destination, const Tensor& source) {
  LMP_CHECK(destination.device() == source.device())
      << "add_inplace requires tensors on the same device";
  LMP_CHECK(destination.shape() == source.shape())
      << "add_inplace requires tensors with identical shapes";
  LMP_CHECK(destination.type() == source.type())
      << "add_inplace requires tensors with identical dtypes";

  auto source_impl = detail::UnsafeTensorAccessor::getImpl(source);
  add_inplace_stub()(destination.device(), destination.data(),
                     source_impl->data(), destination.numel(),
                     destination.type());
}

}  // namespace lmp::tensor::ops
