#include "lamp3/tensor/cpu/offset_util.hpp"

#include "lamp3/tensor/tensor_impl.hpp"

namespace lmp::tensor::detail {

OperandLayout operand_layout(const TensorImpl& tensor) {
  return OperandLayout{tensor.shape(), tensor.strides()};
}

OperandLayout leading_operand_layout(const TensorImpl& tensor,
                                     size_t trailing_dims) {
  LMP_CHECK(tensor.shape().size() >= trailing_dims)
      << "Cannot remove " << trailing_dims << " trailing dimensions from rank "
      << tensor.shape().size();
  const size_t leading_dims = tensor.shape().size() - trailing_dims;
  return OperandLayout{
      shape_list(tensor.shape().begin(),
                 tensor.shape().begin() +
                     static_cast<shape_list::difference_type>(leading_dims)),
      stride_list(
          tensor.strides().begin(),
          tensor.strides().begin() +
              static_cast<stride_list::difference_type>(leading_dims))};
}

LMP_DEFINE_DISPATCH(offset_util_fn<1>, offset_util_stub_1);
LMP_DEFINE_DISPATCH(offset_util_fn<2>, offset_util_stub_2);
LMP_DEFINE_DISPATCH(offset_util_fn<3>, offset_util_stub_3);

namespace cpu {

template class CPUOffsetUtil<1>;
template class CPUOffsetUtil<2>;
template class CPUOffsetUtil<3>;

namespace {
offset_util_fn<1> offset_util_cpu_1 = offset_util_cpu<1>;
offset_util_fn<2> offset_util_cpu_2 = offset_util_cpu<2>;
offset_util_fn<3> offset_util_cpu_3 = offset_util_cpu<3>;
}  // namespace

LMP_REGISTER_DISPATCH(offset_util_stub_1, DeviceType::CPU, offset_util_cpu_1);
LMP_REGISTER_DISPATCH(offset_util_stub_2, DeviceType::CPU, offset_util_cpu_2);
LMP_REGISTER_DISPATCH(offset_util_stub_3, DeviceType::CPU, offset_util_cpu_3);

}  // namespace cpu

}  // namespace lmp::tensor::detail
