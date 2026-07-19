#include "lamp3/tensor/cuda/offset_util.cuh"

namespace lmp::tensor::detail::cuda {

template class CUDAOffsetUtil<1>;
template class CUDAOffsetUtil<2>;
template class CUDAOffsetUtil<3>;

namespace {
offset_util_fn<1> offset_util_cuda_1 = offset_util_cuda<1>;
offset_util_fn<2> offset_util_cuda_2 = offset_util_cuda<2>;
offset_util_fn<3> offset_util_cuda_3 = offset_util_cuda<3>;
}  // namespace

LMP_REGISTER_DISPATCH(offset_util_stub_1, DeviceType::CUDA, offset_util_cuda_1);
LMP_REGISTER_DISPATCH(offset_util_stub_2, DeviceType::CUDA, offset_util_cuda_2);
LMP_REGISTER_DISPATCH(offset_util_stub_3, DeviceType::CUDA, offset_util_cuda_3);

}  // namespace lmp::tensor::detail::cuda
