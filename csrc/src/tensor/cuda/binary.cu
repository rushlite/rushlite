#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#include <cuda/std/tuple>

#include "lamp3/tensor/cuda/binary.cuh"
#include "lamp3/tensor/cuda/kernels.cuh"

namespace lmp::tensor::detail::cuda {

template <typename PtrList, typename OpFn>
__global__ void binary_kernel(PtrList ptr_, OpFn fn_, size_t size,
                              OffsetCalculator<kNArgs> offset) {
  for (size_t i = (blockIdx.x * blockDim.x) + threadIdx.x; i < size;
       i += gridDim.x * blockDim.x) {
    const offsets_t<kNArgs> offsets = offset.get(i);
    ptr_.set_Out(
        i, fn_(::cuda::std::get<1>(ptr_.fns)(ptr_.data[1], offsets[0]),
               ::cuda::std::get<2>(ptr_.fns)(ptr_.data[2], offsets[1])));
  }
}

template <typename PtrList, typename OpFn>
void binary_kernel_launcher(PtrList ptr_, OpFn fn_, size_t size,
                            OffsetCalculator<kNArgs> offset) {
  if (size == 0) return;
  size_t threads = 256;
  size_t blocks = std::min((size + threads - 1) / threads, 1024UL);
  binary_kernel<<<blocks, threads>>>(ptr_, fn_, size, offset);

  LMP_CUDA_INTERNAL_ASSERT(cudaDeviceSynchronize())
      << "binary_kernel_launcher: kernel failed.";
}

template <template <typename> class OpFunctor, typename... Args>
void binary_dispatch_handler(BinaryMetaHandler& meta, Args&&... args) {
  LMP_DISPATCH_ALL_TYPES(meta.out().type(), [&] {
    using out_dtype_t = scalar_t;
    LMP_DISPATCH_ALL_TYPES(meta.in()[0]->type(), [&] {
      using arg1_dtype_t = scalar_t;
      LMP_DISPATCH_ALL_TYPES(meta.in()[1]->type(), [&] {
        using arg2_dtype_t = scalar_t;
        auto* out_ptr = static_cast<out_dtype_t*>(meta.out().data());
        auto* in1_ptr = static_cast<arg1_dtype_t*>(
            const_cast<TensorImpl*>(meta.in()[0])->data());
        auto* in2_ptr = static_cast<arg2_dtype_t*>(
            const_cast<TensorImpl*>(meta.in()[1])->data());
        binary_kernel_launcher(
            internal::CUDAPtrPack<out_dtype_t, arg1_dtype_t, arg2_dtype_t>(
                out_ptr, in1_ptr, in2_ptr),
            OpFunctor<out_dtype_t>(std::forward<Args>(args)...),
            meta.out().numel(),
            static_cast<const CUDAOffsetUtil<kNArgs>*>(meta.offset())
                ->calculator());
      });
    });
  });
}

template void binary_dispatch_handler<AddFunctor>(BinaryMetaHandler&);
template void binary_dispatch_handler<SubFunctor>(BinaryMetaHandler&);
template void binary_dispatch_handler<MulFunctor>(BinaryMetaHandler&);
template void binary_dispatch_handler<DivFunctor>(BinaryMetaHandler&);
template void binary_dispatch_handler<PowFunctor>(BinaryMetaHandler&);
template void binary_dispatch_handler<EqFunctor>(BinaryMetaHandler&);
template void binary_dispatch_handler<NeFunctor>(BinaryMetaHandler&);
template void binary_dispatch_handler<GeFunctor>(BinaryMetaHandler&);
template void binary_dispatch_handler<GtFunctor>(BinaryMetaHandler&);
template void binary_dispatch_handler<LeFunctor>(BinaryMetaHandler&);
template void binary_dispatch_handler<LtFunctor>(BinaryMetaHandler&);
template void binary_dispatch_handler<AbsBackwardFunctor>(BinaryMetaHandler&);
template void binary_dispatch_handler<ClampBackwardFunctor>(BinaryMetaHandler&,
                                                            Scalar&, Scalar&);

}  // namespace lmp::tensor::detail::cuda
