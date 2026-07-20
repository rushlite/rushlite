#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#include "lamp3/tensor/cuda/reduct.cuh"

namespace lmp::tensor::detail::cuda {

template <typename PtrList, typename OpFn>
__global__ void contiguous_reduct_kernel(PtrList ptr_, OpFn fn_, size_t size,
                                         size_t reduced_size, stride_t outer) {
  for (size_t i = (blockIdx.x * blockDim.x) + threadIdx.x; i < size;
       i += gridDim.x * blockDim.x) {
    const stride_t inner = outer * static_cast<stride_t>(reduced_size);
    stride_t idx = ((i / outer) * inner) + (i % outer);

    auto incr = OpFn::kIdentity;
    for (size_t j = 0; j < reduced_size; ++j) {
      incr = fn_(
          incr, ::cuda::std::get<1>(ptr_.fns)(ptr_.data[1], idx + (j * outer)));
    }
    ptr_.set_Out(i, incr);
  }
}

template <typename PtrList, typename OpFn>
__global__ void strided_reduct_kernel(PtrList ptr_, OpFn fn_, size_t size,
                                      size_t reduced_size,
                                      stride_t reduced_stride,
                                      OffsetCalculator<1> offset) {
  for (size_t i = (blockIdx.x * blockDim.x) + threadIdx.x; i < size;
       i += gridDim.x * blockDim.x) {
    const stride_t input_base = offset.get(i)[0];
    auto incr = OpFn::kIdentity;
    for (size_t j = 0; j < reduced_size; ++j) {
      incr = fn_(incr, ::cuda::std::get<1>(ptr_.fns)(
                            ptr_.data[1],
                            input_base +
                                (static_cast<stride_t>(j) * reduced_stride)));
    }
    ptr_.set_Out(i, incr);
  }
}

template <typename PtrList, typename OpFn>
void reduct_kernel_launcher(PtrList ptr_, OpFn fn_, size_t size,
                            size_t reduced_size, stride_t reduced_stride,
                            const OffsetCalculator<1>* offset) {
  if (size == 0) return;
  size_t threads = 256;
  size_t blocks = std::min((size + threads - 1) / threads, 1024UL);
  if (offset == nullptr) {
    contiguous_reduct_kernel<<<blocks, threads>>>(
        ptr_, fn_, size, reduced_size, reduced_stride);
  } else {
    strided_reduct_kernel<<<blocks, threads>>>(
        ptr_, fn_, size, reduced_size, reduced_stride, *offset);
  }
  LMP_CUDA_INTERNAL_ASSERT(cudaDeviceSynchronize())
      << "reduct_kernel_launcher: kernel failed.";
}

template <template <typename> class OpFunctor, typename... Args>
void reduct_dispatch_handler(ReductMetaHandler& meta, size_t axis,
                             Args&&... args) {
  LMP_DISPATCH_ALL_TYPES(meta.out().type(), [&] {
    using out_dtype_t = scalar_t;
    LMP_DISPATCH_ALL_TYPES(meta.in()[0]->type(), [&] {
      using arg_dtype_t = scalar_t;
      OffsetCalculator<1> offset{};
      const OffsetCalculator<1>* offset_ptr = nullptr;
      if (meta.has_offset()) {
        offset =
            static_cast<const CUDAOffsetUtil<1>*>(meta.offset())->calculator();
        offset_ptr = &offset;
      }
      reduct_kernel_launcher(
          internal::CUDAPtrPack<out_dtype_t, arg_dtype_t>(
              static_cast<out_dtype_t*>(meta.out().data()),
              static_cast<arg_dtype_t*>(
                  const_cast<TensorImpl*>(meta.in()[0])->data())),
          OpFunctor<out_dtype_t>(std::forward<Args>(args)...),
          meta.out().numel(), meta.in()[0]->shape()[axis],
          meta.in()[0]->strides()[axis], offset_ptr);
    });
  });
}

template void reduct_dispatch_handler<SumFunctor>(ReductMetaHandler&, size_t);
template void reduct_dispatch_handler<MaxFunctor>(ReductMetaHandler&, size_t);
template void reduct_dispatch_handler<MinFunctor>(ReductMetaHandler&, size_t);
template void reduct_dispatch_handler<ProdFunctor>(ReductMetaHandler&, size_t);

}  // namespace lmp::tensor::detail::cuda
