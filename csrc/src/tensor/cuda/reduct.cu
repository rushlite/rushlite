#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#include <cuda/std/array>

#include "lamp3/tensor/cuda/list_ptr.cuh"
#include "lamp3/tensor/cuda/reduct.cuh"

namespace lmp::tensor::detail::cuda {

template <typename PtrList, typename OpFn>
__global__ void vectorized_reduct_kernel(PtrList ptr_, OpFn fn_, size_t size,
                                         size_t axis, const size_t* shape,
                                         const stride_t* strides) {
  for (size_t i = (blockIdx.x * blockDim.x) + threadIdx.x; i < size;
       i += gridDim.x * blockDim.x) {
    stride_t outer = strides[axis];
    // == strides[axis - 1] for contiguous tensors, but defined at axis == 0.
    stride_t inner = outer * static_cast<stride_t>(shape[axis]);
    stride_t idx = ((i / outer) * inner) + (i % outer);

    auto incr = OpFn::kIdentity;
    for (size_t j = 0; j < shape[axis]; ++j) {
      incr = fn_(
          incr, ::cuda::std::get<1>(ptr_.fns)(ptr_.data[1], idx + (j * outer)));
    }
    ptr_.set_Out(i, incr);
  }
}

template <typename PtrList, typename OpFn>
void reduct_kernel_launcher(PtrList ptr_, OpFn fn_, size_t size, size_t axis,
                            const size_t* shape, const stride_t* strides,
                            size_t ndims) {
  size_t threads = 256;
  size_t blocks = std::min((size + threads - 1) / threads, 1024UL);
  ListDevicePtr<stride_t> d_strides(strides, ndims);
  ListDevicePtr<size_t> d_shape(shape, ndims);
  vectorized_reduct_kernel<<<blocks, threads>>>(ptr_, fn_, size, axis,
                                                d_shape.get(), d_strides.get());
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
      reduct_kernel_launcher(
          internal::CUDAPtrPack<out_dtype_t, arg_dtype_t>(
              static_cast<out_dtype_t*>(meta.out().data()),
              static_cast<arg_dtype_t*>(
                  const_cast<TensorImpl*>(meta.in()[0])->data())),
          OpFunctor<out_dtype_t>(std::forward<Args>(args)...),
          meta.out().numel(), axis, meta.in()[0]->shape().data(),
          meta.in()[0]->strides().data(), meta.out().shape().size());
    });
  });
}

template void reduct_dispatch_handler<SumFunctor>(ReductMetaHandler&, size_t);
template void reduct_dispatch_handler<MaxFunctor>(ReductMetaHandler&, size_t);
template void reduct_dispatch_handler<MinFunctor>(ReductMetaHandler&, size_t);
template void reduct_dispatch_handler<ProdFunctor>(ReductMetaHandler&, size_t);

}  // namespace lmp::tensor::detail::cuda