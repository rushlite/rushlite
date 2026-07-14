#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#include <cuda/std/array>

#include "lamp3/tensor/cuda/kernels.cuh"
#include "lamp3/tensor/cuda/unary.cuh"
#include "lamp3/tensor/cuda/vec.cuh"

namespace lmp::tensor::detail::cuda {

template <typename PtrList, typename OpFn>
__global__ void vectorized_unary_kernel(PtrList ptr_, OpFn fn_, size_t size) {
  for (size_t i = (blockIdx.x * blockDim.x) + threadIdx.x; i < size;
       i += gridDim.x * blockDim.x) {
    ptr_.set_Out(i, fn_(::cuda::std::get<1>(ptr_.fns)(ptr_.data[1], i)));
  }
}

// Fast path: contiguous, same-dtype input/output. Each thread processes a
// width-4 packet, with a scalar tail for the (size % 4) remainder.
template <typename T, typename OpFn>
__global__ void unary_kernel_vec(const T* in, T* out, OpFn fn_, size_t n_vec,
                                 size_t size) {
  const vec4_t<T>* in_v = reinterpret_cast<const vec4_t<T>*>(in);
  vec4_t<T>* out_v = reinterpret_cast<vec4_t<T>*>(out);
  size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;
  for (size_t i = idx; i < n_vec; i += stride) {
    out_v[i] = internal::apply_unary4<T>(in_v[i], fn_);
  }
  for (size_t i = (n_vec * internal::kVecWidth) + idx; i < size; i += stride) {
    out[i] = fn_(in[i]);
  }
}

template <typename PtrList, typename OpFn>
void unary_kernel_launcher(PtrList ptr_, OpFn fn_, size_t size) {
  size_t threads = 256;
  size_t blocks = std::min((size + threads - 1) / threads, 1024UL);
  vectorized_unary_kernel<<<blocks, threads>>>(ptr_, fn_, size);
  LMP_CUDA_INTERNAL_ASSERT(cudaDeviceSynchronize())
      << "unary_kernel_launcher: kernel failed.";
}

template <typename T, typename OpFn>
void unary_kernel_vec_launcher(const T* in, T* out, OpFn fn_, size_t size) {
  size_t n_vec = size / internal::kVecWidth;
  unary_kernel_vec<T><<<internal::elemwise_blocks(n_vec),
                        internal::kElemwiseThreads>>>(in, out, fn_, n_vec, size);
  LMP_CUDA_INTERNAL_ASSERT(cudaDeviceSynchronize())
      << "unary_kernel_vec_launcher: kernel failed.";
}

template <template <typename> class OpFunctor, typename... Args>
void unary_dispatch_handler(UnaryMetaHandler& meta, Args&&... args) {
  LMP_DISPATCH_ALL_TYPES(meta.out().type(), [&] {
    using out_dtype_t = scalar_t;
    LMP_DISPATCH_ALL_TYPES(meta.in()[0]->type(), [&] {
      using arg_dtype_t = scalar_t;
      auto* out_ptr = static_cast<out_dtype_t*>(meta.out().data());
      auto* in_ptr =
          static_cast<arg_dtype_t*>(const_cast<TensorImpl*>(meta.in()[0])->data());
      if constexpr (std::is_same_v<out_dtype_t, arg_dtype_t>) {
        using V = vec4_t<out_dtype_t>;
        if (internal::is_aligned(out_ptr, alignof(V)) &&
            internal::is_aligned(in_ptr, alignof(V))) {
          unary_kernel_vec_launcher(in_ptr, out_ptr,
                                    OpFunctor<out_dtype_t>(args...),
                                    meta.out().numel());
          return;
        }
      }
      unary_kernel_launcher(
          internal::CUDAPtrPack<out_dtype_t, arg_dtype_t>(out_ptr, in_ptr),
          OpFunctor<out_dtype_t>(std::forward<Args>(args)...),
          meta.out().numel());
    });
  });
}

template void unary_dispatch_handler<NegFunctor>(UnaryMetaHandler&);
template void unary_dispatch_handler<ExpFunctor>(UnaryMetaHandler&);
template void unary_dispatch_handler<LogFunctor>(UnaryMetaHandler&);
template void unary_dispatch_handler<SqrtFunctor>(UnaryMetaHandler&);
template void unary_dispatch_handler<AbsFunctor>(UnaryMetaHandler&);
template void unary_dispatch_handler<SinFunctor>(UnaryMetaHandler&);
template void unary_dispatch_handler<CosFunctor>(UnaryMetaHandler&);
template void unary_dispatch_handler<TanFunctor>(UnaryMetaHandler&);
template void unary_dispatch_handler<ClampFunctor>(UnaryMetaHandler&, Scalar&,
                                                   Scalar&);

}  // namespace lmp::tensor::detail::cuda