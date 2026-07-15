#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#include <cuda/std/array>
#include <cuda/std/tuple>

#include "lamp3/tensor/cuda/binary.cuh"
#include "lamp3/tensor/cuda/kernels.cuh"
#include "lamp3/tensor/cuda/list_ptr.cuh"
#include "lamp3/tensor/cuda/vec.cuh"

namespace lmp::tensor::detail::cuda {

template <typename PtrList, typename OpFn>
__global__ void vectorized_binary_kernel(PtrList ptr_, OpFn fn_, size_t size) {
  for (size_t i = (blockIdx.x * blockDim.x) + threadIdx.x; i < size;
       i += gridDim.x * blockDim.x) {
    ptr_.set_Out(i, fn_(::cuda::std::get<1>(ptr_.fns)(ptr_.data[1], i),
                        ::cuda::std::get<2>(ptr_.fns)(ptr_.data[2], i)));
  }
}

// Fast path: contiguous, same-dtype inputs/output. Each thread processes a
// width-4 packet, with a scalar tail for the (size % 4) remainder.
template <typename T, typename OpFn>
__global__ void binary_kernel_vec(const T* in1, const T* in2, T* out, OpFn fn_,
                                  size_t n_vec, size_t size) {
  const vec4_t<T>* a_v = reinterpret_cast<const vec4_t<T>*>(in1);
  const vec4_t<T>* b_v = reinterpret_cast<const vec4_t<T>*>(in2);
  vec4_t<T>* out_v = reinterpret_cast<vec4_t<T>*>(out);
  size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;
  for (size_t i = idx; i < n_vec; i += stride) {
    out_v[i] = internal::apply_binary4<T>(a_v[i], b_v[i], fn_);
  }
  for (size_t i = (n_vec * internal::kVecWidth) + idx; i < size; i += stride) {
    out[i] = fn_(in1[i], in2[i]);
  }
}

template <typename PtrList, typename OpFn>
void binary_kernel_launcher(PtrList ptr_, OpFn fn_, size_t size) {
  size_t threads = 256;
  size_t blocks = std::min((size + threads - 1) / threads, 1024UL);
  vectorized_binary_kernel<<<blocks, threads>>>(ptr_, fn_, size);

  LMP_CUDA_INTERNAL_ASSERT(cudaDeviceSynchronize())
      << "binary_kernel_launcher: kernel failed.";
}

template <typename T, typename OpFn>
void binary_kernel_vec_launcher(const T* in1, const T* in2, T* out, OpFn fn_,
                                size_t size) {
  size_t n_vec = size / internal::kVecWidth;
  binary_kernel_vec<T>
      <<<internal::elemwise_blocks(n_vec), internal::kElemwiseThreads>>>(
          in1, in2, out, fn_, n_vec, size);
  LMP_CUDA_INTERNAL_ASSERT(cudaDeviceSynchronize())
      << "binary_kernel_vec_launcher: kernel failed.";
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
        if constexpr (std::is_same_v<out_dtype_t, arg1_dtype_t> &&
                      std::is_same_v<out_dtype_t, arg2_dtype_t>) {
          using V = vec4_t<out_dtype_t>;
          if (internal::is_aligned(out_ptr, alignof(V)) &&
              internal::is_aligned(in1_ptr, alignof(V)) &&
              internal::is_aligned(in2_ptr, alignof(V))) {
            binary_kernel_vec_launcher(in1_ptr, in2_ptr, out_ptr,
                                       OpFunctor<out_dtype_t>(args...),
                                       meta.out().numel());
            return;
          }
        }
        binary_kernel_launcher(
            internal::CUDAPtrPack<out_dtype_t, arg1_dtype_t, arg2_dtype_t>(
                out_ptr, in1_ptr, in2_ptr),
            OpFunctor<out_dtype_t>(std::forward<Args>(args)...),
            meta.out().numel());
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
