#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#include <cstdint>
#include <cuda/std/array>
#include <type_traits>
#include <utility>

#include "lamp3/tensor/cuda/list_ptr.cuh"
#include "lamp3/tensor/cuda/reduct.cuh"

namespace lmp::tensor::detail::cuda {

template <typename PtrList, typename OpFn>
__global__ void axis_zero_reduct_kernel(PtrList ptr_, OpFn fn_, size_t size,
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
__global__ void strided_warp_per_row_reduct_kernel(PtrList ptr_, OpFn fn_,
                                                   size_t size, size_t axis,
                                                   const size_t* shape,
                                                   const stride_t* strides) {
  constexpr size_t kWarpSize = 32;
  constexpr unsigned int kFullWarpMask = 0xffffffffU;

  const size_t lane = threadIdx.x % kWarpSize;
  const size_t warp_in_block = threadIdx.x / kWarpSize;
  const size_t warps_per_block = blockDim.x / kWarpSize;
  const size_t first_row = (blockIdx.x * warps_per_block) + warp_in_block;
  const size_t warp_grid_stride = gridDim.x * warps_per_block;

  for (size_t row = first_row; row < size; row += warp_grid_stride) {
    const stride_t outer = strides[axis];
    const stride_t inner = outer * static_cast<stride_t>(shape[axis]);
    const stride_t row_start = ((row / outer) * inner) + (row % outer);

    auto increment = OpFn::kIdentity;
    for (size_t column = lane; column < shape[axis]; column += kWarpSize) {
      increment =
          fn_(increment, ::cuda::std::get<1>(ptr_.fns)(
                             ptr_.data[1], row_start + (column * outer)));
    }

#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
      increment =
          fn_(increment, __shfl_down_sync(kFullWarpMask, increment, offset));
    }
    if (lane == 0) {
      ptr_.set_Out(row, increment);
    }
  }
}

template <typename PtrList, typename OpFn>
__global__ void last_axis_warp_reduct_kernel(PtrList ptr_, OpFn fn_,
                                             size_t row_count,
                                             size_t reduction_size) {
  constexpr size_t kWarpSize = 32;
  constexpr unsigned int kFullWarpMask = 0xffffffffU;

  const size_t lane = threadIdx.x % kWarpSize;
  const size_t warp_in_block = threadIdx.x / kWarpSize;
  const size_t warps_per_block = blockDim.x / kWarpSize;
  const size_t first_row = (blockIdx.x * warps_per_block) + warp_in_block;
  const size_t warp_grid_stride = gridDim.x * warps_per_block;

  for (size_t row = first_row; row < row_count; row += warp_grid_stride) {
    const size_t row_start = row * reduction_size;
    auto increment = OpFn::kIdentity;
    for (size_t column = lane; column < reduction_size; column += kWarpSize) {
      increment = fn_(increment, ::cuda::std::get<1>(ptr_.fns)(
                                     ptr_.data[1], row_start + column));
    }

#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
      increment =
          fn_(increment, __shfl_down_sync(kFullWarpMask, increment, offset));
    }
    if (lane == 0) {
      ptr_.set_Out(row, increment);
    }
  }
}

template <typename OpFn>
__global__ void last_axis_vectorized_reduct_kernel(float* output,
                                                   const float* input, OpFn fn_,
                                                   size_t row_count,
                                                   size_t reduction_size) {
  constexpr size_t kWarpSize = 32;
  constexpr size_t kVectorWidth = 4;
  constexpr unsigned int kFullWarpMask = 0xffffffffU;

  const size_t lane = threadIdx.x % kWarpSize;
  const size_t warp_in_block = threadIdx.x / kWarpSize;
  const size_t warps_per_block = blockDim.x / kWarpSize;
  const size_t first_row = (blockIdx.x * warps_per_block) + warp_in_block;
  const size_t warp_grid_stride = gridDim.x * warps_per_block;
  const size_t vectors_per_row = reduction_size / kVectorWidth;

  for (size_t row = first_row; row < row_count; row += warp_grid_stride) {
    const auto* row_input =
        reinterpret_cast<const float4*>(input + (row * reduction_size));
    auto accumulator0 = OpFn::kIdentity;
    auto accumulator1 = OpFn::kIdentity;
    auto accumulator2 = OpFn::kIdentity;
    auto accumulator3 = OpFn::kIdentity;

    for (size_t vector_index = lane; vector_index < vectors_per_row;
         vector_index += kWarpSize) {
      const float4 values = row_input[vector_index];
      accumulator0 = fn_(accumulator0, values.x);
      accumulator1 = fn_(accumulator1, values.y);
      accumulator2 = fn_(accumulator2, values.z);
      accumulator3 = fn_(accumulator3, values.w);
    }

    accumulator0 = fn_(accumulator0, accumulator1);
    accumulator2 = fn_(accumulator2, accumulator3);
    auto increment = fn_(accumulator0, accumulator2);

#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
      increment =
          fn_(increment, __shfl_down_sync(kFullWarpMask, increment, offset));
    }
    if (lane == 0) {
      output[row] = increment;
    }
  }
}

template <typename PtrList, typename OpFn>
void axis_zero_reduct_kernel_launcher(PtrList ptr_, OpFn fn_, size_t size,
                                      size_t axis, const size_t* shape,
                                      const stride_t* strides, size_t ndims) {
  size_t threads = 256;
  size_t blocks = std::min((size + threads - 1) / threads, 1024UL);
  ListDevicePtr<stride_t> d_strides(strides, ndims);
  ListDevicePtr<size_t> d_shape(shape, ndims);
  axis_zero_reduct_kernel<<<blocks, threads>>>(ptr_, fn_, size, axis,
                                               d_shape.get(), d_strides.get());
  LMP_CUDA_INTERNAL_ASSERT(cudaDeviceSynchronize())
      << "axis_zero_reduct_kernel_launcher: kernel failed.";
}

template <typename PtrList, typename OpFn>
void strided_warp_per_row_reduct_kernel_launcher(PtrList ptr_, OpFn fn_,
                                                 size_t size, size_t axis,
                                                 const size_t* shape,
                                                 const stride_t* strides,
                                                 size_t ndims) {
  constexpr size_t kThreads = 256;
  constexpr size_t kWarpsPerBlock = kThreads / 32;
  const size_t blocks =
      std::min((size + kWarpsPerBlock - 1) / kWarpsPerBlock, 1024UL);
  ListDevicePtr<stride_t> d_strides(strides, ndims);
  ListDevicePtr<size_t> d_shape(shape, ndims);
  strided_warp_per_row_reduct_kernel<<<blocks, kThreads>>>(
      ptr_, fn_, size, axis, d_shape.get(), d_strides.get());
  LMP_CUDA_INTERNAL_ASSERT(cudaDeviceSynchronize())
      << "strided_warp_per_row_reduct_kernel_launcher: kernel failed.";
}

template <typename PtrList, typename OpFn>
void last_axis_warp_reduct_kernel_launcher(PtrList ptr_, OpFn fn_,
                                           size_t row_count,
                                           size_t reduction_size) {
  constexpr size_t kThreads = 256;
  constexpr size_t kWarpsPerBlock = kThreads / 32;
  const size_t blocks =
      std::min((row_count + kWarpsPerBlock - 1) / kWarpsPerBlock, 1024UL);
  last_axis_warp_reduct_kernel<<<blocks, kThreads>>>(ptr_, fn_, row_count,
                                                     reduction_size);
  LMP_CUDA_INTERNAL_ASSERT(cudaDeviceSynchronize())
      << "last_axis_warp_reduct_kernel_launcher: kernel failed.";
}

template <typename PtrList, typename OpFn>
void last_axis_vectorized_reduct_kernel_launcher(PtrList ptr_, OpFn fn_,
                                                 size_t row_count,
                                                 size_t reduction_size) {
  constexpr size_t kThreads = 256;
  constexpr size_t kWarpsPerBlock = kThreads / 32;
  const size_t blocks =
      std::min((row_count + kWarpsPerBlock - 1) / kWarpsPerBlock, 1024UL);
  last_axis_vectorized_reduct_kernel<<<blocks, kThreads>>>(
      static_cast<float*>(ptr_.data[0]),
      static_cast<const float*>(ptr_.data[1]), fn_, row_count, reduction_size);
  LMP_CUDA_INTERNAL_ASSERT(cudaDeviceSynchronize())
      << "last_axis_vectorized_reduct_kernel_launcher: kernel failed.";
}

template <template <typename> class OpFunctor, typename... Args>
void reduct_dispatch_handler(ReductMetaHandler& meta, size_t axis,
                             Args&&... args) {
  LMP_DISPATCH_ALL_TYPES(meta.out().type(), [&] {
    using out_dtype_t = scalar_t;
    LMP_DISPATCH_ALL_TYPES(meta.in()[0]->type(), [&] {
      using arg_dtype_t = scalar_t;
      auto ptr_pack = internal::CUDAPtrPack<out_dtype_t, arg_dtype_t>(
          static_cast<out_dtype_t*>(meta.out().data()),
          static_cast<arg_dtype_t*>(
              const_cast<TensorImpl*>(meta.in()[0])->data()));
      auto functor = OpFunctor<out_dtype_t>(std::forward<Args>(args)...);
      const auto& input_shape = meta.in()[0]->shape();
      if (axis == 0) {
        axis_zero_reduct_kernel_launcher(
            ptr_pack, functor, meta.out().numel(), axis, input_shape.data(),
            meta.in()[0]->strides().data(), input_shape.size());
      } else if (axis + 1 == input_shape.size()) {
        constexpr size_t kVectorWidth = 4;
        // Shorter rows already reach the launch/dispatch floor with the scalar
        // warp kernel; vectorization measured a material gain at 4096 elements.
        constexpr size_t kMinimumVectorizedReductionSize = 4096;
        const size_t reduction_size = input_shape[axis];
        const bool vector_aligned =
            (reinterpret_cast<std::uintptr_t>(ptr_pack.data[1]) %
             alignof(float4)) == 0;
        if constexpr (std::is_same_v<out_dtype_t, float> &&
                      std::is_same_v<arg_dtype_t, float>) {
          if (reduction_size >= kMinimumVectorizedReductionSize &&
              reduction_size % kVectorWidth == 0 && vector_aligned) {
            last_axis_vectorized_reduct_kernel_launcher(
                ptr_pack, functor, meta.out().numel(), reduction_size);
          } else {
            last_axis_warp_reduct_kernel_launcher(
                ptr_pack, functor, meta.out().numel(), reduction_size);
          }
        } else {
          last_axis_warp_reduct_kernel_launcher(
              ptr_pack, functor, meta.out().numel(), reduction_size);
        }
      } else {
        strided_warp_per_row_reduct_kernel_launcher(
            ptr_pack, functor, meta.out().numel(), axis, input_shape.data(),
            meta.in()[0]->strides().data(), input_shape.size());
      }
    });
  });
}

template void reduct_dispatch_handler<SumFunctor>(ReductMetaHandler&, size_t);
template void reduct_dispatch_handler<MaxFunctor>(ReductMetaHandler&, size_t);
template void reduct_dispatch_handler<MinFunctor>(ReductMetaHandler&, size_t);
template void reduct_dispatch_handler<ProdFunctor>(ReductMetaHandler&, size_t);

}  // namespace lmp::tensor::detail::cuda
