#pragma once

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <cuda/std/array>
#include "lamp3/tensor/cpu/meta_handler.hpp"
#include "lamp3/tensor/cuda/kernels.cuh"
#include "lamp3/tensor/cuda/offset_util.cuh"
#include "lamp3/tensor/cuda/ptr_pack.cuh"
#include "lamp3/tensor/tensor_impl.hpp"

namespace lmp::tensor::detail::cuda {

/// @internal
template <typename PtrList, typename OpFn>
__global__ void vectorized_unary_kernel(PtrList ptr_, OpFn fn_, size_t size);

template <typename PtrList, typename OpFn>
void unary_kernel_launcher(PtrList ptr_, OpFn fn_, size_t size);

template <typename PtrList, typename OpFn>
__global__ void strided_unary_kernel(PtrList ptr_, OpFn fn_, size_t size,
                                     OffsetCalculator<1> offset);

template <typename PtrList, typename OpFn>
void strided_unary_kernel_launcher(PtrList ptr_, OpFn fn_, size_t size,
                                   OffsetCalculator<1> offset);

template <template <typename> class OpFunctor, typename... Args>
void unary_dispatch_handler(UnaryMetaHandler& meta, Args&&... args);

extern template void unary_dispatch_handler<NegFunctor>(UnaryMetaHandler&);
extern template void unary_dispatch_handler<ExpFunctor>(UnaryMetaHandler&);
extern template void unary_dispatch_handler<LogFunctor>(UnaryMetaHandler&);
extern template void unary_dispatch_handler<SqrtFunctor>(UnaryMetaHandler&);
extern template void unary_dispatch_handler<AbsFunctor>(UnaryMetaHandler&);
extern template void unary_dispatch_handler<SinFunctor>(UnaryMetaHandler&);
extern template void unary_dispatch_handler<CosFunctor>(UnaryMetaHandler&);
extern template void unary_dispatch_handler<TanFunctor>(UnaryMetaHandler&);
extern template void unary_dispatch_handler<ClampFunctor>(
    UnaryMetaHandler&, Scalar&, Scalar&);
/// @endinternal

}  // namespace lmp::tensor::detail::cuda
