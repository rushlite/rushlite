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
__global__ void contiguous_reduct_kernel(PtrList ptr_, OpFn fn_, size_t size,
                                         size_t reduced_size, stride_t outer);

template <typename PtrList, typename OpFn>
__global__ void strided_reduct_kernel(PtrList ptr_, OpFn fn_, size_t size,
                                      size_t reduced_size,
                                      stride_t reduced_stride,
                                      OffsetCalculator<1> offset);

template <typename PtrList, typename OpFn>
void reduct_kernel_launcher(PtrList ptr_, OpFn fn_, size_t size,
                            size_t reduced_size, stride_t reduced_stride,
                            const OffsetCalculator<1>* offset);

template <template <typename> class OpFunctor, typename... Args>
void reduct_dispatch_handler(ReductMetaHandler& meta, size_t axis,
                             Args&&... args);

extern template void reduct_dispatch_handler<SumFunctor>(ReductMetaHandler&,
                                                         size_t);
extern template void reduct_dispatch_handler<MaxFunctor>(ReductMetaHandler&,
                                                         size_t);
extern template void reduct_dispatch_handler<MinFunctor>(ReductMetaHandler&,
                                                         size_t);
extern template void reduct_dispatch_handler<ProdFunctor>(ReductMetaHandler&,
                                                          size_t);

/// @endinternal

}  // namespace lmp::tensor::detail::cuda
