#pragma once

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#include "lamp3/tensor/cpu/meta_handler.hpp"
#include "lamp3/tensor/cuda/binary.cuh"
#include "lamp3/tensor/cuda/kernels.cuh"
#include "lamp3/tensor/cuda/offset_util.cuh"
#include "lamp3/tensor/cuda/ptr_pack.cuh"
#include "lamp3/tensor/tensor_impl.hpp"

namespace lmp::tensor::detail::cuda {

/// @internal
template <typename PtrList, typename OpFn>
__global__ void vectorized_expand_kernel(PtrList ptr_, OpFn fn_, size_t size,
                                         OffsetCalculator<kNArgs> align);

template <typename PtrList, typename OpFn>
void expand_kernel_launcher(PtrList ptr_, OpFn fn_, size_t size,
                            OffsetCalculator<kNArgs> align);

template <template <typename> class OpFunctor, typename... Args>
void expand_dispatch_handler(BinaryMetaHandler& meta, Args&&... args);

extern template void expand_dispatch_handler<AddFunctor>(BinaryMetaHandler&);
extern template void expand_dispatch_handler<SubFunctor>(BinaryMetaHandler&);
extern template void expand_dispatch_handler<MulFunctor>(BinaryMetaHandler&);
extern template void expand_dispatch_handler<DivFunctor>(BinaryMetaHandler&);
extern template void expand_dispatch_handler<PowFunctor>(BinaryMetaHandler&);
extern template void expand_dispatch_handler<EqFunctor>(BinaryMetaHandler&);
extern template void expand_dispatch_handler<NeFunctor>(BinaryMetaHandler&);
extern template void expand_dispatch_handler<GeFunctor>(BinaryMetaHandler&);
extern template void expand_dispatch_handler<GtFunctor>(BinaryMetaHandler&);
extern template void expand_dispatch_handler<LeFunctor>(BinaryMetaHandler&);
extern template void expand_dispatch_handler<LtFunctor>(BinaryMetaHandler&);

/// @endinternal

}  // namespace lmp::tensor::detail::cuda
