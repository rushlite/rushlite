#pragma once

#include "lamp3/tensor/cpu/kernels.hpp"
#include "lamp3/tensor/cpu/meta_handler.hpp"
#include "lamp3/tensor/cpu/ptr_pack.hpp"
#include "lamp3/tensor/tensor_impl.hpp"

namespace lmp::tensor::detail::cpu {

/// @internal
template <typename PtrList, typename OpFn>
void contiguous_reduct_kernel(PtrList ptr_, OpFn fn_, size_t i,
                              size_t reduced_size, stride_t outer);

template <typename PtrList, typename OpFn>
void strided_reduct_kernel(PtrList ptr_, OpFn fn_, size_t i,
                           size_t reduced_size, stride_t reduced_stride,
                           const CPUOffsetUtil<1>* offset);

template <typename PtrList, typename OpFn>
void reduct_kernel_launcher(PtrList ptr_, OpFn fn_, size_t size,
                            size_t reduced_size, stride_t reduced_stride,
                            const CPUOffsetUtil<1>* offset);

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

}  // namespace lmp::tensor::detail::cpu
