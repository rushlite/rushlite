#pragma once

#include "lamp3/tensor/cpu/kernels.hpp"
#include "lamp3/tensor/cpu/meta_handler.hpp"
#include "lamp3/tensor/cpu/offset_util.hpp"
#include "lamp3/tensor/cpu/ptr_pack.hpp"
#include "lamp3/tensor/tensor_impl.hpp"

namespace lmp::tensor::detail::cpu {

constexpr size_t kNArgs = BinaryMetaHandler::kNumElem;

/// @internal
template <typename PtrList, typename OpFn>
void binary_kernel(PtrList ptr_, OpFn fn_, size_t i,
                   const CPUOffsetUtil<kNArgs>* offset);

template <typename PtrList, typename OpFn>
void binary_kernel_launcher(PtrList ptr_, OpFn fn_, size_t size,
                            const CPUOffsetUtil<kNArgs>* offset);
/// @endinternal

/// @internal
template <template <typename> class OpFunctor, typename... Args>
void binary_dispatch_handler(BinaryMetaHandler& meta, Args&&... args);
/// @endinternal

extern template void binary_dispatch_handler<AddFunctor>(BinaryMetaHandler&);
extern template void binary_dispatch_handler<SubFunctor>(BinaryMetaHandler&);
extern template void binary_dispatch_handler<MulFunctor>(BinaryMetaHandler&);
extern template void binary_dispatch_handler<DivFunctor>(BinaryMetaHandler&);
extern template void binary_dispatch_handler<PowFunctor>(BinaryMetaHandler&);
extern template void binary_dispatch_handler<EqFunctor>(BinaryMetaHandler&);
extern template void binary_dispatch_handler<NeFunctor>(BinaryMetaHandler&);
extern template void binary_dispatch_handler<GeFunctor>(BinaryMetaHandler&);
extern template void binary_dispatch_handler<GtFunctor>(BinaryMetaHandler&);
extern template void binary_dispatch_handler<LeFunctor>(BinaryMetaHandler&);
extern template void binary_dispatch_handler<LtFunctor>(BinaryMetaHandler&);
extern template void binary_dispatch_handler<AbsBackwardFunctor>(
    BinaryMetaHandler&);
extern template void binary_dispatch_handler<ClampBackwardFunctor>(
    BinaryMetaHandler&, Scalar&, Scalar&);

}  // namespace lmp::tensor::detail::cpu
