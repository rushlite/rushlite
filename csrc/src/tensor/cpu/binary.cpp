#include "lamp3/tensor/cpu/binary.hpp"

#include "lamp3/tensor/cpu/kernels.hpp"

namespace lmp::tensor::detail::cpu {

template <typename PtrList, typename OpFn>
void vectorized_binary_kernel(PtrList ptr_, OpFn fn_, size_t i) {
  ptr_.set_Out(i, fn_(::std::get<1>(ptr_.fns)(ptr_.data[1], i),
                      ::std::get<2>(ptr_.fns)(ptr_.data[2], i)));
}

template <typename PtrList, typename OpFn>
void binary_kernel_launcher(PtrList ptr_, OpFn fn_, size_t size) {
#pragma omp parallel for simd
  for (size_t i = 0; i < size; i++) {
    vectorized_binary_kernel(ptr_, fn_, i);
  }
}

template <template <typename> class OpFunctor, typename... Args>
// NOLINTNEXTLINE(readability-function-size,google-readability-function-size)
void binary_dispatch_handler(BinaryMetaHandler& meta, Args&&... args) {
  LMP_DISPATCH_ALL_TYPES(meta.out().type(), [&] {
    using out_dtype_t = scalar_t;
    LMP_DISPATCH_ALL_TYPES(meta.in()[0]->type(), [&] {
      using arg1_dtype_t = scalar_t;
      LMP_DISPATCH_ALL_TYPES(meta.in()[1]->type(), [&] {
        using arg2_dtype_t = scalar_t;
        binary_kernel_launcher(
            internal::PtrPack<out_dtype_t, arg1_dtype_t, arg2_dtype_t>(
                static_cast<out_dtype_t*>(meta.out().data()),
                static_cast<arg1_dtype_t*>(
                    const_cast<TensorImpl*>(meta.in()[0])->data()),
                static_cast<arg2_dtype_t*>(
                    const_cast<TensorImpl*>(meta.in()[1])->data())),
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

}  // namespace lmp::tensor::detail::cpu
