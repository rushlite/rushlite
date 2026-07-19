#include "lamp3/tensor/cpu/unary.hpp"

#include "lamp3/tensor/cpu/kernels.hpp"

namespace lmp::tensor::detail::cpu {

template <typename PtrList, typename OpFn>
void vectorized_unary_kernel(PtrList ptr_, OpFn fn_, size_t i) {
  ptr_.set_Out(i, fn_(::std::get<1>(ptr_.fns)(ptr_.data[1], i)));
}

template <typename PtrList, typename OpFn>
void unary_kernel_launcher(PtrList ptr_, OpFn fn_, size_t size) {
// TODO(astronaut): should guarantee alignment?? with aligned(a,out:64)?
#pragma omp parallel for simd
  for (size_t i = 0; i < size; i++) {
    vectorized_unary_kernel(ptr_, fn_, i);
  }
}

template <typename PtrList, typename OpFn>
void strided_unary_kernel(PtrList ptr_, OpFn fn_, size_t i,
                          const CPUOffsetUtil<1>* offset) {
  const offsets_t<1> offsets = offset->get(i);
  ptr_.set_Out(i, fn_(::std::get<1>(ptr_.fns)(ptr_.data[1], offsets[0])));
}

template <typename PtrList, typename OpFn>
void strided_unary_kernel_launcher(PtrList ptr_, OpFn fn_, size_t size,
                                   const CPUOffsetUtil<1>* offset) {
#pragma omp parallel for
  for (size_t i = 0; i < size; ++i) {
    strided_unary_kernel(ptr_, fn_, i, offset);
  }
}

template <template <typename> class OpFunctor, typename... Args>
void unary_dispatch_handler(UnaryMetaHandler& meta, Args&&... args) {
  LMP_DISPATCH_ALL_TYPES(meta.out().type(), [&] {
    using out_dtype_t = scalar_t;
    LMP_DISPATCH_ALL_TYPES(meta.in()[0]->type(), [&] {
      using arg_dtype_t = scalar_t;
      internal::PtrPack<out_dtype_t, arg_dtype_t> pointers(
          static_cast<out_dtype_t*>(meta.out().data()),
          static_cast<arg_dtype_t*>(
              const_cast<TensorImpl*>(meta.in()[0])->data()));
      OpFunctor<out_dtype_t> fn(std::forward<Args>(args)...);
      if (meta.has_offset()) {
        strided_unary_kernel_launcher(
            pointers, fn, meta.out().numel(),
            static_cast<const CPUOffsetUtil<1>*>(meta.offset()));
      } else {
        unary_kernel_launcher(pointers, fn, meta.out().numel());
      }
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

}  // namespace lmp::tensor::detail::cpu
