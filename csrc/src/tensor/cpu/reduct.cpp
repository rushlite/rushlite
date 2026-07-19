#include "lamp3/tensor/cpu/reduct.hpp"

namespace lmp::tensor::detail::cpu {

template <typename PtrList, typename OpFn>
void contiguous_reduct_kernel(PtrList ptr_, OpFn fn_, size_t i,
                              size_t reduced_size, stride_t outer) {
  const stride_t inner = outer * static_cast<stride_t>(reduced_size);
  stride_t idx = ((i / outer) * inner) + (i % outer);

  auto incr = OpFn::kIdentity;
  for (size_t j = 0; j < reduced_size; ++j) {
    incr = fn_(incr, ::std::get<1>(ptr_.fns)(ptr_.data[1], idx + (j * outer)));
  }
  ptr_.set_Out(i, incr);
}

template <typename PtrList, typename OpFn>
void strided_reduct_kernel(PtrList ptr_, OpFn fn_, size_t i,
                           size_t reduced_size, stride_t reduced_stride,
                           const CPUOffsetUtil<1>* offset) {
  const stride_t input_base = offset->get(i)[0];
  auto incr = OpFn::kIdentity;
  for (size_t j = 0; j < reduced_size; ++j) {
    incr = fn_(incr,
               ::std::get<1>(ptr_.fns)(
                   ptr_.data[1],
                   input_base + (static_cast<stride_t>(j) * reduced_stride)));
  }
  ptr_.set_Out(i, incr);
}

template <typename PtrList, typename OpFn>
void reduct_kernel_launcher(PtrList ptr_, OpFn fn_, size_t size,
                            size_t reduced_size, stride_t reduced_stride,
                            const CPUOffsetUtil<1>* offset) {
#pragma omp parallel for simd
  for (size_t i = 0; i < size; i++) {
    if (offset == nullptr) {
      contiguous_reduct_kernel(ptr_, fn_, i, reduced_size, reduced_stride);
    } else {
      strided_reduct_kernel(ptr_, fn_, i, reduced_size, reduced_stride,
                           offset);
    }
  }
}

template <template <typename> class OpFunctor, typename... Args>
void reduct_dispatch_handler(ReductMetaHandler& meta, size_t axis,
                             Args&&... args) {
  LMP_DISPATCH_ALL_TYPES(meta.out().type(), [&] {
    using out_dtype_t = scalar_t;
    LMP_DISPATCH_ALL_TYPES(meta.in()[0]->type(), [&] {
      using arg_dtype_t = scalar_t;
      reduct_kernel_launcher(
          internal::PtrPack<out_dtype_t, arg_dtype_t>(
              static_cast<out_dtype_t*>(meta.out().data()),
              static_cast<arg_dtype_t*>(
                  const_cast<TensorImpl*>(meta.in()[0])->data())),
          OpFunctor<out_dtype_t>(std::forward<Args>(args)...),
          meta.out().numel(), meta.in()[0]->shape()[axis],
          meta.in()[0]->strides()[axis],
          meta.has_offset()
              ? static_cast<const CPUOffsetUtil<1>*>(meta.offset())
              : nullptr);
    });
  });
}

template void reduct_dispatch_handler<SumFunctor>(ReductMetaHandler&, size_t);
template void reduct_dispatch_handler<MaxFunctor>(ReductMetaHandler&, size_t);
template void reduct_dispatch_handler<MinFunctor>(ReductMetaHandler&, size_t);
template void reduct_dispatch_handler<ProdFunctor>(ReductMetaHandler&, size_t);

}  // namespace lmp::tensor::detail::cpu
