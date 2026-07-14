#include "lamp3/tensor/cpu/reduct.hpp"

#include "lamp3/tensor/utils/align_utils.hpp"

namespace lmp::tensor::detail::cpu {

template <typename PtrList, typename OpFn>
void vectorized_reduct_kernel(PtrList ptr_, OpFn fn_, size_t i, size_t axis,
                              const size_t* shape, const stride_t* strides) {
  stride_t outer = strides[axis];
  // == strides[axis - 1] for contiguous tensors, but defined at axis == 0.
  stride_t inner = outer * static_cast<stride_t>(shape[axis]);
  stride_t idx = ((i / outer) * inner) + (i % outer);

  auto incr = OpFn::kIdentity;
  for (size_t j = 0; j < shape[axis]; ++j) {
    incr = fn_(incr, ::std::get<1>(ptr_.fns)(ptr_.data[1], idx + (j * outer)));
  }
  ptr_.set_Out(i, incr);
}

template <typename PtrList, typename OpFn>
void reduct_kernel_launcher(PtrList ptr_, OpFn fn_, size_t size, size_t axis,
                            const size_t* shape, const stride_t* strides,
                            size_t /*ndims*/) {
#pragma omp parallel for simd
  for (size_t i = 0; i < size; i++) {
    vectorized_reduct_kernel(ptr_, fn_, i, axis, shape, strides);
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
          meta.out().numel(), axis, meta.in()[0]->shape().data(),
          meta.in()[0]->strides().data(), meta.out().shape().size());
    });
  });
}

template void reduct_dispatch_handler<SumFunctor>(ReductMetaHandler&, size_t);
template void reduct_dispatch_handler<MaxFunctor>(ReductMetaHandler&, size_t);
template void reduct_dispatch_handler<MinFunctor>(ReductMetaHandler&, size_t);
template void reduct_dispatch_handler<ProdFunctor>(ReductMetaHandler&, size_t);

}  // namespace lmp::tensor::detail::cpu