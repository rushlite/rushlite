#include "lamp3/common/macros.hpp"
#include "lamp3/tensor/cpu/meta_handler.hpp"
#include "lamp3/tensor/cuda/binary.cuh"
#include "lamp3/tensor/cuda/kernels.cuh"
#include "lamp3/tensor/cuda/matrix.cuh"
#include "lamp3/tensor/cuda/reduct.cuh"
#include "lamp3/tensor/cuda/unary.cuh"
#include "lamp3/tensor/infer_meta.hpp"
#include "lamp3/tensor/tensor_impl.hpp"

namespace lmp::tensor::detail::cuda {

#define DECLARE_BINARY_OPS_CUDA(args) DECLARE_BINARY_OPS_CUDA_HELPER args
#define DECLARE_BINARY_OPS_CUDA_HELPER(op, functor)                \
  TensorImpl op##_cuda(const TensorImpl& a, const TensorImpl& b) { \
    TensorMetaHandler meta(&a, &b);                                \
    binary_dispatch_handler<functor>(meta);                        \
    return meta.out();                                             \
  }

LMP_FOR_EACH_CARTESIAN_PRODUCT(
    DECLARE_BINARY_OPS_CUDA,
    ((add, AddFunctor), (sub, SubFunctor), (mul, MulFunctor), (div, DivFunctor),
     (pow, PowFunctor), (eq, EqFunctor), (ne, NeFunctor), (le, LeFunctor),
     (lt, LtFunctor), (ge, GeFunctor), (gt, GtFunctor), ));

#define DECLARE_UNARY_OPS_CUDA(args) DECLARE_UNARY_OPS_CUDA_HELPER args
#define DECLARE_UNARY_OPS_CUDA_HELPER(op, functor) \
  TensorImpl op##_cuda(const TensorImpl& a) {      \
    TensorMetaHandler meta(&a);                    \
    unary_dispatch_handler<functor>(meta);         \
    return meta.out();                             \
  }

LMP_FOR_EACH_CARTESIAN_PRODUCT(DECLARE_UNARY_OPS_CUDA,
                               ((neg, NegFunctor), (log, LogFunctor),
                                (exp, ExpFunctor), (sqrt, SqrtFunctor),
                                (abs, AbsFunctor), (sin, SinFunctor),
                                (cos, CosFunctor), (tan, TanFunctor), ));

TensorImpl clamp_cuda(const TensorImpl& a, Scalar min_val, Scalar max_val) {
  TensorMetaHandler meta(&a);
  unary_dispatch_handler<ClampFunctor>(meta, min_val, max_val);
  return meta.out();
}

TensorImpl abs_backward_cuda(const TensorImpl& input,
                             const TensorImpl& grad_output) {
  BinaryMetaHandler meta(&input, &grad_output);
  binary_dispatch_handler<AbsBackwardFunctor>(meta);
  return meta.out();
}

TensorImpl clamp_backward_cuda(const TensorImpl& input,
                               const TensorImpl& grad_output, Scalar min_val,
                               Scalar max_val) {
  BinaryMetaHandler meta(&input, &grad_output);
  binary_dispatch_handler<ClampBackwardFunctor>(meta, min_val, max_val);
  return meta.out();
}

TensorImpl transpose_cuda(const TensorImpl& a) {
  LMP_CHECK(a.shape().size() == 2) << "Invalid argument, transpose can only be "
                                      "performed on matrices of dim 2";
  size_t m = a.shape()[0];
  size_t n = a.shape()[1];

  DataType out_dtype = a.type();

  return LMP_DISPATCH_ALL_TYPES(a.type(), [&] {
    Storage c_storage(m * n * sizeof(scalar_t), DeviceType::CUDA);
    ::lmp::tensor::detail::cuda::cudaTranspose<scalar_t>(
        static_cast<const scalar_t*>(const_cast<TensorImpl&>(a).data()),
        static_cast<scalar_t*>(c_storage.data()), m, n);
    return TensorImpl(c_storage, {n, m}, out_dtype);
  });
}

TensorImpl matmul_cuda(const TensorImpl& a, const TensorImpl& b) {
  const MatmulMeta meta = infer_matmul(&a, &b);
  std::unique_ptr<OffsetUtil> offset = offset_util_stub_2()(
      DeviceType::CUDA, meta.batch_shape,
      std::array<OperandLayout, 2>{leading_operand_layout(a),
                                   leading_operand_layout(b)});
  const OffsetCalculator<2> batch_offsets =
      static_cast<const CUDAOffsetUtil<2>*>(offset.get())->calculator();
  const size_t a_rank = a.shape().size();
  const size_t b_rank = b.shape().size();

  return LMP_DISPATCH_ALL_TYPES(a.type(), [&] {
    using a_type_t = scalar_t;
    return LMP_DISPATCH_ALL_TYPES(b.type(), [&] {
      using b_type_t = scalar_t;
      return LMP_DISPATCH_ALL_TYPES(meta.dtype, [&] {
        using out_type_t = scalar_t;
        Storage c_storage(meta.size * sizeof(out_type_t), DeviceType::CUDA);
        ::lmp::tensor::detail::cuda::cudaMatMul<a_type_t, b_type_t, out_type_t>(
            static_cast<const a_type_t*>(const_cast<TensorImpl&>(a).data()),
            static_cast<const b_type_t*>(const_cast<TensorImpl&>(b).data()),
            static_cast<out_type_t*>(c_storage.data()), meta.m, meta.n, meta.k,
            meta.batch_count, a.strides()[a_rank - 2],
            a.strides()[a_rank - 1], b.strides()[b_rank - 2],
            b.strides()[b_rank - 1], batch_offsets);
        return TensorImpl(c_storage, meta.shape, meta.dtype);
      });
    });
  });
}

#define DECLARE_REDUCT_OPS_CUDA(args) DECLARE_REDUCT_OPS_CUDA_HELPER args
#define DECLARE_REDUCT_OPS_CUDA_HELPER(op, functor)        \
  TensorImpl op##_cuda(const TensorImpl& a, size_t axis) { \
    TensorMetaHandler meta(&a, axis);                      \
    reduct_dispatch_handler<functor>(meta, axis);          \
    return meta.out();                                     \
  }

LMP_FOR_EACH_CARTESIAN_PRODUCT(DECLARE_REDUCT_OPS_CUDA,
                               ((sum, SumFunctor), (max, MaxFunctor),
                                (min, MinFunctor), (prod, ProdFunctor), ));

LMP_REGISTER_DISPATCH(ops::add_stub, DeviceType::CUDA, add_cuda);
LMP_REGISTER_DISPATCH(ops::sub_stub, DeviceType::CUDA, sub_cuda);
LMP_REGISTER_DISPATCH(ops::mul_stub, DeviceType::CUDA, mul_cuda);
LMP_REGISTER_DISPATCH(ops::div_stub, DeviceType::CUDA, div_cuda);
LMP_REGISTER_DISPATCH(ops::pow_stub, DeviceType::CUDA, pow_cuda);
LMP_REGISTER_DISPATCH(ops::eq_stub, DeviceType::CUDA, eq_cuda);
LMP_REGISTER_DISPATCH(ops::ne_stub, DeviceType::CUDA, ne_cuda);
LMP_REGISTER_DISPATCH(ops::le_stub, DeviceType::CUDA, le_cuda);
LMP_REGISTER_DISPATCH(ops::lt_stub, DeviceType::CUDA, lt_cuda);
LMP_REGISTER_DISPATCH(ops::ge_stub, DeviceType::CUDA, ge_cuda);
LMP_REGISTER_DISPATCH(ops::gt_stub, DeviceType::CUDA, gt_cuda);

LMP_REGISTER_DISPATCH(ops::neg_stub, DeviceType::CUDA, neg_cuda);
LMP_REGISTER_DISPATCH(ops::log_stub, DeviceType::CUDA, log_cuda);
LMP_REGISTER_DISPATCH(ops::exp_stub, DeviceType::CUDA, exp_cuda);
LMP_REGISTER_DISPATCH(ops::sqrt_stub, DeviceType::CUDA, sqrt_cuda);
LMP_REGISTER_DISPATCH(ops::abs_stub, DeviceType::CUDA, abs_cuda);
LMP_REGISTER_DISPATCH(ops::sin_stub, DeviceType::CUDA, sin_cuda);
LMP_REGISTER_DISPATCH(ops::cos_stub, DeviceType::CUDA, cos_cuda);
LMP_REGISTER_DISPATCH(ops::tan_stub, DeviceType::CUDA, tan_cuda);
LMP_REGISTER_DISPATCH(ops::clamp_stub, DeviceType::CUDA, clamp_cuda);
LMP_REGISTER_DISPATCH(ops::abs_backward_stub, DeviceType::CUDA,
                      abs_backward_cuda);
LMP_REGISTER_DISPATCH(ops::clamp_backward_stub, DeviceType::CUDA,
                      clamp_backward_cuda);

LMP_REGISTER_DISPATCH(ops::matmul_stub, DeviceType::CUDA, matmul_cuda);
LMP_REGISTER_DISPATCH(ops::transpose_stub, DeviceType::CUDA, transpose_cuda);

LMP_REGISTER_DISPATCH(ops::sum_stub, DeviceType::CUDA, sum_cuda);
LMP_REGISTER_DISPATCH(ops::max_stub, DeviceType::CUDA, max_cuda);
LMP_REGISTER_DISPATCH(ops::min_stub, DeviceType::CUDA, min_cuda);
LMP_REGISTER_DISPATCH(ops::prod_stub, DeviceType::CUDA, prod_cuda);

}  // namespace lmp::tensor::detail::cuda
