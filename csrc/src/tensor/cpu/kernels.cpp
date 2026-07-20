#include "lamp3/tensor/cpu/kernels.hpp"

#include "lamp3/common/macros.hpp"
#include "lamp3/tensor/cpu/binary.hpp"
#include "lamp3/tensor/cpu/matrix.hpp"
#include "lamp3/tensor/cpu/meta_handler.hpp"
#include "lamp3/tensor/cpu/reduct.hpp"
#include "lamp3/tensor/cpu/unary.hpp"
#include "lamp3/tensor/infer_meta.hpp"
#include "lamp3/tensor/native/expand_ops.hpp"
#include "lamp3/tensor/native/reduct_ops.hpp"

namespace lmp::tensor::detail::cpu {

#define DECLARE_BINARY_OPS_CPU(args) DECLARE_BINARY_OPS_CPU_HELPER args
#define DECLARE_BINARY_OPS_CPU_HELPER(op, functor)                \
  TensorImpl op##_cpu(const TensorImpl& a, const TensorImpl& b) { \
    TensorMetaHandler meta(&a, &b);                               \
    binary_dispatch_handler<functor>(meta);                       \
    return meta.out();                                            \
  }

LMP_FOR_EACH_CARTESIAN_PRODUCT(
    DECLARE_BINARY_OPS_CPU,
    ((add, AddFunctor), (sub, SubFunctor), (mul, MulFunctor), (div, DivFunctor),
     (pow, PowFunctor), (eq, EqFunctor), (ne, NeFunctor), (le, LeFunctor),
     (lt, LtFunctor), (ge, GeFunctor), (gt, GtFunctor), ));

#define DECLARE_UNARY_OPS_CPU(args) DECLARE_UNARY_OPS_CPU_HELPER args
#define DECLARE_UNARY_OPS_CPU_HELPER(op, functor) \
  TensorImpl op##_cpu(const TensorImpl& a) {      \
    TensorMetaHandler meta(&a);                   \
    unary_dispatch_handler<functor>(meta);        \
    return meta.out();                            \
  }

LMP_FOR_EACH_CARTESIAN_PRODUCT(DECLARE_UNARY_OPS_CPU,
                               ((neg, NegFunctor), (log, LogFunctor),
                                (exp, ExpFunctor), (sqrt, SqrtFunctor),
                                (abs, AbsFunctor), (sin, SinFunctor),
                                (cos, CosFunctor), (tan, TanFunctor), ));

TensorImpl clamp_cpu(const TensorImpl& a, Scalar min_val, Scalar max_val) {
  TensorMetaHandler meta(&a);
  unary_dispatch_handler<ClampFunctor>(meta, min_val, max_val);
  return meta.out();
}

TensorImpl abs_backward_cpu(const TensorImpl& input,
                            const TensorImpl& grad_output) {
  BinaryMetaHandler meta(&input, &grad_output);
  binary_dispatch_handler<AbsBackwardFunctor>(meta);
  return meta.out();
}

TensorImpl clamp_backward_cpu(const TensorImpl& input,
                              const TensorImpl& grad_output, Scalar min_val,
                              Scalar max_val) {
  BinaryMetaHandler meta(&input, &grad_output);
  binary_dispatch_handler<ClampBackwardFunctor>(meta, min_val, max_val);
  return meta.out();
}

TensorImpl transpose_cpu(const TensorImpl& a) {
  LMP_CHECK(a.shape().size() == 2) << "Invalid argument, transpose can only be "
                                      "performed on matrices of dim 2";
  size_t m = a.shape()[0];
  size_t n = a.shape()[1];

  DataType out_dtype = a.type();

  return LMP_DISPATCH_ALL_TYPES(a.type(), [&] {
    Storage c_storage(m * n * sizeof(scalar_t), DeviceType::CPU);
    ::lmp::tensor::detail::cpu::cpuTranspose<scalar_t>(
        static_cast<const scalar_t*>(const_cast<TensorImpl&>(a).data()),
        static_cast<scalar_t*>(c_storage.data()), m, n);
    return TensorImpl(c_storage, {n, m}, out_dtype);
  });
}

// NOLINTNEXTLINE(readability-function-size,google-readability-function-size)
TensorImpl matmul_cpu(const TensorImpl& a, const TensorImpl& b) {
  const MatmulMeta meta = infer_matmul(&a, &b);
  std::unique_ptr<OffsetUtil> offset = offset_util_stub_2()(
      DeviceType::CPU, meta.batch_shape,
      std::array<OperandLayout, 2>{leading_operand_layout(a),
                                   leading_operand_layout(b)});
  const auto& batch_offsets =
      static_cast<const CPUOffsetUtil<2>*>(offset.get())->calculator();
  const size_t a_rank = a.shape().size();
  const size_t b_rank = b.shape().size();

  return LMP_DISPATCH_ALL_TYPES(a.type(), [&] {
    using a_type_t = scalar_t;
    return LMP_DISPATCH_ALL_TYPES(b.type(), [&] {
      using b_type_t = scalar_t;
      return LMP_DISPATCH_ALL_TYPES(meta.dtype, [&] {
        using out_type_t = scalar_t;
        Storage c_storage(meta.size * sizeof(out_type_t), DeviceType::CPU);
        ::lmp::tensor::detail::cpu::cpuMatMul<a_type_t, b_type_t, out_type_t>(
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

#define DECLARE_REDUCT_OPS_CPU(args) DECLARE_REDUCT_OPS_CPU_HELPER args
#define DECLARE_REDUCT_OPS_CPU_HELPER(op, functor)        \
  TensorImpl op##_cpu(const TensorImpl& a, size_t axis) { \
    TensorMetaHandler meta(&a, axis);                     \
    reduct_dispatch_handler<functor>(meta, axis);         \
    return meta.out();                                    \
  }

LMP_FOR_EACH_CARTESIAN_PRODUCT(DECLARE_REDUCT_OPS_CPU,
                               ((sum, SumFunctor), (max, MaxFunctor),
                                (min, MinFunctor), (prod, ProdFunctor), ));

LMP_REGISTER_DISPATCH(ops::add_stub, DeviceType::CPU, add_cpu);
LMP_REGISTER_DISPATCH(ops::sub_stub, DeviceType::CPU, sub_cpu);
LMP_REGISTER_DISPATCH(ops::mul_stub, DeviceType::CPU, mul_cpu);
LMP_REGISTER_DISPATCH(ops::div_stub, DeviceType::CPU, div_cpu);
LMP_REGISTER_DISPATCH(ops::pow_stub, DeviceType::CPU, pow_cpu);
LMP_REGISTER_DISPATCH(ops::eq_stub, DeviceType::CPU, eq_cpu);
LMP_REGISTER_DISPATCH(ops::ne_stub, DeviceType::CPU, ne_cpu);
LMP_REGISTER_DISPATCH(ops::le_stub, DeviceType::CPU, le_cpu);
LMP_REGISTER_DISPATCH(ops::lt_stub, DeviceType::CPU, lt_cpu);
LMP_REGISTER_DISPATCH(ops::ge_stub, DeviceType::CPU, ge_cpu);
LMP_REGISTER_DISPATCH(ops::gt_stub, DeviceType::CPU, gt_cpu);

LMP_REGISTER_DISPATCH(ops::neg_stub, DeviceType::CPU, neg_cpu);
LMP_REGISTER_DISPATCH(ops::abs_stub, DeviceType::CPU, abs_cpu);
LMP_REGISTER_DISPATCH(ops::clamp_stub, DeviceType::CPU, clamp_cpu);
LMP_REGISTER_DISPATCH(ops::abs_backward_stub, DeviceType::CPU,
                      abs_backward_cpu);
LMP_REGISTER_DISPATCH(ops::clamp_backward_stub, DeviceType::CPU,
                      clamp_backward_cpu);
LMP_REGISTER_DISPATCH(ops::cos_stub, DeviceType::CPU, cos_cpu);
LMP_REGISTER_DISPATCH(ops::exp_stub, DeviceType::CPU, exp_cpu);
LMP_REGISTER_DISPATCH(ops::log_stub, DeviceType::CPU, log_cpu);
LMP_REGISTER_DISPATCH(ops::sin_stub, DeviceType::CPU, sin_cpu);
LMP_REGISTER_DISPATCH(ops::sqrt_stub, DeviceType::CPU, sqrt_cpu);
LMP_REGISTER_DISPATCH(ops::tan_stub, DeviceType::CPU, tan_cpu);

LMP_REGISTER_DISPATCH(ops::transpose_stub, DeviceType::CPU, transpose_cpu);
LMP_REGISTER_DISPATCH(ops::matmul_stub, DeviceType::CPU, matmul_cpu);

LMP_REGISTER_DISPATCH(ops::sum_stub, DeviceType::CPU, sum_cpu);
LMP_REGISTER_DISPATCH(ops::max_stub, DeviceType::CPU, max_cpu);
LMP_REGISTER_DISPATCH(ops::min_stub, DeviceType::CPU, min_cpu);
LMP_REGISTER_DISPATCH(ops::prod_stub, DeviceType::CPU, prod_cpu);

}  // namespace lmp::tensor::detail::cpu
