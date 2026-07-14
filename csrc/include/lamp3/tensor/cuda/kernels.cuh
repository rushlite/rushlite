#pragma once

#include <cuda/std/array>
#include "lamp3/tensor/dispatch_stub.hpp"
#include "lamp3/tensor/native/matrix_ops.hpp"
#include "lamp3/tensor/native/expand_ops.hpp"
#include "lamp3/tensor/native/reduct_ops.hpp"
#include "lamp3/tensor/native/unary_ops.hpp"
#include "lamp3/tensor/tensor_impl.hpp"
#include <cmath>
#include <type_traits>

namespace lmp::tensor::detail::cuda {

/// @internal
/**
 * @brief Compute type for the transcendental functors.
 * @details Floating-point inputs are kept in their own precision so that
 * `float` dispatches to the FP32 SFU intrinsics (e.g. `expf`) instead of being
 * promoted to FP64, which runs many times slower on the SFU/FP64 units.
 * Integral inputs are promoted to `double` so the math is still real-valued
 * before being truncated back to the integer result type.
 */
template <typename T>
using math_acc_t = std::conditional_t<std::is_floating_point_v<T>, T, double>;
template <typename T>
struct AddFunctor {
  __device__ __host__ T operator()(T arg1, T arg2) { return arg1 + arg2; }
};
template <typename T>
struct SubFunctor {
  __device__ __host__ T operator()(T arg1, T arg2) { return arg1 - arg2; }
};
template <typename T>
struct MulFunctor {
  __device__ __host__ T operator()(T arg1, T arg2) { return arg1 * arg2; }
};
template <typename T>
struct DivFunctor {
  __device__ __host__ T operator()(T arg1, T arg2) { return arg1 / arg2; }
};
template <typename T>
struct PowFunctor {
  __device__ __host__ T operator()(T arg1, T arg2) {
    return static_cast<T>(std::pow(static_cast<math_acc_t<T>>(arg1),
                                   static_cast<math_acc_t<T>>(arg2)));
  }
};
template <typename T>
struct EqFunctor {
  __device__ __host__ T operator()(T arg1, T arg2) { return arg1 == arg2; }
};
template <typename T>
struct NeFunctor {
  __device__ __host__ T operator()(T arg1, T arg2) { return arg1 != arg2; }
};
template <typename T>
struct LeFunctor {
  __device__ __host__ T operator()(T arg1, T arg2) { return arg1 <= arg2; }
};
template <typename T>
struct LtFunctor {
  __device__ __host__ T operator()(T arg1, T arg2) { return arg1 < arg2; }
};
template <typename T>
struct GtFunctor {
  __device__ __host__ T operator()(T arg1, T arg2) { return arg1 > arg2; }
};
template <typename T>
struct GeFunctor {
  __device__ __host__ T operator()(T arg1, T arg2) { return arg1 >= arg2; }
};
template <typename T>
struct NegFunctor {
  __device__ __host__ T operator()(T arg) { return (-arg); }
};
template <typename T>
struct LogFunctor {
  __device__ __host__ T operator()(T arg) {
    return static_cast<T>(std::log(static_cast<math_acc_t<T>>(arg)));
  }
};
template <typename T>
struct ExpFunctor {
  __device__ __host__ T operator()(T arg) {
    return static_cast<T>(std::exp(static_cast<math_acc_t<T>>(arg)));
  }
};
template <typename T>
struct SqrtFunctor {
  __device__ __host__ T operator()(T arg) {
    return static_cast<T>(std::sqrt(static_cast<math_acc_t<T>>(arg)));
  }
};
template <typename T>
struct AbsFunctor {
  __device__ __host__ T operator()(T arg) {
    return static_cast<T>(std::abs(static_cast<math_acc_t<T>>(arg)));
  }
};
template <typename T>
struct SinFunctor {
  __device__ __host__ T operator()(T arg) {
    return static_cast<T>(std::sin(static_cast<math_acc_t<T>>(arg)));
  }
};
template <typename T>
struct CosFunctor {
  __device__ __host__ T operator()(T arg) {
    return static_cast<T>(std::cos(static_cast<math_acc_t<T>>(arg)));
  }
};
template <typename T>
struct TanFunctor {
  __device__ __host__ T operator()(T arg) {
    return static_cast<T>(std::tan(static_cast<math_acc_t<T>>(arg)));
  }
};
template <typename T>
struct ClampFunctor {
  explicit ClampFunctor(Scalar min_val, Scalar max_val)
      : min_val_(min_val), max_val_(max_val) {}
  __device__ __host__ T operator()(T arg) {
    return arg < min_val_ ? min_val_ : (arg > max_val_ ? max_val_ : arg);
  }

 private:
  Scalar min_val_, max_val_;
};
template <typename T>
struct SumFunctor {
  static constexpr T kIdentity = 0;
  __device__ __host__ T operator()(T arg1, T arg2) { return arg1 + arg2; }
};
template <typename T>
struct MaxFunctor {
  static constexpr T kIdentity = std::numeric_limits<T>::lowest();
  __device__ __host__ T operator()(T arg1, T arg2) { return max(arg1, arg2); }
};
template <typename T>
struct MinFunctor {
  static constexpr T kIdentity = std::numeric_limits<T>::max();
  __device__ __host__ T operator()(T arg1, T arg2) { return min(arg1, arg2); }
};
template <typename T>
struct ProdFunctor {
  static constexpr T kIdentity = 1;
  __device__ __host__ T operator()(T arg1, T arg2) { return arg1 * arg2; }
};

TensorImpl add_cuda(const TensorImpl& a, const TensorImpl& b);
TensorImpl sub_cuda(const TensorImpl& a, const TensorImpl& b);
TensorImpl mul_cuda(const TensorImpl& a, const TensorImpl& b);
TensorImpl div_cuda(const TensorImpl& a, const TensorImpl& b);
TensorImpl pow_cuda(const TensorImpl& a, const TensorImpl& b);
TensorImpl eq_cuda(const TensorImpl& a, const TensorImpl& b);
TensorImpl ne_cuda(const TensorImpl& a, const TensorImpl& b);
TensorImpl le_cuda(const TensorImpl& a, const TensorImpl& b);
TensorImpl lt_cuda(const TensorImpl& a, const TensorImpl& b);
TensorImpl ge_cuda(const TensorImpl& a, const TensorImpl& b);
TensorImpl gt_cuda(const TensorImpl& a, const TensorImpl& b);

TensorImpl neg_cuda(const TensorImpl& a);
TensorImpl log_cuda(const TensorImpl& a);
TensorImpl exp_cuda(const TensorImpl& a);
TensorImpl sqrt_cuda(const TensorImpl& a);
TensorImpl abs_cuda(const TensorImpl& a);
TensorImpl sin_cuda(const TensorImpl& a);
TensorImpl cos_cuda(const TensorImpl& a);
TensorImpl tan_cuda(const TensorImpl& a);
TensorImpl clamp_cuda(const TensorImpl& a, Scalar min_val, Scalar max_val);

TensorImpl transpose_cuda(const TensorImpl& a);
TensorImpl matmul_cuda(const TensorImpl& a, const TensorImpl& b);

TensorImpl sum_cuda(const TensorImpl& a, size_t axis);
TensorImpl max_cuda(const TensorImpl& a, size_t axis);
TensorImpl min_cuda(const TensorImpl& a, size_t axis);
TensorImpl prod_cuda(const TensorImpl& a, size_t axis);
/// @endinternal

}  // namespace lmp::tensor::detail::cuda