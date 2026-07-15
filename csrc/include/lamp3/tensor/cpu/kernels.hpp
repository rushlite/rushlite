#pragma once

#include <cmath>
#include <type_traits>
#include "lamp3/tensor/device_type.hpp"
#include "lamp3/tensor/dispatch_stub.hpp"
#include "lamp3/tensor/native/matrix_ops.hpp"
#include "lamp3/tensor/native/unary_ops.hpp"
#include "lamp3/tensor/tensor_impl.hpp"

namespace lmp::tensor::detail::cpu {

/// @internal
/**
 * @brief Compute type for the transcendental functors.
 * @details Floating-point inputs stay in their own precision instead of being
 * promoted to `double`, which avoids doing every `float` op in FP64. Integral
 * inputs are promoted to `double` for a real-valued result before truncation.
 */
template <typename T>
using math_acc_t = std::conditional_t<std::is_floating_point_v<T>, T, double>;
template <typename T>
struct AddFunctor {
  T operator()(T arg1, T arg2) { return arg1 + arg2; }
};
template <typename T>
struct SubFunctor {
  T operator()(T arg1, T arg2) { return arg1 - arg2; }
};
template <typename T>
struct MulFunctor {
  T operator()(T arg1, T arg2) { return arg1 * arg2; }
};
template <typename T>
struct DivFunctor {
  T operator()(T arg1, T arg2) { return arg1 / arg2; }
};
template <typename T>
struct PowFunctor {
  T operator()(T arg1, T arg2) {
    return static_cast<T>(std::pow(static_cast<math_acc_t<T>>(arg1),
                                   static_cast<math_acc_t<T>>(arg2)));
  }
};
template <typename T>
struct EqFunctor {
  T operator()(T arg1, T arg2) { return arg1 == arg2; }
};
template <typename T>
struct NeFunctor {
  T operator()(T arg1, T arg2) { return arg1 != arg2; }
};
template <typename T>
struct LeFunctor {
  T operator()(T arg1, T arg2) { return arg1 <= arg2; }
};
template <typename T>
struct LtFunctor {
  T operator()(T arg1, T arg2) { return arg1 < arg2; }
};
template <typename T>
struct GtFunctor {
  T operator()(T arg1, T arg2) { return arg1 > arg2; }
};
template <typename T>
struct GeFunctor {
  T operator()(T arg1, T arg2) { return arg1 >= arg2; }
};
template <typename T>
struct NegFunctor {
  T operator()(T arg) { return (-arg); }
};
template <typename T>
struct LogFunctor {
  T operator()(T arg) {
    return static_cast<T>(std::log(static_cast<math_acc_t<T>>(arg)));
  }
};
template <typename T>
struct ExpFunctor {
  T operator()(T arg) {
    return static_cast<T>(std::exp(static_cast<math_acc_t<T>>(arg)));
  }
};
template <typename T>
struct SqrtFunctor {
  T operator()(T arg) {
    return static_cast<T>(std::sqrt(static_cast<math_acc_t<T>>(arg)));
  }
};
template <typename T>
struct AbsFunctor {
  T operator()(T arg) {
    return static_cast<T>(std::abs(static_cast<math_acc_t<T>>(arg)));
  }
};
template <typename T>
struct SinFunctor {
  T operator()(T arg) {
    return static_cast<T>(std::sin(static_cast<math_acc_t<T>>(arg)));
  }
};
template <typename T>
struct CosFunctor {
  T operator()(T arg) {
    return static_cast<T>(std::cos(static_cast<math_acc_t<T>>(arg)));
  }
};
template <typename T>
struct TanFunctor {
  T operator()(T arg) {
    return static_cast<T>(std::tan(static_cast<math_acc_t<T>>(arg)));
  }
};
template <typename T>
struct ClampFunctor {
  explicit ClampFunctor(Scalar min_val, Scalar max_val)
      : min_val_(static_cast<T>(min_val)), max_val_(static_cast<T>(max_val)) {}
  T operator()(T arg) {
    if (arg < min_val_) {
      return min_val_;
    }
    if (arg > max_val_) {
      return max_val_;
    }
    return arg;
  }

 private:
  T min_val_, max_val_;
};
template <typename T>
struct AbsBackwardFunctor {
  T operator()(T input, T grad) {
    if (input > T{0}) {
      return grad;
    }
    if (input < T{0}) {
      return -grad;
    }
    return T{0};
  }
};
template <typename T>
struct ClampBackwardFunctor {
  ClampBackwardFunctor(Scalar min_val, Scalar max_val)
      : min_val_(static_cast<T>(min_val)), max_val_(static_cast<T>(max_val)) {}
  T operator()(T input, T grad) {
    return input > min_val_ && input < max_val_ ? grad : T{0};
  }

 private:
  T min_val_, max_val_;
};
template <typename T>
struct SumFunctor {
  static constexpr T kIdentity = 0;
  T operator()(T arg1, T arg2) { return arg1 + arg2; }
};
template <typename T>
struct MaxFunctor {
  static constexpr T kIdentity = std::numeric_limits<T>::lowest();
  T operator()(T arg1, T arg2) { return ::std::max(arg1, arg2); }
};
template <typename T>
struct MinFunctor {
  static constexpr T kIdentity = std::numeric_limits<T>::max();
  T operator()(T arg1, T arg2) { return ::std::min(arg1, arg2); }
};
template <typename T>
struct ProdFunctor {
  static constexpr T kIdentity = 1;
  T operator()(T arg1, T arg2) { return arg1 * arg2; }
};
/// @endinternal

/// @internal
TensorImpl add_cpu(const TensorImpl& a, const TensorImpl& b);
TensorImpl sub_cpu(const TensorImpl& a, const TensorImpl& b);
TensorImpl mul_cpu(const TensorImpl& a, const TensorImpl& b);
TensorImpl div_cpu(const TensorImpl& a, const TensorImpl& b);
TensorImpl pow_cpu(const TensorImpl& a, const TensorImpl& b);
TensorImpl eq_cpu(const TensorImpl& a, const TensorImpl& b);
TensorImpl ne_cpu(const TensorImpl& a, const TensorImpl& b);
TensorImpl le_cpu(const TensorImpl& a, const TensorImpl& b);
TensorImpl lt_cpu(const TensorImpl& a, const TensorImpl& b);
TensorImpl ge_cpu(const TensorImpl& a, const TensorImpl& b);
TensorImpl gt_cpu(const TensorImpl& a, const TensorImpl& b);

TensorImpl neg_cpu(const TensorImpl& a);
TensorImpl log_cpu(const TensorImpl& a);
TensorImpl exp_cpu(const TensorImpl& a);
TensorImpl sqrt_cpu(const TensorImpl& a);
TensorImpl abs_cpu(const TensorImpl& a);
TensorImpl sin_cpu(const TensorImpl& a);
TensorImpl cos_cpu(const TensorImpl& a);
TensorImpl tan_cpu(const TensorImpl& a);
TensorImpl clamp_cpu(const TensorImpl& a, Scalar min_val, Scalar max_val);
TensorImpl abs_backward_cpu(const TensorImpl& input,
                            const TensorImpl& grad_output);
TensorImpl clamp_backward_cpu(const TensorImpl& input,
                              const TensorImpl& grad_output, Scalar min_val,
                              Scalar max_val);

TensorImpl matmul_cpu(const TensorImpl& a, const TensorImpl& b);
TensorImpl transpose_cpu(const TensorImpl& a);

TensorImpl sum_cpu(const TensorImpl& a, size_t axis);
TensorImpl max_cpu(const TensorImpl& a, size_t axis);
TensorImpl min_cpu(const TensorImpl& a, size_t axis);
TensorImpl prod_cpu(const TensorImpl& a, size_t axis);
/// @endinternal

}  // namespace lmp::tensor::detail::cpu
