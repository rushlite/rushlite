#pragma once

#include "lamp3/tensor/device_type.hpp"
#include "lamp3/tensor/dispatch_stub.hpp"
#include "lamp3/tensor/tensor.hpp"
#include "lamp3/tensor/tensor_impl.hpp"

namespace lmp::tensor::ops {

/// @internal
using neg_fn = TensorImpl (*)(const TensorImpl&);
using exp_fn = TensorImpl (*)(const TensorImpl&);
using log_fn = TensorImpl (*)(const TensorImpl&);
using sqrt_fn = TensorImpl (*)(const TensorImpl&);
using abs_fn = TensorImpl (*)(const TensorImpl&);
using sin_fn = TensorImpl (*)(const TensorImpl&);
using cos_fn = TensorImpl (*)(const TensorImpl&);
using tan_fn = TensorImpl (*)(const TensorImpl&);
using clamp_fn = TensorImpl (*)(const TensorImpl&, Scalar, Scalar);
using abs_backward_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&);
using clamp_backward_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&,
                                         Scalar, Scalar);

LMP_DECLARE_DISPATCH(neg_fn, neg_stub);
LMP_DECLARE_DISPATCH(exp_fn, exp_stub);
LMP_DECLARE_DISPATCH(log_fn, log_stub);
LMP_DECLARE_DISPATCH(sqrt_fn, sqrt_stub);
LMP_DECLARE_DISPATCH(abs_fn, abs_stub);
LMP_DECLARE_DISPATCH(sin_fn, sin_stub);
LMP_DECLARE_DISPATCH(cos_fn, cos_stub);
LMP_DECLARE_DISPATCH(tan_fn, tan_stub);
LMP_DECLARE_DISPATCH(clamp_fn, clamp_stub);
LMP_DECLARE_DISPATCH(abs_backward_fn, abs_backward_stub);
LMP_DECLARE_DISPATCH(clamp_backward_fn, clamp_backward_stub);
/// @endinternal

/**
 * @brief Negate a tensor
 * @param a The tensor to negate
 * @return A new tensor with the result of the negation
 */
Tensor neg(const Tensor& a);

/**
 * @brief Exponentiate a tensor
 * @param a The tensor to exponentiate
 * @return A new tensor with the result of the exponentiation
 */
Tensor exp(const Tensor& a);

/**
 * @brief Logarithm of a tensor
 * @param a The tensor to take the logarithm of
 * @return A new tensor with the result of the logarithm
 */
Tensor log(const Tensor& a);

/**
 * @brief Square root of a tensor
 * @param a The tensor to take the square root of
 * @return A new tensor with the result of the square root
 */
Tensor sqrt(const Tensor& a);

/**
 * @brief Absolute value of a tensor
 * @param a The tensor to take the absolute value of
 * @return A new tensor with the result of the absolute value
 */
Tensor abs(const Tensor& a);

/**
 * @brief Sine of a tensor
 * @param a The tensor to take the sine of
 * @return A new tensor with the result of the sine
 */
Tensor sin(const Tensor& a);

/**
 * @brief Cosine of a tensor
 * @param a The tensor to take the cosine of
 * @return A new tensor with the result of the cosine
 */
Tensor cos(const Tensor& a);

/**
 * @brief Tangent of a tensor
 * @param a The tensor to take the tangent of
 * @return A new tensor with the result of the tangent
 */
Tensor tan(const Tensor& a);

/**
 * @brief Clamp a tensor
 * @param a The tensor to clamp
 * @param min_val The minimum value
 * @param max_val The maximum value
 * @return A new tensor with the result of the clamping
 */
Tensor clamp(const Tensor& a, Scalar min_val, Scalar max_val);

/// @internal
Tensor abs_backward(const Tensor& input, const Tensor& grad_output);
Tensor clamp_backward(const Tensor& input, const Tensor& grad_output,
                      Scalar min_val, Scalar max_val);
/// @endinternal

}  // namespace lmp::tensor::ops
