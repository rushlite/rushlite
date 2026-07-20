#include "lamp3/autograd/functions/matrix_ops.hpp"

#include "lamp3/autograd/function.hpp"
#include "lamp3/autograd/utils/grad_utils.hpp"
#include "lamp3/autograd/variable.hpp"

namespace lmp::autograd::ops {
namespace {

tensor::Tensor transpose_last_two(const tensor::Tensor& input) {
  LMP_INTERNAL_ASSERT(input.shape().size() >= 2)
      << "Matmul operands must have rank at least two";
  const size_t rank = input.shape().size();
  return input.transpose(rank - 2, rank - 1);
}

}  // namespace

variable_list MatrixMultiplicationBackward::apply(
    const variable_list& gradOutputs) {
  LMP_INTERNAL_ASSERT(gradOutputs.size() == 1) << "Output size mismatch.";
  const Variable& grad = gradOutputs[0];
  Variable& self = (*saved_inputs)[0];
  Variable& other = (*saved_inputs)[1];

  if (self.requires_grad()) {
    tensor::Tensor self_grad =
        tensor::ops::matmul(grad.grad(), transpose_last_two(other.data()));
    self.incr_grad(
        detail::sum_broadcast_axis(self_grad, self.data().shape()));
  }
  if (other.requires_grad()) {
    tensor::Tensor other_grad =
        tensor::ops::matmul(transpose_last_two(self.data()), grad.grad());
    other.incr_grad(
        detail::sum_broadcast_axis(other_grad, other.data().shape()));
  }

  variable_list grad_inputs = {};
  return grad_inputs;
}

variable_list TransposeBackward::apply(const variable_list& gradOutputs) {
  LMP_INTERNAL_ASSERT(gradOutputs.size() == 1) << "Output size mismatch.";
  const Variable& grad = gradOutputs[0];
  Variable& self = (*saved_inputs)[0];

  if (self.requires_grad()) self.incr_grad(tensor::ops::transpose(grad.grad()));

  variable_list grad_inputs = {};
  return grad_inputs;
}

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
tensor::Tensor MatrixMultiplication::execute(const variable_list& inputs) {
  LMP_INTERNAL_ASSERT(inputs.size() == 2) << "Function must take 2 inputs";
  const Variable& self = inputs[0];
  const Variable& other = inputs[1];

  return tensor::ops::matmul(self.data(), other.data());
}

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
tensor::Tensor Transpose::execute(const variable_list& inputs) {
  LMP_INTERNAL_ASSERT(inputs.size() == 1) << "Function must take one input";
  const Variable& self = inputs[0];

  return tensor::ops::transpose(self.data());
}

}  // namespace lmp::autograd::ops
