
#include "lamp3/autograd/functions/unary_ops.hpp"

#include <cmath>

#include "lamp3/autograd/functions/overloads.hpp"
#include "lamp3/autograd/functions/unary_decl.hpp"
#include "lamp3/autograd/variable.hpp"
#include "lamp3/common/assert.hpp"
#include "lamp3/common/macros.hpp"
#include "lamp3/tensor/tensor.hpp"
#include "lamp3/tensor/utils/fill_like.hpp"

namespace lmp::autograd::ops {

LMP_FOR_EACH_CARTESIAN_PRODUCT(
    LMP_AUTOGRAD_FN_UNARY_DECL,
    ((NegationBackward, -grad.grad()),
     (ExponentialBackward, grad.data() * grad.grad()),
     (LogarithmBackward, grad.grad() / self.data()),
     (SquareRootBackward, grad.grad() / (grad.data() + grad.data())),
     (AbsoluteValueBackward,
      tensor::ops::abs_backward(self.data(), grad.grad())),
     (SineBackward, grad.grad() * tensor::ops::cos(self.data())),
     (CosineBackward, (-tensor::ops::sin(self.data())) * grad.grad()),
     (TangentBackward,
      grad.grad() + (grad.grad() * grad.data() * grad.data())),
     (ClampBackward,
      tensor::ops::clamp_backward(self.data(), grad.grad(), min_val_,
                                  max_val_))));

LMP_FOR_EACH_CARTESIAN_PRODUCT(
    LMP_AUTOGRAD_FFN_UNARY_DECL,
    ((Negation, tensor::ops::neg), (Exponential, tensor::ops::exp),
     (Logarithm, tensor::ops::log), (SquareRoot, tensor::ops::sqrt),
     (AbsoluteValue, tensor::ops::abs), (Sine, tensor::ops::sin),
     (Cosine, tensor::ops::cos), (Tangent, tensor::ops::tan),
     (Clamp, tensor::ops::clamp, min_val_, max_val_)));

}  // namespace lmp::autograd::ops
