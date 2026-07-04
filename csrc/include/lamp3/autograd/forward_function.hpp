#pragma once

#include <numeric>
#include "function.hpp"
#include "grad_mode.hpp"
#include "variable.hpp"

namespace lmp::autograd {

/// @internal
template <typename Derived>
struct ForwardFunction : public Function {
  static bool requires_grad(const variable_list& variables) {
    return std::accumulate(variables.begin(), variables.end(), false,
                           [](bool accumulated, const Variable& b) {
                             return accumulated || b.requires_grad();
                           });
  }

  variable_list apply(const variable_list&  /*inputs*/) override {
    throw std::runtime_error(
        "Forward function should not be called without template args");
    return {};
  }

  template <typename... Args>
  variable_list apply(const variable_list& inputs, Args&&... args) {
    bool requires_grad_ = is_grad_enabled() && requires_grad(inputs);
    Variable result =
        Variable(static_cast<Derived*>(this)->execute(inputs), requires_grad_);
    if (requires_grad_) {
      auto backward_fn = std::make_shared<typename Derived::DefaultBackward>(
          std::forward<Args>(args)...);
      backward_fn->saved_inputs = std::make_unique<variable_list>(inputs);
      result.set_grad_fn(backward_fn);
    }

    return {result};
  }
};
/// @endinternal

}  // namespace lmp::autograd
