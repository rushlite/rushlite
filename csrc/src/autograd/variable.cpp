#include "lamp3/autograd/variable.hpp"

#include <algorithm>
#include <iostream>
#include <memory>
#include <unordered_set>

#include "lamp3/autograd/function.hpp"
#include "lamp3/common/assert.hpp"

namespace lmp::autograd {

const tensor::Tensor& Variable::grad() const {
  LMP_CHECK(impl_->requires_grad)
      << "Cannot access grad() if requires_grad = false";
  return impl_->grad;
}
const tensor::Tensor& Variable::data() const noexcept { return impl_->data; }
std::weak_ptr<Function> Variable::grad_fn() const noexcept {
  std::weak_ptr<Function> grad_fn = impl_->_grad_fn;
  return grad_fn;
}
bool Variable::requires_grad() const noexcept { return impl_->requires_grad; }

void Variable::zero_grad() {
  LMP_CHECK(requires_grad()) << "Cannot access grad() if requires_grad = false";
  impl_->grad = zeros_like(impl_->grad);
  impl_->_grad_fn = nullptr;
}  // TODO(root): this can be better, implement fill in tensor
void Variable::incr_grad(const tensor::Tensor& other_grad) {
  LMP_CHECK(requires_grad()) << "Cannot access grad() if requires_grad = false";
  LMP_INTERNAL_ASSERT(other_grad.shape() == impl_->grad.shape())
      << "There should be no broadcasting in incr_grad";
  impl_->grad = impl_->grad + other_grad;
}
void Variable::set_grad_fn(std::shared_ptr<Function> grad_fn) {
  LMP_CHECK(requires_grad()) << "Cannot access grad() if requires_grad = false";
  impl_->_grad_fn = std::move(grad_fn);
}

// copy does NOT copy gradients, only data; similar to Pytorch behavior
void Variable::copy(const Variable& other) { impl_->data.copy(other.data()); }
void Variable::fill(tensor::Scalar item) { impl_->data.fill(item); }

void Variable::backward() {
  LMP_CHECK(requires_grad()) << "Must be declared with requires_grad";
  std::vector<Variable> topo = topological_sort();
  impl_->grad = ones_like(impl_->grad);
  for (Variable& node : topo) {
    if (node.grad_fn().lock() != nullptr) {
      node.grad_fn().lock()->apply({node});
    }
  }
}

void Variable::dfs(const Variable& v,  // NOLINT
                   std::unordered_set<void*>& visited,
                   std::vector<Variable>& topo) const {
  if (!visited.contains(static_cast<void*>(v.impl_.get()))) {
    visited.insert(static_cast<void*>(v.impl_.get()));
    if (v.grad_fn().lock() == nullptr) {
      topo.push_back(v);
      return;
    }
    for (const auto& child : *(v.grad_fn().lock()->saved_inputs)) {
      dfs(child, visited, topo);
    }
    topo.push_back(v);
  }
}

std::vector<Variable> Variable::topological_sort() {
  std::unordered_set<void*> visited;
  std::vector<Variable> topo;

  dfs(*this, visited, topo);
  std::ranges::reverse(topo);
  return topo;
}

std::ostream& operator<<(std::ostream& os, const Variable& obj) {
  os << "Variable(requires_grad=" << obj.requires_grad();
  os << ", data=" << obj.data();
  if (obj.requires_grad()) {
    os << ", grad=" << obj.grad();
  } else {
    os << ", grad=" << nullptr;
  }
  os << ", grad_fn="
     << (obj.grad_fn().expired() ? nullptr : obj.grad_fn().lock()) << ")";
  return os;
}

}  // namespace lmp::autograd