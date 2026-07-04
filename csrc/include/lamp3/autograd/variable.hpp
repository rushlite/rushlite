#pragma once

#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>

#include "lamp3/tensor/core.hpp"
#include "lamp3/tensor/data_type.hpp"
#include "lamp3/tensor/tensor.hpp"
#include "lamp3/tensor/utils/fill_like.hpp"

namespace lmp::autograd {

class Function;
class Variable;

struct VariableImpl {
  tensor::Tensor data;
  tensor::Tensor grad;
  std::shared_ptr<Function> _grad_fn;
  bool requires_grad;

  explicit VariableImpl(const tensor::Tensor& data, bool requires_grad = false)
      : data(tensor::Tensor(data)),
        grad(requires_grad ? zeros_like(data) : tensor::Tensor()),
        requires_grad(requires_grad),
        _grad_fn(nullptr) {}
  explicit VariableImpl(const tensor::Tensor& data, const tensor::Tensor& grad,
                        bool requires_grad,
                        const std::shared_ptr<Function>& grad_fn)
      : data(tensor::Tensor(data)),
        grad(tensor::Tensor(grad)),
        requires_grad(requires_grad),
        _grad_fn(std::move(grad_fn)) {}
};

class Variable {
 public:
  Variable() = default;
  explicit Variable(const tensor::Tensor& data, bool requires_grad = false)
      : impl_(std::make_shared<VariableImpl>(data, requires_grad)) {}

  const tensor::Tensor& grad() const;
  const tensor::Tensor& data() const noexcept;
  std::weak_ptr<Function> grad_fn() const noexcept;
  bool requires_grad() const noexcept;

  void zero_grad();
  void incr_grad(const tensor::Tensor& other_grad);
  void set_grad_fn(std::shared_ptr<Function> grad_fn);

  void copy(const Variable& other);
  void fill(tensor::Scalar item);

  void backward();
  friend std::ostream& operator<<(std::ostream& os, const Variable& obj);

 private:
  explicit Variable(std::shared_ptr<VariableImpl> impl)
      : impl_(std::move(impl)) {}
  std::shared_ptr<VariableImpl> impl_;
  std::vector<Variable> topological_sort();
  void dfs(const Variable& v, std::unordered_set<void*>& visited,
           std::vector<Variable>& topo) const;
};

using variable_list = std::vector<Variable>;

struct VariableOpFact {
  template <typename Op, typename... Args>
  static variable_list apply(const variable_list& variables, Args&&... args) {
    Op op_fn(std::forward<Args>(args)...);
    variable_list result =
        op_fn.template apply<Args...>(variables, std::forward<Args>(args)...);
    return result;
  }
};

}  // namespace lmp::autograd
