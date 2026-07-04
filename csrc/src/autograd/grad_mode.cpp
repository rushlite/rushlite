#include "lamp3/autograd/grad_mode.hpp"

namespace lmp::autograd {

bool thread_local grad_enabled = true;

GradModeGuard::GradModeGuard(bool grad_enabled) {
  prev = is_grad_enabled();
  set_grad_enabled(grad_enabled);
}

GradModeGuard::~GradModeGuard() { set_grad_enabled(prev); }

bool is_grad_enabled() { return grad_enabled; }

void set_grad_enabled(bool enable) { grad_enabled = enable; }

}  // namespace lmp::autograd
