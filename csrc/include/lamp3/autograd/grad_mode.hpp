#pragma once

namespace lmp::autograd {

class GradModeGuard {
public:
    explicit GradModeGuard(bool grad_enabled);
    ~GradModeGuard();

private:
bool prev;
};

bool is_grad_enabled();

void set_grad_enabled(bool enable);

}  // namespace lmp::autograd
