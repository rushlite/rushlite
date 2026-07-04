#pragma once

namespace lmp::autograd {

class GradGuard {
public:
    explicit GradGuard(bool grad_enabled);
    ~GradGuard();

private:
bool prev;
};

bool is_grad_enabled();

void set_grad_enabled(bool enable);

}  // namespace lmp::autograd
