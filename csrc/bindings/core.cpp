#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lamp3/autograd/core.hpp"
#include "lamp3/tensor/core.hpp"

#include "data_type.hpp"
#include "device_type.hpp"
#include "cuda.hpp"
#include "tensor.hpp"
#include "variable.hpp"
#include "constructor.hpp"
#include "capture_mode.hpp"
#include "grad_mode.hpp"
#include "functions/binary.hpp"
#include "functions/unary.hpp"
#include "functions/matrix.hpp"
#include "functions/reduct.hpp"
#include "functions/view.hpp"

PYBIND11_MODULE(_C, m) {
    init_data_type(m);
    init_device_type(m);
    init_cuda(m);
    init_tensor(m);
    init_variable(m);
    init_constructor(m);
    init_capture_mode(m);
    init_grad_mode(m);

    init_binary(m);
    init_unary(m);
    init_matrix(m);
    init_reduct(m);
    init_view(m);
}
