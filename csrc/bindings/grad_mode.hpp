#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lamp3/autograd/grad_mode.hpp"

namespace py = pybind11;

inline void init_grad_mode(py::module_& m) {
    m.def("set_grad_enabled", &lmp::autograd::set_grad_enabled);
    m.def("is_grad_enabled", &lmp::autograd::is_grad_enabled);
}
