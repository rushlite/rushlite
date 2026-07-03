#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lamp3/autograd/core.hpp"

namespace py = pybind11;

inline void init_matrix(py::module_& m) {
  m.def("matmul", &lmp::autograd::ops::matmul, py::arg("a"), py::arg("b"));
  m.def("transpose", &lmp::autograd::ops::transpose, py::arg("a"));
}
