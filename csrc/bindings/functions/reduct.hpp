#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lamp3/autograd/core.hpp"

namespace py = pybind11;

inline void init_reduct(py::module_& m) {
  m.def("sum", &lmp::autograd::ops::sum, py::arg("a"), py::arg("axis"));
  m.def("max", &lmp::autograd::ops::max, py::arg("a"), py::arg("axis"));
  m.def("min", &lmp::autograd::ops::min, py::arg("a"), py::arg("axis"));
  m.def("prod", &lmp::autograd::ops::prod, py::arg("a"), py::arg("axis"));
}
