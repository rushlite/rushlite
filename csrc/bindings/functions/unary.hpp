#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lamp3/autograd/core.hpp"

namespace py = pybind11;

inline void init_unary(py::module_& m) {
  m.def("neg", &lmp::autograd::ops::neg, py::arg("a"));
  m.def("exp", &lmp::autograd::ops::exp, py::arg("a"));
  m.def("log", &lmp::autograd::ops::log, py::arg("a"));
  m.def("sqrt", &lmp::autograd::ops::sqrt, py::arg("a"));
  m.def("abs", &lmp::autograd::ops::abs, py::arg("a"));
  m.def("sin", &lmp::autograd::ops::sin, py::arg("a"));
  m.def("cos", &lmp::autograd::ops::cos, py::arg("a"));
  m.def("tan", &lmp::autograd::ops::tan, py::arg("a"));
  m.def("clamp", &lmp::autograd::ops::clamp, py::arg("a"), py::arg("min"),
        py::arg("max"));
}
