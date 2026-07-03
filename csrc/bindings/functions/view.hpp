#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lamp3/autograd/core.hpp"

namespace py = pybind11;

inline void init_view(py::module_& m) {
  m.def("reshape", &lmp::autograd::ops::reshape, py::arg("a"),
        py::arg("shape"));
  m.def("squeeze", &lmp::autograd::ops::squeeze, py::arg("a"), py::arg("axis"));
  m.def("expand_dims", &lmp::autograd::ops::expand_dims, py::arg("a"),
        py::arg("axis"));
  m.def("to", &lmp::autograd::ops::to, py::arg("a"), py::arg("device"));
}
