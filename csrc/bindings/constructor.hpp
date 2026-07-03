#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lamp3/autograd/core.hpp"
#include "lamp3/common/config.hpp"
#include "lamp3/common/macros.hpp"
#include "lamp3/tensor/core.hpp"
#include "lamp3/tensor/data_type.hpp"

namespace py = pybind11;

inline void init_constructor(py::module_& m) {
  m.def("zeros", &lmp::autograd::zeros, py::arg("shape"),
        py::arg("requires_grad") = false,
        py::arg("device") = lmp::DEFAULT_DEVICE,
        py::arg("dtype") = lmp::DEFAULT_DTYPE);
  m.def("ones", &lmp::autograd::ones, py::arg("shape"),
        py::arg("requires_grad") = false,
        py::arg("device") = lmp::DEFAULT_DEVICE,
        py::arg("dtype") = lmp::DEFAULT_DTYPE);
  m.def("rand", &lmp::autograd::rand, py::arg("shape"),
        py::arg("requires_grad") = false,
        py::arg("device") = lmp::DEFAULT_DEVICE,
        py::arg("dtype") = lmp::DEFAULT_DTYPE);
  m.def("randn", &lmp::autograd::randn, py::arg("mean"), py::arg("var"),
        py::arg("shape"), py::arg("requires_grad") = false,
        py::arg("device") = lmp::DEFAULT_DEVICE,
        py::arg("dtype") = lmp::DEFAULT_DTYPE);
}
