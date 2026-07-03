#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lamp3/autograd/core.hpp"

namespace py = pybind11;

#define LMP_BIND_BINARY_OP(name)                                       \
  m.def(#name, &lmp::autograd::ops::name, py::arg("a"), py::arg("b")); \
  m.def(#name,                                                         \
        static_cast<lmp::autograd::Variable (*)(                       \
            const lmp::autograd::Variable&, lmp::tensor::Scalar)>(     \
            &lmp::autograd::binary_op<&lmp::autograd::ops::name>),     \
        py::arg("a"), py::arg("b"));                                   \
  m.def(#name,                                                         \
        static_cast<lmp::autograd::Variable (*)(                       \
            lmp::tensor::Scalar, const lmp::autograd::Variable&)>(     \
            &lmp::autograd::binary_op<&lmp::autograd::ops::name>),     \
        py::arg("a"), py::arg("b"));

inline void init_binary(py::module_& m) {
  LMP_BIND_BINARY_OP(add)
  LMP_BIND_BINARY_OP(sub)
  LMP_BIND_BINARY_OP(mul)
  LMP_BIND_BINARY_OP(div)
  LMP_BIND_BINARY_OP(pow)
  LMP_BIND_BINARY_OP(eq)
  LMP_BIND_BINARY_OP(ne)
  LMP_BIND_BINARY_OP(ge)
  LMP_BIND_BINARY_OP(le)
  LMP_BIND_BINARY_OP(gt)
  LMP_BIND_BINARY_OP(lt)
}

#undef LMP_BIND_BINARY_OP
