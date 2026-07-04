#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lamp3/tensor/core.hpp"

namespace py = pybind11;

using lmp::tensor::DataType;    // NOLINT(google-global-names-in-headers)
using lmp::tensor::DeviceType;  // NOLINT(google-global-names-in-headers)
using lmp::tensor::Scalar;      // NOLINT(google-global-names-in-headers)
using lmp::tensor::Tensor;      // NOLINT(google-global-names-in-headers)

#define LMP_TEN_BINARY_OPERATOR(op) \
  cls.def(py::self op py::self)     \
      .def(py::self op Scalar())    \
      .def(Scalar() op py::self);

inline void init_tensor_overloads(py::class_<Tensor>& cls) {
  LMP_TEN_BINARY_OPERATOR(+)
  LMP_TEN_BINARY_OPERATOR(-)
  LMP_TEN_BINARY_OPERATOR(*)
  LMP_TEN_BINARY_OPERATOR(/)
  LMP_TEN_BINARY_OPERATOR(==)
  LMP_TEN_BINARY_OPERATOR(!=)
  LMP_TEN_BINARY_OPERATOR(<)
  LMP_TEN_BINARY_OPERATOR(<=)
  LMP_TEN_BINARY_OPERATOR(>)
  LMP_TEN_BINARY_OPERATOR(>=)

  cls.def(-py::self);
}

#undef LMP_TEN_BINARY_OPERATOR

inline void init_tensor(py::module_& m) {
  auto cls =
      py::class_<Tensor>(m, "_Tensor")
          .def(py::init<const std::vector<double>, const std::vector<size_t>,
                        DeviceType, DataType>(),
               py::arg("data"), py::arg("shape"), py::arg("device"),
               py::arg("dtype"))
          .def_property("shape", &Tensor::shape, nullptr)
          .def_property("device", &Tensor::device, nullptr)
          .def_property("dtype", &Tensor::type, nullptr)
          .def("index", &Tensor::index, py::arg("idx"))
          .def("reshape", &Tensor::reshape, py::arg("new_shape"))
          .def("squeeze", &Tensor::squeeze, py::arg("dim"))
          .def("expand_dims", &Tensor::expand_dims, py::arg("dim"))
          .def("to", &Tensor::to, py::arg("device"))
          .def("copy", &Tensor::copy, py::arg("other"))
          .def("fill", &Tensor::fill, py::arg("item"))
          .def("tolist",
               [](Tensor& t) -> std::vector<double> {
                 return t.to_vector<double>();
               })
          .def("__repr__", [](const Tensor& self) {
            std::ostringstream oss;
            oss << self;
            return oss.str();
          });

  init_tensor_overloads(cls);
}
