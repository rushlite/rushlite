#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lamp3/autograd/core.hpp"
#include "lamp3/common/config.hpp"
#include "lamp3/common/macros.hpp"
#include "lamp3/tensor/core.hpp"
#include "lamp3/tensor/data_type.hpp"

namespace py = pybind11;

using lmp::autograd::Variable;  // NOLINT(google-global-names-in-headers)
using lmp::tensor::DataType;    // NOLINT(google-global-names-in-headers)
using lmp::tensor::DeviceType;  // NOLINT(google-global-names-in-headers)
using lmp::tensor::Scalar;      // NOLINT(google-global-names-in-headers)
using lmp::tensor::Tensor;      // NOLINT(google-global-names-in-headers)

namespace detail {

inline void flatten_sequence(const py::handle& node, std::vector<Scalar>& data,
                             std::vector<size_t>& shape, size_t depth) {
  if (py::isinstance<py::sequence>(node) && !py::isinstance<py::str>(node) &&
      !py::isinstance<py::bytes>(node)) {
    auto seq = py::reinterpret_borrow<py::sequence>(node);
    const size_t n = seq.size();
    if (depth == shape.size()) {
      if (!data.empty()) {
        throw py::value_error("Rushlite: array must be uniform");
      }
      shape.push_back(n);
    } else if (shape[depth] != n) {
      throw py::value_error("Rushlite: array must be uniform");
    }
    for (const auto& item : seq) {
      flatten_sequence(item, data, shape, depth + 1);
    }
  } else {
    if (depth != shape.size()) {
      throw py::value_error("Rushlite: array must be uniform");
    }
    try {
      data.push_back(node.cast<Scalar>());
    } catch (const py::cast_error&) {
      throw py::value_error("Rushlite: invalid input type");
    }
  }
}

}  // namespace detail

#define LMP_VAR_BINARY_OPERATOR(op) \
  cls.def(py::self op py::self)     \
      .def(py::self op Scalar())    \
      .def(Scalar() op py::self);

#define LMP_VAR_FUNCTION(x) .def(#x, &lmp::autograd::ops::x)

inline void init_variable_overloads(py::class_<Variable>& cls) {
  LMP_VAR_BINARY_OPERATOR(+)
  LMP_VAR_BINARY_OPERATOR(-)
  LMP_VAR_BINARY_OPERATOR(*)
  LMP_VAR_BINARY_OPERATOR(/)
  LMP_VAR_BINARY_OPERATOR(==)
  LMP_VAR_BINARY_OPERATOR(!=)
  LMP_VAR_BINARY_OPERATOR(<)
  LMP_VAR_BINARY_OPERATOR(<=)
  LMP_VAR_BINARY_OPERATOR(>)
  LMP_VAR_BINARY_OPERATOR(>=)

  cls.def(-py::self)
      .def("__abs__", &lmp::autograd::ops::abs, py::is_operator())
      .def("__matmul__", &lmp::autograd::ops::matmul, py::is_operator())
      .def("__pow__", &lmp::autograd::ops::pow, py::is_operator())
      .def("__pow__",
           static_cast<Variable (*)(const Variable&, Scalar)>(
               &lmp::autograd::binary_op<&lmp::autograd::ops::pow>),
           py::is_operator())
      .def(
          "__rpow__",
          [](const Variable& self, Scalar base) {
            return lmp::autograd::binary_op<&lmp::autograd::ops::pow>(base,
                                                                      self);
          },
          py::is_operator());

  cls LMP_FOR_EACH_CARTESIAN_PRODUCT(LMP_VAR_FUNCTION,
                                     (exp, log, sqrt, abs, sin, cos, tan))
      .def("to", &lmp::autograd::ops::to, py::arg("device"))
      .def("reshape", &lmp::autograd::ops::reshape, py::arg("shape"))
      .def("squeeze", &lmp::autograd::ops::squeeze, py::arg("axis"))
      .def("expand_dims", &lmp::autograd::ops::expand_dims, py::arg("axis"))
      .def("clamp", &lmp::autograd::ops::clamp, py::arg("min"), py::arg("max"))
      .def("sum", &lmp::autograd::ops::sum, py::arg("axis"))
      .def("max", &lmp::autograd::ops::max, py::arg("axis"))
      .def("min", &lmp::autograd::ops::min, py::arg("axis"))
      .def("prod", &lmp::autograd::ops::prod, py::arg("axis"))
      .def("matmul", &lmp::autograd::ops::matmul, py::arg("other"))
      .def("transpose", &lmp::autograd::ops::transpose)
      .def_property_readonly("T", &lmp::autograd::ops::transpose);
}

#undef LMP_VAR_BINARY_OPERATOR
#undef LMP_VAR_FUNCTION

inline void init_variable(py::module_& m) {
  auto cls =
      py::class_<Variable>(m, "Variable")
          .def(py::init<Tensor, bool>(), py::arg("data"),
               py::arg("requires_grad") = false)
          .def(py::init([](const py::sequence& data, bool requires_grad,
                           DeviceType device, DataType dtype) {
                 std::vector<Scalar> flat;
                 std::vector<size_t> shape;
                 detail::flatten_sequence(data, flat, shape, 0);
                 return Variable(Tensor(flat, shape, device, dtype),
                                 requires_grad);
               }),
               py::arg("data"), py::arg("requires_grad") = false,
               py::arg("device") = lmp::DEFAULT_DEVICE,
               py::arg("dtype") = lmp::DEFAULT_DTYPE)
          .def_property("data", &Variable::data, &Variable::set_data)
          .def_property("grad", &Variable::grad, nullptr)
          .def_property("grad_fn", &Variable::grad_fn, nullptr)
          .def_property("requires_grad", &Variable::requires_grad, nullptr)
          .def_property_readonly(
              "shape", [](const Variable& self) { return self.data().shape(); })
          .def_property_readonly(
              "device",
              [](const Variable& self) { return self.data().device(); })
          .def_property_readonly(
              "dtype", [](const Variable& self) { return self.data().type(); })
          .def_property_readonly(
              "ndim",
              [](const Variable& self) { return self.data().shape().size(); })
          .def("backward", &Variable::backward)
          .def("tolist",
               [](const Variable& self) {
                 return self.data().to_vector<lmp::tensor::Scalar>();
               })
          .def("zero_grad", &Variable::zero_grad)
          .def("__repr__", [](const Variable& self) {
            std::ostringstream oss;
            oss << self;
            return oss.str();
          });

  init_variable_overloads(cls);
}
