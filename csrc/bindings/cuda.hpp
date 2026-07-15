#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lamp3/common/assert.hpp"

namespace py = pybind11;

inline void init_cuda(py::module_& m) {
  auto cuda = m.def_submodule("cuda");

  cuda.def("sync", []() {
#ifdef LMP_ENABLE_CUDA
    cudaDeviceSynchronize();
#endif
  });

  cuda.def("is_available", []() -> bool {
#ifdef LMP_ENABLE_CUDA
    int count = 0;
    cudaGetDeviceCount(&count);
    return count > 0;
#else
    return false;
#endif
  });
}
