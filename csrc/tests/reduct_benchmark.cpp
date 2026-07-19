#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

#include "lamp3/tensor/core.hpp"

namespace {

using lmp::tensor::DataType;
using lmp::tensor::DeviceType;
using lmp::tensor::Tensor;
using lmp::tensor::TypeMeta;

void synchronize_cuda() {
  const cudaError_t error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA synchronization failed: ") +
                             cudaGetErrorString(error));
  }
}

template <typename T, typename BinaryOp>
std::vector<T> reduce_reference(const std::vector<T>& input,
                                const std::vector<size_t>& shape, size_t axis,
                                T identity, BinaryOp&& op) {
  const size_t reduction_size = shape.at(axis);
  const size_t output_size = input.size() / reduction_size;
  const size_t outer =
      std::accumulate(shape.begin() + static_cast<std::ptrdiff_t>(axis) + 1,
                      shape.end(), size_t{1}, std::multiplies<>());
  const size_t inner = outer * reduction_size;
  std::vector<T> output(output_size, identity);

  for (size_t i = 0; i < output_size; ++i) {
    const size_t base = ((i / outer) * inner) + (i % outer);
    for (size_t j = 0; j < reduction_size; ++j) {
      output[i] = op(output[i], input[base + (j * outer)]);
    }
  }
  return output;
}

template <typename T>
void expect_values(std::string_view name, const std::vector<T>& actual,
                   const std::vector<T>& expected) {
  if (actual.size() != expected.size()) {
    throw std::runtime_error(std::string(name) + ": output size mismatch");
  }
  for (size_t i = 0; i < actual.size(); ++i) {
    bool matches = actual[i] == expected[i];
    if constexpr (std::is_floating_point_v<T>) {
      constexpr T kTolerance =
          std::is_same_v<T, float> ? T{3.0e-4F} : T{1.0e-10};
      const T scale = std::max(T{1}, std::abs(expected[i]));
      matches = std::abs(actual[i] - expected[i]) <= kTolerance * scale;
    }
    if (!matches) {
      throw std::runtime_error(
          std::string(name) + ": mismatch at " + std::to_string(i) +
          ", got " + std::to_string(actual[i]) + ", expected " +
          std::to_string(expected[i]));
    }
  }
}

template <typename T>
std::vector<T> make_input(size_t size) {
  std::vector<T> input(size);
  for (size_t i = 0; i < size; ++i) {
    const int value = static_cast<int>(i % 17) - 8;
    if constexpr (std::is_floating_point_v<T>) {
      input[i] = static_cast<T>(value) * T{0.0625};
    } else {
      input[i] = static_cast<T>(value);
    }
  }
  return input;
}

template <typename T>
std::vector<T> make_product_input(size_t size) {
  std::vector<T> input(size);
  for (size_t i = 0; i < size; ++i) {
    if constexpr (std::is_floating_point_v<T>) {
      const int offset = static_cast<int>(i % 5) - 2;
      input[i] = T{1} + (static_cast<T>(offset) * T{0.0001});
    } else {
      input[i] = (i % 3 == 0) ? T{-1} : T{1};
    }
  }
  return input;
}

void check_sum(const std::vector<size_t>& shape, size_t axis) {
  const size_t size =
      std::accumulate(shape.begin(), shape.end(), size_t{1},
                      std::multiplies<>());
  const std::vector<float> input = make_input<float>(size);
  const Tensor tensor(input, shape, DeviceType::CUDA, DataType::Float32);
  const Tensor output = lmp::tensor::ops::sum(tensor, axis);
  const auto expected =
      reduce_reference(input, shape, axis, 0.0F, std::plus<>());
  expect_values("sum axis " + std::to_string(axis),
                output.to_vector<float>(), expected);
}

template <typename T>
void check_last_axis_reductions(size_t rows, size_t columns,
                                std::string_view dtype_name) {
  const std::vector<size_t> shape{rows, columns};
  const std::vector<T> input = make_input<T>(rows * columns);
  const Tensor tensor(input, shape, DeviceType::CUDA, TypeMeta<T>::kValue);
  const std::string suffix = " " + std::string(dtype_name) + " " +
                             std::to_string(rows) + "x" +
                             std::to_string(columns);

  expect_values(
      "sum" + suffix,
      lmp::tensor::ops::sum(tensor, 1).template to_vector<T>(),
      reduce_reference(input, shape, 1, T{0}, std::plus<>()));
  expect_values(
      "max" + suffix,
      lmp::tensor::ops::max(tensor, 1).template to_vector<T>(),
      reduce_reference(input, shape, 1, std::numeric_limits<T>::lowest(),
                       [](T a, T b) { return std::max(a, b); }));
  expect_values(
      "min" + suffix,
      lmp::tensor::ops::min(tensor, 1).template to_vector<T>(),
      reduce_reference(input, shape, 1, std::numeric_limits<T>::max(),
                       [](T a, T b) { return std::min(a, b); }));

  const std::vector<T> product_input = make_product_input<T>(rows * columns);
  const Tensor product_tensor(product_input, shape, DeviceType::CUDA,
                              TypeMeta<T>::kValue);
  expect_values(
      "prod" + suffix,
      lmp::tensor::ops::prod(product_tensor, 1).template to_vector<T>(),
      reduce_reference(product_input, shape, 1, T{1}, std::multiplies<>()));
}

void run_correctness_checks() {
  check_sum({7, 65}, 0);
  check_sum({7, 65}, 1);
  check_sum({4, 5, 37}, 0);
  check_sum({4, 5, 37}, 1);
  check_sum({4, 5, 37}, 2);

  constexpr size_t kBoundaryLengths[] = {
      31, 32, 33, 127, 128, 129, 4092, 4096, 4100,
  };
  for (const size_t length : kBoundaryLengths) {
    check_last_axis_reductions<float>(7, length, "float32");
    check_last_axis_reductions<double>(3, length, "float64");
    check_last_axis_reductions<int>(5, length, "int32");
  }
}

template <typename Reduction>
double benchmark_reduction(size_t rows, size_t columns, size_t axis,
                           size_t iterations, Reduction&& reduction) {
  const std::vector<float> input = make_input<float>(rows * columns);
  const Tensor tensor(input, {rows, columns}, DeviceType::CUDA,
                      DataType::Float32);
  for (size_t i = 0; i < 10; ++i) {
    static_cast<void>(reduction(tensor, axis));
  }
  synchronize_cuda();

  const auto start = std::chrono::steady_clock::now();
  for (size_t i = 0; i < iterations; ++i) {
    static_cast<void>(reduction(tensor, axis));
  }
  synchronize_cuda();
  const auto stop = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::micro>(stop - start).count() /
         static_cast<double>(iterations);
}

size_t parse_iterations(int argc, char** argv) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string_view(argv[i]) == "--iterations") {
      return std::stoul(argv[i + 1]);
    }
  }
  return 25;
}

bool check_only(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    if (std::string_view(argv[i]) == "--check-only") {
      return true;
    }
  }
  return false;
}

template <typename Reduction>
void print_benchmark(std::string_view op, size_t rows, size_t columns,
                     size_t axis, size_t iterations, Reduction&& reduction) {
  std::cout << op << ',' << rows << 'x' << columns << ',' << axis << ','
            << benchmark_reduction(rows, columns, axis, iterations,
                                   std::forward<Reduction>(reduction))
            << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    run_correctness_checks();
    std::cout << "reduction correctness: PASS\n";
    if (check_only(argc, argv)) {
      return EXIT_SUCCESS;
    }

    const size_t iterations = parse_iterations(argc, argv);
    struct MatrixShape {
      size_t rows;
      size_t columns;
    };
    constexpr MatrixShape kShapes[] = {
        {512, 4096}, {4096, 512}, {1024, 2048}, {512, 64}, {256, 128},
    };
    std::cout << "op,shape,axis,mean_us\n"
              << std::fixed << std::setprecision(2);
    for (const auto& [rows, columns] : kShapes) {
      for (size_t axis = 0; axis < 2; ++axis) {
        print_benchmark("sum", rows, columns, axis, iterations,
                        [](const Tensor& tensor, size_t reduction_axis) {
                          return lmp::tensor::ops::sum(tensor, reduction_axis);
                        });
        print_benchmark("min", rows, columns, axis, iterations,
                        [](const Tensor& tensor, size_t reduction_axis) {
                          return lmp::tensor::ops::min(tensor, reduction_axis);
                        });
        print_benchmark("max", rows, columns, axis, iterations,
                        [](const Tensor& tensor, size_t reduction_axis) {
                          return lmp::tensor::ops::max(tensor, reduction_axis);
                        });
        print_benchmark("prod", rows, columns, axis, iterations,
                        [](const Tensor& tensor, size_t reduction_axis) {
                          return lmp::tensor::ops::prod(tensor, reduction_axis);
                        });
      }
    }
  } catch (const std::exception& error) {
    std::cerr << "reduction benchmark failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
