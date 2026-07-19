#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "lamp3/autograd/core.hpp"
#include "lamp3/tensor/core.hpp"

namespace {

using lmp::autograd::Variable;
using lmp::tensor::DataType;
using lmp::tensor::DeviceType;
using lmp::tensor::Tensor;
using lmp::tensor::TensorImpl;
using stride_t = lmp::tensor::detail::stride_t;

constexpr double kTolerance = 2e-4;

size_t numel(const std::vector<size_t>& shape) {
  if (shape.empty()) return 0;
  return std::accumulate(shape.begin(), shape.end(), size_t{1},
                         std::multiplies<>());
}

std::vector<size_t> unravel(size_t linear,
                            const std::vector<size_t>& shape) {
  std::vector<size_t> coordinates(shape.size());
  for (size_t i = shape.size(); i > 0; --i) {
    const size_t dim = i - 1;
    coordinates[dim] = linear % shape[dim];
    linear /= shape[dim];
  }
  return coordinates;
}

std::vector<stride_t> canonical_strides(const std::vector<size_t>& shape) {
  std::vector<stride_t> strides(shape.size());
  stride_t stride = 1;
  for (size_t i = shape.size(); i > 0; --i) {
    const size_t dim = i - 1;
    strides[dim] = stride;
    stride *= static_cast<stride_t>(shape[dim]);
  }
  return strides;
}

std::vector<double> logical_values(const Tensor& tensor) {
  std::vector<double> values;
  values.reserve(tensor.numel());
  for (size_t i = 0; i < tensor.numel(); ++i) {
    values.push_back(tensor.index(unravel(i, tensor.shape())));
  }
  return values;
}

void expect_values(const Tensor& actual, const std::vector<double>& expected,
                   double tolerance = kTolerance) {
  EXPECT_THAT(actual.to_vector<double>(),
              testing::Pointwise(testing::DoubleNear(tolerance), expected));
}

void expect_canonical(const Tensor& tensor) {
  EXPECT_TRUE(tensor.is_contiguous());
  EXPECT_THAT(tensor.strides(),
              testing::ElementsAreArray(canonical_strides(tensor.shape())));
}

Tensor make_f32(DeviceType device, const std::vector<size_t>& shape,
                float first = 1.0F, float step = 1.0F) {
  std::vector<float> data(numel(shape));
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = first + (step * static_cast<float>(i));
  }
  return Tensor(data, shape, device, DataType::Float32);
}

std::vector<size_t> broadcast_shape(const std::vector<size_t>& a,
                                    const std::vector<size_t>& b) {
  const size_t rank = std::max(a.size(), b.size());
  std::vector<size_t> out(rank);
  for (size_t dim = 0; dim < rank; ++dim) {
    const size_t a_missing = rank - a.size();
    const size_t b_missing = rank - b.size();
    const size_t a_size = dim < a_missing ? 1 : a[dim - a_missing];
    const size_t b_size = dim < b_missing ? 1 : b[dim - b_missing];
    EXPECT_TRUE(a_size == 1 || b_size == 1 || a_size == b_size);
    out[dim] = a_size == 1 ? b_size : a_size;
  }
  return out;
}

std::vector<size_t> broadcast_coordinates(
    const std::vector<size_t>& output_coordinates,
    const std::vector<size_t>& input_shape) {
  std::vector<size_t> input_coordinates(input_shape.size());
  const size_t missing = output_coordinates.size() - input_shape.size();
  for (size_t dim = 0; dim < input_shape.size(); ++dim) {
    input_coordinates[dim] =
        input_shape[dim] == 1 ? 0 : output_coordinates[missing + dim];
  }
  return input_coordinates;
}

std::vector<double> binary_reference(
    const Tensor& a, const Tensor& b,
    const std::function<double(double, double)>& fn) {
  const std::vector<size_t> shape = broadcast_shape(a.shape(), b.shape());
  std::vector<double> expected;
  expected.reserve(numel(shape));
  for (size_t i = 0; i < numel(shape); ++i) {
    const std::vector<size_t> coordinates = unravel(i, shape);
    expected.push_back(
        fn(a.index(broadcast_coordinates(coordinates, a.shape())),
           b.index(broadcast_coordinates(coordinates, b.shape()))));
  }
  return expected;
}

std::vector<size_t> matmul_shape(const Tensor& a, const Tensor& b) {
  std::vector<size_t> a_batch(a.shape().begin(), a.shape().end() - 2);
  std::vector<size_t> b_batch(b.shape().begin(), b.shape().end() - 2);
  std::vector<size_t> shape = broadcast_shape(a_batch, b_batch);
  shape.push_back(a.shape()[a.shape().size() - 2]);
  shape.push_back(b.shape().back());
  return shape;
}

std::vector<double> matmul_reference(const Tensor& a, const Tensor& b) {
  const std::vector<size_t> output_shape = matmul_shape(a, b);
  const size_t batch_rank = output_shape.size() - 2;
  const size_t m = output_shape[batch_rank];
  const size_t n = output_shape[batch_rank + 1];
  const size_t k = a.shape().back();
  std::vector<size_t> batch_shape(output_shape.begin(),
                                  output_shape.begin() + batch_rank);
  const size_t batch_count = batch_shape.empty() ? 1 : numel(batch_shape);

  std::vector<size_t> a_batch(a.shape().begin(), a.shape().end() - 2);
  std::vector<size_t> b_batch(b.shape().begin(), b.shape().end() - 2);
  std::vector<double> expected(batch_count * m * n);
  for (size_t batch = 0; batch < batch_count; ++batch) {
    const std::vector<size_t> batch_coordinates =
        batch_shape.empty() ? std::vector<size_t>{}
                            : unravel(batch, batch_shape);
    const std::vector<size_t> a_batch_coordinates =
        broadcast_coordinates(batch_coordinates, a_batch);
    const std::vector<size_t> b_batch_coordinates =
        broadcast_coordinates(batch_coordinates, b_batch);
    for (size_t row = 0; row < m; ++row) {
      for (size_t col = 0; col < n; ++col) {
        double sum = 0;
        for (size_t inner = 0; inner < k; ++inner) {
          std::vector<size_t> a_coordinates = a_batch_coordinates;
          a_coordinates.push_back(row);
          a_coordinates.push_back(inner);
          std::vector<size_t> b_coordinates = b_batch_coordinates;
          b_coordinates.push_back(inner);
          b_coordinates.push_back(col);
          sum += a.index(a_coordinates) * b.index(b_coordinates);
        }
        expected[(batch * m * n) + (row * n) + col] = sum;
      }
    }
  }
  return expected;
}

class NoncontiguousOpsTest
    : public testing::TestWithParam<DeviceType> {
 protected:
  DeviceType device() const { return GetParam(); }
};

TEST_P(NoncontiguousOpsTest, EveryUnaryFunctorReadsTranspose) {
  Tensor input =
      Tensor(std::vector<float>{0.5F, 1.0F, 1.5F, 2.0F, 2.5F, 3.0F},
             {2, 3}, device(), DataType::Float32)
          .transpose(0, 1);
  ASSERT_FALSE(input.is_contiguous());
  const void* storage = input.data();
  const std::vector<size_t> shape = input.shape();
  const std::vector<stride_t> strides = input.strides();
  const std::vector<double> logical = logical_values(input);

  struct UnaryCase {
    std::string name;
    std::function<Tensor(const Tensor&)> operation;
    std::function<double(double)> reference;
  };
  const std::vector<UnaryCase> cases{
      {"neg", [](const Tensor& x) { return lmp::tensor::ops::neg(x); },
       [](double x) { return -x; }},
      {"exp", [](const Tensor& x) { return lmp::tensor::ops::exp(x); },
       [](double x) { return std::exp(x); }},
      {"log", [](const Tensor& x) { return lmp::tensor::ops::log(x); },
       [](double x) { return std::log(x); }},
      {"sqrt", [](const Tensor& x) { return lmp::tensor::ops::sqrt(x); },
       [](double x) { return std::sqrt(x); }},
      {"abs", [](const Tensor& x) { return lmp::tensor::ops::abs(x); },
       [](double x) { return std::abs(x); }},
      {"sin", [](const Tensor& x) { return lmp::tensor::ops::sin(x); },
       [](double x) { return std::sin(x); }},
      {"cos", [](const Tensor& x) { return lmp::tensor::ops::cos(x); },
       [](double x) { return std::cos(x); }},
      {"tan", [](const Tensor& x) { return lmp::tensor::ops::tan(x); },
       [](double x) { return std::tan(x); }},
      {"clamp",
       [](const Tensor& x) { return lmp::tensor::ops::clamp(x, 0.9, 2.2); },
       [](double x) { return std::clamp(x, 0.9, 2.2); }},
  };

  for (const UnaryCase& test : cases) {
    SCOPED_TRACE(test.name);
    Tensor output = test.operation(input);
    std::vector<double> expected(logical.size());
    std::transform(logical.begin(), logical.end(), expected.begin(),
                   test.reference);
    expect_values(output, expected, 1e-3);
    EXPECT_THAT(output.shape(), testing::ElementsAreArray(shape));
    expect_canonical(output);
  }

  EXPECT_EQ(input.data(), storage);
  EXPECT_THAT(input.shape(), testing::ElementsAreArray(shape));
  EXPECT_THAT(input.strides(), testing::ElementsAreArray(strides));
  EXPECT_THAT(logical_values(input), testing::ElementsAreArray(logical));
}

TEST_P(NoncontiguousOpsTest, UnaryReadsHigherRankPermutation) {
  Tensor input = make_f32(device(), {2, 3, 4}, -6.0F, 0.5F)
                     .permute({2, 0, 1});
  Tensor output = lmp::tensor::ops::clamp(input, -2.0, 3.0);
  std::vector<double> expected = logical_values(input);
  for (double& value : expected) value = std::clamp(value, -2.0, 3.0);
  expect_values(output, expected);
  expect_canonical(output);
}

TEST_P(NoncontiguousOpsTest, EveryBinaryFunctorReadsMatchingTransposes) {
  Tensor a = make_f32(device(), {2, 3}, 1.0F, 0.25F).transpose(0, 1);
  Tensor b = make_f32(device(), {2, 3}, 0.5F, 0.2F).transpose(0, 1);

  struct BinaryCase {
    std::string name;
    std::function<Tensor(const Tensor&, const Tensor&)> operation;
    std::function<double(double, double)> reference;
    double tolerance{kTolerance};
  };
  const std::vector<BinaryCase> cases{
      {"add", [](const Tensor& x, const Tensor& y) { return x + y; },
       std::plus<>()},
      {"sub", [](const Tensor& x, const Tensor& y) { return x - y; },
       std::minus<>()},
      {"mul", [](const Tensor& x, const Tensor& y) { return x * y; },
       std::multiplies<>()},
      {"div", [](const Tensor& x, const Tensor& y) { return x / y; },
       std::divides<>()},
      {"pow",
       [](const Tensor& x, const Tensor& y) {
         return lmp::tensor::ops::pow(x, y);
       },
       [](double x, double y) { return std::pow(x, y); }, 1e-3},
      {"eq", [](const Tensor& x, const Tensor& y) { return x == y; },
       [](double x, double y) { return x == y; }},
      {"ne", [](const Tensor& x, const Tensor& y) { return x != y; },
       [](double x, double y) { return x != y; }},
      {"le", [](const Tensor& x, const Tensor& y) { return x <= y; },
       [](double x, double y) { return x <= y; }},
      {"lt", [](const Tensor& x, const Tensor& y) { return x < y; },
       [](double x, double y) { return x < y; }},
      {"ge", [](const Tensor& x, const Tensor& y) { return x >= y; },
       [](double x, double y) { return x >= y; }},
      {"gt", [](const Tensor& x, const Tensor& y) { return x > y; },
       [](double x, double y) { return x > y; }},
  };

  for (const BinaryCase& test : cases) {
    SCOPED_TRACE(test.name);
    Tensor output = test.operation(a, b);
    expect_values(output, binary_reference(a, b, test.reference),
                  test.tolerance);
    expect_canonical(output);
  }
}

TEST_P(NoncontiguousOpsTest, BinaryHandlesMixedLayoutsAndPermutations) {
  Tensor a = make_f32(device(), {2, 3, 4}, 1.0F, 0.2F)
                 .permute({1, 2, 0});
  Tensor b = make_f32(device(), {4, 2, 3}, -2.0F, 0.3F)
                 .permute({2, 0, 1});
  ASSERT_EQ(a.shape(), b.shape());
  ASSERT_NE(a.strides(), b.strides());

  Tensor permuted = a + b;
  expect_values(permuted, binary_reference(a, b, std::plus<>()));
  expect_canonical(permuted);

  Tensor contiguous(make_f32(device(), a.shape(), 3.0F, -0.1F));
  Tensor mixed = a * contiguous;
  expect_values(mixed, binary_reference(a, contiguous, std::multiplies<>()));
  expect_canonical(mixed);
}

TEST_P(NoncontiguousOpsTest, BinaryBroadcastsNoncontiguousOperands) {
  Tensor a = make_f32(device(), {2, 3, 4}, 1.0F, 0.1F)
                 .permute({0, 2, 1});
  Tensor b = make_f32(device(), {1, 3, 4}, -1.0F, 0.2F)
                 .transpose(1, 2);
  ASSERT_THAT(a.shape(), testing::ElementsAre(2, 4, 3));
  ASSERT_THAT(b.shape(), testing::ElementsAre(1, 4, 3));
  ASSERT_FALSE(a.is_contiguous());
  ASSERT_FALSE(b.is_contiguous());

  Tensor output = a + b;
  EXPECT_THAT(output.shape(), testing::ElementsAre(2, 4, 3));
  expect_values(output, binary_reference(a, b, std::plus<>()));
  expect_canonical(output);

  Tensor c = make_f32(device(), {2, 1, 2, 3}, 0.5F, 0.1F)
                 .transpose(2, 3);
  Tensor d = make_f32(device(), {1, 4, 1, 2}, 2.0F, 0.25F);
  Tensor multiple = c - d;
  EXPECT_THAT(multiple.shape(), testing::ElementsAre(2, 4, 3, 2));
  expect_values(multiple, binary_reference(c, d, std::minus<>()));
  expect_canonical(multiple);
}

TEST_P(NoncontiguousOpsTest, BinarySupportsMixedDtypesAndBackwardHelpers) {
  Tensor input =
      Tensor(std::vector<float>{-3, -2, -1, 1, 2, 3}, {2, 3}, device(),
             DataType::Float32)
          .transpose(0, 1);
  Tensor integers =
      Tensor(std::vector<int32_t>{1, 2, 3, 4, 5, 6}, {3, 2}, device(),
             DataType::Int32);
  Tensor mixed = input + integers;
  EXPECT_EQ(mixed.type(), DataType::Float32);
  expect_values(mixed, binary_reference(input, integers, std::plus<>()));

  Tensor grad = make_f32(device(), {3, 2}, 1.0F, 0.0F);
  Tensor abs_grad = lmp::tensor::ops::abs_backward(input, grad);
  expect_values(abs_grad, {-1, 1, -1, 1, -1, 1});
  Tensor clamp_grad =
      lmp::tensor::ops::clamp_backward(input, grad, -2.0, 2.0);
  expect_values(clamp_grad, {0, 1, 0, 0, 1, 0});
}

enum class ReductionKind { Sum, Min, Max, Prod };

Tensor reduce(const Tensor& input, size_t axis, ReductionKind kind) {
  switch (kind) {
    case ReductionKind::Sum:
      return lmp::tensor::ops::sum(input, axis);
    case ReductionKind::Min:
      return lmp::tensor::ops::min(input, axis);
    case ReductionKind::Max:
      return lmp::tensor::ops::max(input, axis);
    case ReductionKind::Prod:
      return lmp::tensor::ops::prod(input, axis);
  }
  return {};
}

std::vector<double> reduction_reference(const Tensor& input, size_t axis,
                                        ReductionKind kind) {
  std::vector<size_t> output_shape = input.shape();
  output_shape[axis] = 1;
  std::vector<double> expected(numel(output_shape));
  for (size_t i = 0; i < expected.size(); ++i) {
    std::vector<size_t> coordinates = unravel(i, output_shape);
    double value = 0;
    if (kind == ReductionKind::Min) {
      value = std::numeric_limits<double>::max();
    } else if (kind == ReductionKind::Max) {
      value = std::numeric_limits<double>::lowest();
    } else if (kind == ReductionKind::Prod) {
      value = 1;
    }
    for (size_t j = 0; j < input.shape()[axis]; ++j) {
      coordinates[axis] = j;
      const double item = input.index(coordinates);
      switch (kind) {
        case ReductionKind::Sum:
          value += item;
          break;
        case ReductionKind::Min:
          value = std::min(value, item);
          break;
        case ReductionKind::Max:
          value = std::max(value, item);
          break;
        case ReductionKind::Prod:
          value *= item;
          break;
      }
    }
    expected[i] = value;
  }
  return expected;
}

TEST_P(NoncontiguousOpsTest, ReductionsReadEveryTransposeAxis) {
  Tensor input = make_f32(device(), {2, 3}, 1.0F, 0.5F).transpose(0, 1);
  for (ReductionKind kind :
       {ReductionKind::Sum, ReductionKind::Min, ReductionKind::Max,
        ReductionKind::Prod}) {
    for (size_t axis = 0; axis < input.shape().size(); ++axis) {
      SCOPED_TRACE(axis);
      Tensor output = reduce(input, axis, kind);
      expect_values(output, reduction_reference(input, axis, kind));
      std::vector<size_t> expected_shape = input.shape();
      expected_shape[axis] = 1;
      EXPECT_THAT(output.shape(), testing::ElementsAreArray(expected_shape));
      expect_canonical(output);
    }
  }
}

TEST_P(NoncontiguousOpsTest, ReductionsReadDiscriminatingPermutation) {
  Tensor input = make_f32(device(), {2, 3, 4}, 1.0F, 0.1F)
                     .permute({1, 2, 0});
  ASSERT_THAT(input.shape(), testing::ElementsAre(3, 4, 2));
  ASSERT_THAT(input.strides(), testing::ElementsAre(4, 1, 12));
  for (ReductionKind kind :
       {ReductionKind::Sum, ReductionKind::Min, ReductionKind::Max,
        ReductionKind::Prod}) {
    for (size_t axis = 0; axis < input.shape().size(); ++axis) {
      Tensor output = reduce(input, axis, kind);
      expect_values(output, reduction_reference(input, axis, kind), 1e-3);
      expect_canonical(output);
    }
  }

  Tensor size_one =
      make_f32(device(), {2, 1, 3}, 1.0F, 0.5F).permute({2, 1, 0});
  Tensor output = lmp::tensor::ops::sum(size_one, 1);
  expect_values(output,
                reduction_reference(size_one, 1, ReductionKind::Sum));
}

TEST_P(NoncontiguousOpsTest, MatmulReadsRankTwoTransposes) {
  Tensor a_contiguous = make_f32(device(), {2, 3}, 1.0F, 0.5F);
  Tensor b_contiguous = make_f32(device(), {3, 2}, -1.0F, 0.25F);
  Tensor a = make_f32(device(), {3, 2}, 1.0F, 0.5F).transpose(0, 1);
  Tensor b = make_f32(device(), {2, 3}, -1.0F, 0.25F).transpose(0, 1);

  for (const auto& operands :
       std::vector<std::pair<Tensor, Tensor>>{{a, b_contiguous},
                                               {a_contiguous, b},
                                               {a, b}}) {
    Tensor output =
        lmp::tensor::ops::matmul(operands.first, operands.second);
    EXPECT_THAT(output.shape(), testing::ElementsAre(2, 2));
    expect_values(output,
                  matmul_reference(operands.first, operands.second));
    expect_canonical(output);
  }
}

TEST_P(NoncontiguousOpsTest, MatmulBroadcastsBatchesWithoutMaterializing) {
  Tensor a = make_f32(device(), {2, 2, 1, 3}, 1.0F, 0.1F)
                 .permute({1, 2, 0, 3});
  Tensor b = make_f32(device(), {3, 1, 3, 2}, -0.5F, 0.05F)
                 .permute({1, 2, 0, 3});
  ASSERT_THAT(a.shape(), testing::ElementsAre(2, 1, 2, 3));
  ASSERT_THAT(b.shape(), testing::ElementsAre(1, 3, 3, 2));
  ASSERT_FALSE(a.is_contiguous());
  ASSERT_FALSE(b.is_contiguous());

  Tensor output = lmp::tensor::ops::matmul(a, b);
  EXPECT_THAT(output.shape(), testing::ElementsAre(2, 3, 2, 2));
  expect_values(output, matmul_reference(a, b), 1e-3);
  expect_canonical(output);

  Tensor shared_right = make_f32(device(), {3, 2}, 0.5F, 0.2F);
  Tensor with_shared_right = lmp::tensor::ops::matmul(a, shared_right);
  EXPECT_THAT(with_shared_right.shape(), testing::ElementsAre(2, 1, 2, 2));
  expect_values(with_shared_right, matmul_reference(a, shared_right), 1e-3);

  Tensor shared_left = make_f32(device(), {2, 3}, 0.25F, 0.15F);
  Tensor with_shared_left = lmp::tensor::ops::matmul(shared_left, b);
  EXPECT_THAT(with_shared_left.shape(), testing::ElementsAre(1, 3, 2, 2));
  expect_values(with_shared_left, matmul_reference(shared_left, b), 1e-3);
}

TEST_P(NoncontiguousOpsTest, MatmulHandlesPermutedBatchDimensions) {
  Tensor a = make_f32(device(), {2, 3, 2, 3}, 1.0F, 0.05F)
                 .permute({1, 0, 2, 3});
  Tensor b = make_f32(device(), {3, 2, 3, 2}, -1.0F, 0.04F);
  ASSERT_THAT(a.shape(), testing::ElementsAre(3, 2, 2, 3));
  ASSERT_FALSE(a.is_contiguous());
  Tensor output = lmp::tensor::ops::matmul(a, b);
  EXPECT_THAT(output.shape(), testing::ElementsAre(3, 2, 2, 2));
  expect_values(output, matmul_reference(a, b), 1e-3);
  expect_canonical(output);
}

TEST_P(NoncontiguousOpsTest, MatmulSupportsMixedDtypesAndValidatesShapes) {
  Tensor a = make_f32(device(), {3, 2}, 1.0F, 0.5F).transpose(0, 1);
  Tensor b =
      Tensor(std::vector<int32_t>{1, 2, 3, 4, 5, 6}, {3, 2}, device(),
             DataType::Int32);
  Tensor output = lmp::tensor::ops::matmul(a, b);
  EXPECT_EQ(output.type(), DataType::Float32);
  expect_values(output, matmul_reference(a, b));

  EXPECT_THROW(
      lmp::tensor::ops::matmul(make_f32(device(), {3}),
                               make_f32(device(), {3, 2})),
      std::runtime_error);
  EXPECT_THROW(
      lmp::tensor::ops::matmul(make_f32(device(), {2, 3}),
                               make_f32(device(), {4, 2})),
      std::runtime_error);
  EXPECT_THROW(
      lmp::tensor::ops::matmul(make_f32(device(), {2, 2, 3}),
                               make_f32(device(), {3, 3, 2})),
      std::runtime_error);
}

TEST_P(NoncontiguousOpsTest, UnaryAndReductionAutogradUseLogicalOrder) {
  Tensor data =
      Tensor(std::vector<float>{-1, 2, -3, 4, -5, 6}, {2, 3}, device(),
             DataType::Float32)
          .transpose(0, 1);
  Variable input(data, true);
  Variable absolute = lmp::autograd::ops::abs(input);
  absolute.backward();
  expect_values(input.grad(), {-1, 1, 1, -1, -1, 1});

  Variable reduced_input(data, true);
  Variable reduced = lmp::autograd::ops::sum(reduced_input, 1);
  reduced.backward();
  expect_values(reduced_input.grad(), {1, 1, 1, 1, 1, 1});
}

TEST_P(NoncontiguousOpsTest, BatchedMatmulAutogradReducesBroadcastAxes) {
  Tensor a_data(std::vector<float>(6, 1.0F), {1, 1, 2, 3}, device(),
                DataType::Float32);
  Tensor b_data(std::vector<float>(24, 1.0F), {2, 4, 3, 1}, device(),
                DataType::Float32);
  Variable a(a_data, true);
  Variable b(b_data, true);
  Variable output = lmp::autograd::ops::matmul(a, b);
  EXPECT_THAT(output.data().shape(), testing::ElementsAre(2, 4, 2, 1));
  expect_values(output.data(), std::vector<double>(16, 3.0));

  output.backward();
  expect_values(a.grad(), std::vector<double>(6, 8.0));
  expect_values(b.grad(), std::vector<double>(24, 2.0));
  output.backward();
  expect_values(a.grad(), std::vector<double>(6, 16.0));
  expect_values(b.grad(), std::vector<double>(24, 4.0));
}

TEST_P(NoncontiguousOpsTest, MatmulAutogradConsumesTransposedOperands) {
  Tensor a_data(std::vector<float>(6, 1.0F), {3, 2}, device(),
                DataType::Float32);
  Tensor b_data(std::vector<float>(6, 1.0F), {2, 3}, device(),
                DataType::Float32);
  Variable a(a_data.transpose(0, 1), true);
  Variable b(b_data.transpose(0, 1), true);
  Variable output = lmp::autograd::ops::matmul(a, b);
  output.backward();
  expect_values(a.grad(), std::vector<double>(6, 2.0));
  expect_values(b.grad(), std::vector<double>(6, 2.0));
}

TEST_P(NoncontiguousOpsTest, ZeroElementRoutesDoNotLaunch) {
  Tensor empty(std::vector<float>{}, {2, 0, 3}, device(),
               DataType::Float32);
  Tensor unary = lmp::tensor::ops::neg(empty);
  EXPECT_EQ(unary.numel(), 0);
  EXPECT_THAT(unary.shape(), testing::ElementsAre(2, 0, 3));
  expect_canonical(unary);

  Tensor binary = empty + empty;
  EXPECT_EQ(binary.numel(), 0);
  expect_canonical(binary);

  Tensor sum = lmp::tensor::ops::sum(empty, 1);
  EXPECT_THAT(sum.shape(), testing::ElementsAre(2, 1, 3));
  expect_values(sum, std::vector<double>(6, 0.0));
  Tensor product = lmp::tensor::ops::prod(empty, 1);
  expect_values(product, std::vector<double>(6, 1.0));

  Tensor zero_batch_a(std::vector<float>{}, {0, 2, 3}, device(),
                      DataType::Float32);
  Tensor zero_batch_b =
      make_f32(device(), {1, 3, 4}, 1.0F, 0.0F);
  Tensor zero_batch =
      lmp::tensor::ops::matmul(zero_batch_a, zero_batch_b);
  EXPECT_THAT(zero_batch.shape(), testing::ElementsAre(0, 2, 4));
  EXPECT_EQ(zero_batch.numel(), 0);
  expect_canonical(zero_batch);

  Tensor zero_k_a(std::vector<float>{}, {2, 0}, device(),
                  DataType::Float32);
  Tensor zero_k_b(std::vector<float>{}, {0, 4}, device(),
                  DataType::Float32);
  Tensor zero_k = lmp::tensor::ops::matmul(zero_k_a, zero_k_b);
  EXPECT_THAT(zero_k.shape(), testing::ElementsAre(2, 4));
  expect_values(zero_k, std::vector<double>(8, 0.0));
}

class EagerLazyBackend final : public lmp::tensor::lazy::LazyBackend {
 public:
  void realize(TensorImpl* impl) override {
    impl->lazy_op()->run_eager(*impl);
  }
};

class ScopedCpuLazyBackend {
 public:
  explicit ScopedCpuLazyBackend(lmp::tensor::lazy::LazyBackend* backend)
      : previous_(lmp::tensor::lazy::backend(DeviceType::CPU)) {
    lmp::tensor::lazy::register_backend(DeviceType::CPU, backend);
  }

  ~ScopedCpuLazyBackend() {
    lmp::tensor::lazy::register_backend(DeviceType::CPU, previous_);
  }

 private:
  lmp::tensor::lazy::LazyBackend* previous_;
};

TEST(NoncontiguousFusionTest, StridedAndBroadcastConsumersExecuteEagerly) {
  EagerLazyBackend backend;
  ScopedCpuLazyBackend registration(&backend);
  lmp::tensor::lazy::CaptureGuard capture(true);

  Tensor contiguous = make_f32(DeviceType::CPU, {2, 3});
  Tensor deferred = lmp::tensor::ops::neg(contiguous);
  EXPECT_TRUE(lmp::tensor::detail::UnsafeTensorAccessor::getImpl(deferred)
                  ->is_deferred());

  Tensor strided = contiguous.transpose(0, 1);
  Tensor unary = lmp::tensor::ops::neg(strided);
  EXPECT_FALSE(lmp::tensor::detail::UnsafeTensorAccessor::getImpl(unary)
                   ->is_deferred());
  expect_canonical(unary);

  Tensor row = make_f32(DeviceType::CPU, {2});
  Tensor broadcast = strided + row;
  EXPECT_FALSE(lmp::tensor::detail::UnsafeTensorAccessor::getImpl(broadcast)
                   ->is_deferred());
  expect_canonical(broadcast);
}

#ifdef LMP_ENABLE_CUDA
TEST(NoncontiguousCUDAOpsTest, OptimizedMatmulHandlesBothBColumnStrides) {
  constexpr size_t kBatch = 2;
  constexpr size_t kM = 128;
  constexpr size_t kN = 128;
  constexpr size_t kK = 17;
  Tensor a(std::vector<float>(kBatch * kM * kK, 1.0F),
           {kBatch, kM, kK}, DeviceType::CUDA, DataType::Float32);
  Tensor b_contiguous(std::vector<float>(kBatch * kK * kN, 1.0F),
                      {kBatch, kK, kN}, DeviceType::CUDA,
                      DataType::Float32);
  Tensor b_transposed(
      std::vector<float>(kBatch * kN * kK, 1.0F), {kBatch, kN, kK},
      DeviceType::CUDA, DataType::Float32);
  b_transposed = b_transposed.transpose(1, 2);
  ASSERT_EQ(b_transposed.strides().back(), static_cast<stride_t>(kK));

  for (const Tensor& b : {b_contiguous, b_transposed}) {
    Tensor output = lmp::tensor::ops::matmul(a, b);
    expect_values(output,
                  std::vector<double>(kBatch * kM * kN, kK), 1e-3);
    expect_canonical(output);
  }
}

TEST(NoncontiguousCUDAOpsTest, MatmulGridStrideCoversMoreThanGridZ) {
  constexpr size_t kBatch = 65537;
  Tensor a(std::vector<float>(kBatch, 2.0F), {kBatch, 1, 1},
           DeviceType::CUDA, DataType::Float32);
  Tensor b(std::vector<float>(kBatch, 3.0F), {kBatch, 1, 1},
           DeviceType::CUDA, DataType::Float32);
  Tensor output = lmp::tensor::ops::matmul(a, b);
  expect_values(output, std::vector<double>(kBatch, 6.0), 1e-5);
}
#endif

std::vector<DeviceType> devices() {
  std::vector<DeviceType> values{DeviceType::CPU};
#ifdef LMP_ENABLE_CUDA
  values.push_back(DeviceType::CUDA);
#endif
  return values;
}

INSTANTIATE_TEST_SUITE_P(CPUAndCUDA, NoncontiguousOpsTest,
                         testing::ValuesIn(devices()));

}  // namespace
