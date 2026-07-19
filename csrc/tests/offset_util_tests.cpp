#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <functional>
#include <numeric>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "lamp3/tensor/core.hpp"
#include "lamp3/tensor/cpu/meta_handler.hpp"
#include "lamp3/tensor/cpu/offset_util.hpp"
#include "lamp3/tensor/infer_meta.hpp"
#include "offset_util_test_cases.hpp"

namespace {

using lmp::tensor::DataType;
using lmp::tensor::DeviceType;
using lmp::tensor::Tensor;
using lmp::tensor::detail::BinaryMetaHandler;
using lmp::tensor::detail::OffsetCalculator;
using lmp::tensor::detail::OperandLayout;
using lmp::tensor::detail::ReductMetaHandler;
using lmp::tensor::detail::UnaryMetaHandler;
using lmp::tensor::detail::make_offset_calculator;
using lmp::tensor::detail::operand_layout;
using lmp::tensor::detail::stride_t;

template <size_t NArgs>
void expect_case(const lmp::tensor::test::OffsetCase<NArgs>& test_case) {
  SCOPED_TRACE(test_case.name);
  ASSERT_EQ(test_case.logical_indices.size(),
            test_case.expected_offsets.size());

  const OffsetCalculator<NArgs> calculator =
      make_offset_calculator(test_case.iteration_shape, test_case.operands);
  for (size_t i = 0; i < test_case.logical_indices.size(); ++i) {
    const auto actual = calculator.get(test_case.logical_indices[i]);
    for (size_t operand = 0; operand < NArgs; ++operand) {
      EXPECT_EQ(actual[operand], test_case.expected_offsets[i][operand])
          << "logical index " << test_case.logical_indices[i] << ", operand "
          << operand;
    }
  }
}

Tensor make_tensor(const std::vector<size_t>& shape) {
  const size_t numel =
      shape.empty()
          ? 0
          : std::accumulate(shape.begin(), shape.end(), size_t{1},
                            std::multiplies<>());
  std::vector<float> values(numel);
  std::iota(values.begin(), values.end(), 0.0F);
  return Tensor(values, shape, DeviceType::CPU, DataType::Float32);
}

TEST(OffsetUtilTest, UnaryFixtureTable) {
  for (const auto& test_case : lmp::tensor::test::unary_offset_cases()) {
    expect_case(test_case);
  }
}

TEST(OffsetUtilTest, BinaryFixtureTable) {
  for (const auto& test_case : lmp::tensor::test::binary_offset_cases()) {
    expect_case(test_case);
  }
}

TEST(OffsetUtilTest, RetainsArityThree) {
  for (const auto& test_case : lmp::tensor::test::ternary_offset_cases()) {
    expect_case(test_case);
  }
}

TEST(OffsetUtilTest, CPUDispatchRegistersEveryArity) {
  const lmp::tensor::detail::shape_list shape{2, 3};
  const lmp::tensor::detail::shape_list ternary_shape{2, 2};
  const auto unary = lmp::tensor::detail::offset_util_stub_1()(
      DeviceType::CPU, shape,
      std::array{OperandLayout{{2, 3}, {3, 1}}});
  const auto binary = lmp::tensor::detail::offset_util_stub_2()(
      DeviceType::CPU, shape,
      std::array{OperandLayout{{2, 3}, {3, 1}},
                 OperandLayout{{3}, {1}}});
  const auto ternary = lmp::tensor::detail::offset_util_stub_3()(
      DeviceType::CPU, ternary_shape,
      std::array{OperandLayout{{2, 2}, {2, 1}},
                 OperandLayout{{2, 2}, {1, 2}},
                 OperandLayout{{1}, {1}}});

  ASSERT_NE(unary, nullptr);
  ASSERT_NE(binary, nullptr);
  ASSERT_NE(ternary, nullptr);
  EXPECT_EQ(unary->arity(), 1);
  EXPECT_EQ(binary->arity(), 2);
  EXPECT_EQ(ternary->arity(), 3);

  const auto* unary_cpu =
      static_cast<const lmp::tensor::detail::cpu::CPUOffsetUtil<1>*>(
          unary.get());
  EXPECT_EQ(unary_cpu->get(4)[0], 4);
}

TEST(OffsetUtilTest, TensorAndLeadingLayoutAdaptersSnapshotMetadata) {
  Tensor tensor = make_tensor({2, 3, 4}).permute({1, 0, 2});
  const OperandLayout full = operand_layout(
      *lmp::tensor::detail::UnsafeTensorAccessor::getImpl(tensor));
  EXPECT_THAT(full.shape, testing::ElementsAre(3, 2, 4));
  EXPECT_THAT(full.strides, testing::ElementsAre(4, 12, 1));
  const auto calculator =
      make_offset_calculator(full.shape, std::array{full});
  EXPECT_EQ(calculator.get(12)[0], 16);

  const OperandLayout leading = lmp::tensor::detail::leading_operand_layout(
      *lmp::tensor::detail::UnsafeTensorAccessor::getImpl(tensor));
  EXPECT_THAT(leading.shape, testing::ElementsAre(3));
  EXPECT_THAT(leading.strides, testing::ElementsAre(4));

  Tensor rank_one = make_tensor({4});
  EXPECT_THROW(
      lmp::tensor::detail::leading_operand_layout(
          *lmp::tensor::detail::UnsafeTensorAccessor::getImpl(rank_one)),
      std::runtime_error);
}

TEST(OffsetUtilTest, ReductionProjectionUsesRealReducedAxisStride) {
  const auto calculator = make_offset_calculator(
      lmp::tensor::detail::shape_list{3, 1, 2},
      std::array{OperandLayout{{3, 4, 2}, {4, 1, 12}}});

  const stride_t base = calculator.get(5)[0];
  ASSERT_EQ(base, 20);
  const stride_t reduced_axis_stride = 1;
  const std::vector<stride_t> offsets{
      base, base + reduced_axis_stride, base + (2 * reduced_axis_stride),
      base + (3 * reduced_axis_stride)};
  EXPECT_THAT(offsets, testing::ElementsAre(20, 21, 22, 23));
}

TEST(OffsetUtilTest, StructuralValidationRejectsInvalidLayouts) {
  EXPECT_THROW(
      make_offset_calculator(
          lmp::tensor::detail::shape_list{2, 3},
          std::array{OperandLayout{{2, 3}, {3}}}),
      std::runtime_error);
  EXPECT_THROW(
      make_offset_calculator(
          lmp::tensor::detail::shape_list{3},
          std::array{OperandLayout{{2, 3}, {3, 1}}}),
      std::runtime_error);

  lmp::tensor::detail::shape_list excessive_rank(LMP_MAX_DIMS + 1, 1);
  EXPECT_THROW(
      make_offset_calculator(
          excessive_rank,
          std::array{OperandLayout{excessive_rank,
                                   lmp::tensor::detail::stride_list(
                                       LMP_MAX_DIMS + 1, 1)}}),
      std::runtime_error);
}

TEST(OffsetUtilTest, EmptyIterationShapeConstructsWithoutReading) {
  const auto calculator = make_offset_calculator(
      lmp::tensor::detail::shape_list{2, 0, 3},
      std::array{OperandLayout{{2, 0, 3}, {0, 3, 1}}});
  EXPECT_EQ(calculator.rank, 3);
  EXPECT_EQ(calculator.logical_strides[0], 0);
  EXPECT_EQ(calculator.logical_strides[1], 3);
  EXPECT_EQ(calculator.logical_strides[2], 1);
}

TEST(OffsetUtilTest, GetIsSafeAcrossConcurrentReaders) {
  const auto calculator = make_offset_calculator(
      lmp::tensor::detail::shape_list{3, 4, 2},
      std::array{OperandLayout{{3, 4, 2}, {4, 1, 12}}});
  constexpr size_t kReaders = 8;
  constexpr size_t kIterations = 4096;
  std::array<std::vector<stride_t>, kReaders> results;
  std::array<std::thread, kReaders> threads;

  for (size_t reader = 0; reader < kReaders; ++reader) {
    results[reader].resize(kIterations);
    threads[reader] = std::thread([&, reader] {
      for (size_t i = 0; i < kIterations; ++i) {
        results[reader][i] = calculator.get(i % 24)[0];
      }
    });
  }
  for (auto& thread : threads) thread.join();

  for (size_t reader = 1; reader < kReaders; ++reader) {
    EXPECT_EQ(results[reader], results[0]);
  }
}

TEST(OffsetUtilTest, MetaHandlersOwnOffsetsForSelectedRoutes) {
  Tensor a = make_tensor({2, 3});
  Tensor same_shape = make_tensor({2, 3});
  Tensor row = make_tensor({3});
  auto a_impl = lmp::tensor::detail::UnsafeTensorAccessor::getImpl(a);
  auto same_impl =
      lmp::tensor::detail::UnsafeTensorAccessor::getImpl(same_shape);
  auto row_impl = lmp::tensor::detail::UnsafeTensorAccessor::getImpl(row);

  BinaryMetaHandler equal(a_impl.get(), same_impl.get());
  EXPECT_TRUE(equal.has_offset());
  EXPECT_FALSE(equal.expand());
  EXPECT_EQ(equal.offset()->arity(), 2);

  BinaryMetaHandler broadcast(a_impl.get(), row_impl.get());
  EXPECT_TRUE(broadcast.has_offset());
  EXPECT_TRUE(broadcast.expand());

  UnaryMetaHandler contiguous(a_impl.get());
  EXPECT_FALSE(contiguous.has_offset());

  Tensor transposed = a.transpose(0, 1);
  auto transposed_impl =
      lmp::tensor::detail::UnsafeTensorAccessor::getImpl(transposed);
  UnaryMetaHandler strided_unary(transposed_impl.get());
  EXPECT_TRUE(strided_unary.has_offset());
  EXPECT_EQ(strided_unary.offset()->arity(), 1);

  ReductMetaHandler strided_reduction(transposed_impl.get(), 1);
  EXPECT_TRUE(strided_reduction.has_offset());
  EXPECT_EQ(strided_reduction.offset()->arity(), 1);
}

TEST(OffsetUtilTest, ReductionAxisIsValidatedBeforeShapeAccess) {
  Tensor tensor = make_tensor({2, 3});
  auto impl = lmp::tensor::detail::UnsafeTensorAccessor::getImpl(tensor);
  EXPECT_THROW(lmp::tensor::detail::infer_reduct(impl.get(), 2),
               std::runtime_error);
}

static_assert(std::is_same_v<
              decltype(std::declval<
                       const lmp::tensor::detail::offsets_t<1>&>()[0]),
              const stride_t&>);

}  // namespace
