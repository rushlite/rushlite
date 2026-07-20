#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <type_traits>
#include <vector>

#include "lamp3/tensor/cuda/offset_util.cuh"
#include "offset_util_test_cases.hpp"

namespace {

using lmp::tensor::DeviceType;
using lmp::tensor::detail::OffsetCalculator;
using lmp::tensor::detail::OperandLayout;
using lmp::tensor::detail::make_offset_calculator;

template <size_t NArgs>
__global__ void calculate_offsets_kernel(
    OffsetCalculator<NArgs> calculator, const size_t* logical_indices,
    lmp::tensor::detail::offsets_t<NArgs>* offsets, size_t count) {
  for (size_t i = (blockIdx.x * blockDim.x) + threadIdx.x; i < count;
       i += gridDim.x * blockDim.x) {
    offsets[i] = calculator.get(logical_indices[i]);
  }
}

bool has_cuda_device() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

template <size_t NArgs>
void expect_cuda_case(const lmp::tensor::test::OffsetCase<NArgs>& test_case) {
  SCOPED_TRACE(test_case.name);
  ASSERT_EQ(test_case.logical_indices.size(),
            test_case.expected_offsets.size());

  const OffsetCalculator<NArgs> calculator =
      make_offset_calculator(test_case.iteration_shape, test_case.operands);
  const size_t count = test_case.logical_indices.size();
  size_t* device_indices = nullptr;
  lmp::tensor::detail::offsets_t<NArgs>* device_offsets = nullptr;

  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_indices),
                       count * sizeof(size_t)),
            cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_offsets),
                       count *
                           sizeof(lmp::tensor::detail::offsets_t<NArgs>)),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_indices, test_case.logical_indices.data(),
                       count * sizeof(size_t), cudaMemcpyHostToDevice),
            cudaSuccess);

  constexpr size_t kThreads = 64;
  const size_t blocks = (count + kThreads - 1) / kThreads;
  for (size_t launch = 0; launch < 3; ++launch) {
    calculate_offsets_kernel<<<blocks, kThreads>>>(
        calculator, device_indices, device_offsets, count);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  }

  std::vector<lmp::tensor::detail::offsets_t<NArgs>> actual(count);
  ASSERT_EQ(cudaMemcpy(actual.data(), device_offsets,
                       count *
                           sizeof(lmp::tensor::detail::offsets_t<NArgs>),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(cudaFree(device_offsets), cudaSuccess);
  EXPECT_EQ(cudaFree(device_indices), cudaSuccess);

  for (size_t i = 0; i < count; ++i) {
    for (size_t operand = 0; operand < NArgs; ++operand) {
      EXPECT_EQ(actual[i][operand], test_case.expected_offsets[i][operand])
          << "logical index " << test_case.logical_indices[i] << ", operand "
          << operand;
    }
  }
}

TEST(OffsetUtilCUDATest, DeviceCalculatorMatchesCPUFixtures) {
  if (!has_cuda_device()) GTEST_SKIP() << "No CUDA device available";

  for (const auto& test_case : lmp::tensor::test::unary_offset_cases()) {
    expect_cuda_case(test_case);
  }
  for (const auto& test_case : lmp::tensor::test::binary_offset_cases()) {
    expect_cuda_case(test_case);
  }
  for (const auto& test_case : lmp::tensor::test::ternary_offset_cases()) {
    expect_cuda_case(test_case);
  }
}

TEST(OffsetUtilCUDATest, DispatchRegistersEveryArity) {
  const lmp::tensor::detail::shape_list shape{2, 3};
  const lmp::tensor::detail::shape_list ternary_shape{2, 2};
  const auto unary = lmp::tensor::detail::offset_util_stub_1()(
      DeviceType::CUDA, shape,
      std::array{OperandLayout{{2, 3}, {3, 1}}});
  const auto binary = lmp::tensor::detail::offset_util_stub_2()(
      DeviceType::CUDA, shape,
      std::array{OperandLayout{{2, 3}, {3, 1}},
                 OperandLayout{{3}, {1}}});
  const auto ternary = lmp::tensor::detail::offset_util_stub_3()(
      DeviceType::CUDA, ternary_shape,
      std::array{OperandLayout{{2, 2}, {2, 1}},
                 OperandLayout{{2, 2}, {1, 2}},
                 OperandLayout{{1}, {1}}});

  ASSERT_NE(unary, nullptr);
  ASSERT_NE(binary, nullptr);
  ASSERT_NE(ternary, nullptr);
  EXPECT_EQ(unary->arity(), 1);
  EXPECT_EQ(binary->arity(), 2);
  EXPECT_EQ(ternary->arity(), 3);

  const auto* binary_cuda =
      static_cast<const lmp::tensor::detail::cuda::CUDAOffsetUtil<2>*>(
          binary.get());
  EXPECT_EQ(binary_cuda->calculator().get(4)[0], 4);
  EXPECT_EQ(binary_cuda->calculator().get(4)[1], 1);
}

static_assert(
    std::is_trivially_copyable_v<lmp::tensor::detail::OffsetCalculator<1>>);
static_assert(
    std::is_trivially_copyable_v<lmp::tensor::detail::OffsetCalculator<2>>);
static_assert(
    std::is_trivially_copyable_v<lmp::tensor::detail::OffsetCalculator<3>>);

}  // namespace
