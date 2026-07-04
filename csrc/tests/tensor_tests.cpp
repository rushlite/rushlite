#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <tuple>
#include <vector>

#include "lamp3/tensor/core.hpp"
#include "lamp3/tensor/data_type.hpp"
#include "lamp3/tensor/device_type.hpp"

using lmp::tensor::DataType;
using lmp::tensor::DeviceType;
using lmp::tensor::Scalar;
using lmp::tensor::Tensor;

const Scalar kEps = 1e-5;

using ParamTypes = std::tuple<DeviceType, DataType>;

class TensorOpTest : public testing::Test,
                     public testing::WithParamInterface<ParamTypes> {
 protected:
  TensorOpTest() = default;
  ~TensorOpTest() override = default;

  void SetUp() override {
    device_ = std::get<0>(GetParam());
    dtype_ = std::get<1>(GetParam());

    tensor_f32_A_ =
        Tensor(std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F},
               std::vector<size_t>{3U, 2U}, device_, dtype_);
    tensor_f32_B_ =
        Tensor(std::vector<float>{0.5F, 1.5F, 2.5F, 3.5F, 4.5F, 5.5F},
               std::vector<size_t>{3U, 2U}, device_, dtype_);

    tensor_f32_1x2_broadcast_ =
        Tensor(std::vector<float>{10.0F, 20.0F}, std::vector<size_t>{1U, 2U},
               device_, dtype_);
    tensor_f32_3x1_broadcast_ =
        Tensor(std::vector<float>{10.0F, 20.0F, 30.0F},
               std::vector<size_t>{3U, 1U}, device_, dtype_);
    scalar_tensor_f32_ = Tensor(std::vector<Scalar>{100.0},
                                std::vector<size_t>{1}, device_, dtype_);
    tensor_f32_1x2x1_squeeze_expand_ =
        Tensor(std::vector<float>{7.0F, 8.0F}, std::vector<size_t>{1U, 2U, 1U},
               device_, dtype_);
  }
  void TearDown() override {};
  static std::vector<Scalar> getTenData(const Tensor& ten) {
    return ten.to_vector<Scalar>();
  }
  template <typename T>
  std::vector<T> getIntegerTenData(const Tensor& ten) {
    return ten.to_vector<T>();
  }

  Tensor tensor_f32_A_;
  Tensor tensor_f32_B_;
  Tensor tensor_f32_1x2_broadcast_;
  Tensor tensor_f32_3x1_broadcast_;
  Tensor scalar_tensor_f32_;
  Tensor tensor_f32_1x2x1_squeeze_expand_;

  DeviceType device_;
  DataType dtype_;
};

TEST_P(TensorOpTest, TypeUpcastTest) {
  Tensor tensor_i32_c;
  std::vector<int32_t> data_i32_c_vec = {1, 2, 3, 4, 5, 6};
  tensor_i32_c = Tensor(data_i32_c_vec, std::vector<size_t>{3U, 2U}, device_,
                        DataType::Int32);
  Tensor result = tensor_f32_A_ + tensor_i32_c;

  EXPECT_EQ(result.type(),
            lmp::tensor::type_upcast(tensor_i32_c.type(), dtype_))
      << "Result data type mismatch";
  EXPECT_THAT(result.shape(), ::testing::ElementsAreArray({3U, 2U}))
      << "Result shape mismatch";

  std::vector<Scalar> expected_values = {2.0F, 4.0F, 6.0F, 8.0F, 10.0F, 12.0F};
  EXPECT_THAT(getTenData(result),
              ::testing::Pointwise(::testing::FloatNear(kEps), expected_values))
      << "Result data mismatch";

  Tensor result_rev = tensor_i32_c + tensor_f32_A_;
  EXPECT_EQ(result_rev.type(),
            lmp::tensor::type_upcast(tensor_i32_c.type(), dtype_))
      << "Reversed op: Result data type mismatch";
  EXPECT_THAT(result_rev.shape(), ::testing::ElementsAreArray({3U, 2U}))
      << "Reversed op: Result shape mismatch";
  EXPECT_THAT(getTenData(result_rev),
              ::testing::Pointwise(::testing::FloatNear(kEps), expected_values))
      << "Reversed op: Result data mismatch";
}
TEST_P(TensorOpTest, SimpleBroadcastTest) {
  {
    Tensor result = tensor_f32_A_ + scalar_tensor_f32_;
    EXPECT_EQ(result.type(), dtype_)
        << "Scalar broadcast: Result data type mismatch";
    EXPECT_THAT(result.shape(), ::testing::ElementsAreArray({3U, 2U}))
        << "Scalar broadcast: Result shape mismatch";
    std::vector<Scalar> expected_values = {101.0F, 102.0F, 103.0F,
                                           104.0F, 105.0F, 106.0F};
    EXPECT_THAT(
        getTenData(result),
        ::testing::Pointwise(::testing::FloatNear(kEps), expected_values))
        << "Scalar broadcast: Result data mismatch";
  }

  {
    Tensor result = tensor_f32_A_ + tensor_f32_1x2_broadcast_;
    EXPECT_EQ(result.type(), dtype_)
        << "Row broadcast: Result data type mismatch";
    EXPECT_THAT(result.shape(), ::testing::ElementsAreArray({3U, 2U}))
        << "Row broadcast: Result shape mismatch";
    std::vector<Scalar> expected_values = {11.0F, 22.0F, 13.0F,
                                           24.0F, 15.0F, 26.0F};
    EXPECT_THAT(
        getTenData(result),
        ::testing::Pointwise(::testing::FloatNear(kEps), expected_values))
        << "Row broadcast: Result data mismatch";
  }

  {
    Tensor result = tensor_f32_A_ + tensor_f32_3x1_broadcast_;
    EXPECT_EQ(result.type(), dtype_)
        << "Column broadcast: Result data type mismatch";
    EXPECT_THAT(result.shape(), ::testing::ElementsAreArray({3U, 2U}))
        << "Column broadcast: Result shape mismatch";
    std::vector<Scalar> expected_values = {11.0F, 12.0F, 23.0F,
                                           24.0F, 35.0F, 36.0F};
    EXPECT_THAT(
        getTenData(result),
        ::testing::Pointwise(::testing::FloatNear(kEps), expected_values))
        << "Column broadcast: Result data mismatch";
  }
}
TEST_P(TensorOpTest, ReshapeBroadcastTest) {
  Tensor flat_tensor =
      Tensor(std::vector<float>{10.F, 20.F, 30.F, 40.F, 50.F, 60.F},
             std::vector<size_t>{6U}, device_, dtype_);
  Tensor reshaped_tensor = flat_tensor.reshape({3U, 2U});

  EXPECT_THAT(reshaped_tensor.shape(), ::testing::ElementsAreArray({3U, 2U}))
      << "Reshaped tensor shape mismatch";
  std::vector<Scalar> expected_reshaped_values = {10.F, 20.F, 30.F,
                                                  40.F, 50.F, 60.F};
  EXPECT_THAT(getTenData(reshaped_tensor),
              ::testing::Pointwise(::testing::FloatNear(kEps),
                                   expected_reshaped_values))
      << "Reshaped tensor data mismatch";
  Tensor result = reshaped_tensor + tensor_f32_1x2_broadcast_;

  EXPECT_EQ(result.type(), dtype_)
      << "Reshape-Broadcast: Result data type mismatch";
  EXPECT_THAT(result.shape(), ::testing::ElementsAreArray({3U, 2U}))
      << "Reshape-Broadcast: Result shape mismatch";

  std::vector<Scalar> expected_final_values = {20.0F, 40.0F, 40.0F,
                                               60.0F, 60.0F, 80.0F};
  EXPECT_THAT(
      getTenData(result),
      ::testing::Pointwise(::testing::FloatNear(kEps), expected_final_values))
      << "Reshape-Broadcast: Result data mismatch";
}
TEST_P(TensorOpTest, ExpandBroadcastTest) {
  Tensor tensor_f32_a_expand = tensor_f32_A_.expand_dims(2);
  Tensor result = tensor_f32_a_expand + tensor_f32_1x2x1_squeeze_expand_;

  EXPECT_EQ(result.type(), dtype_)
      << "Expand-Broadcast: Result data type mismatch";
  EXPECT_THAT(result.shape(), ::testing::ElementsAreArray({3U, 2U, 1U}))
      << "Expand-Broadcast: Result shape mismatch";

  std::vector<Scalar> expected_values = {1.0F + 7.0F, 2.0F + 8.0F, 3.0F + 7.0F,
                                         4.0F + 8.0F, 5.0F + 7.0F, 6.0F + 8.0F};

  EXPECT_THAT(getTenData(result),
              ::testing::Pointwise(::testing::FloatNear(kEps), expected_values))
      << "Expand-Broadcast: Result data mismatch";

  Tensor result_rev = tensor_f32_1x2x1_squeeze_expand_ + tensor_f32_a_expand;
  EXPECT_EQ(result_rev.type(), dtype_)
      << "Reversed Expand-Broadcast: Result data type mismatch";
  EXPECT_THAT(result_rev.shape(), ::testing::ElementsAreArray({3U, 2U, 1U}))
      << "Reversed Expand-Broadcast: Result shape mismatch";
  EXPECT_THAT(getTenData(result_rev),
              ::testing::Pointwise(::testing::FloatNear(kEps), expected_values))
      << "Reversed Expand-Broadcast: Result data mismatch";
}
TEST_P(TensorOpTest, ReductSqueezeTest) {
  {
    Tensor sum_axis0_keepdims = lmp::tensor::ops::sum(tensor_f32_A_, 0);
    EXPECT_EQ(sum_axis0_keepdims.type(), dtype_)
        << "Sum axis 0 (keepdims=true): Type mismatch";
    EXPECT_THAT(sum_axis0_keepdims.shape(),
                ::testing::ElementsAreArray({1U, 2U}))
        << "Sum axis 0 (keepdims=true): Shape mismatch";
    std::vector<Scalar> expected_sum_values_axis0 = {9.0F, 12.0F};
    EXPECT_THAT(getTenData(sum_axis0_keepdims),
                ::testing::Pointwise(::testing::FloatNear(kEps),
                                     expected_sum_values_axis0))
        << "Sum axis 0 (keepdims=true): Data mismatch";

    Tensor sum_axis1_keepdims = lmp::tensor::ops::sum(tensor_f32_A_, 1);
    EXPECT_EQ(sum_axis1_keepdims.type(), dtype_)
        << "Sum axis 1 (keepdims=true): Type mismatch";
    EXPECT_THAT(sum_axis1_keepdims.shape(),
                ::testing::ElementsAreArray({3U, 1U}))
        << "Sum axis 1 (keepdims=true): Shape mismatch";
    std::vector<Scalar> expected_sum_values_axis1 = {3.0F, 7.0F, 11.0F};
    EXPECT_THAT(getTenData(sum_axis1_keepdims),
                ::testing::Pointwise(::testing::FloatNear(kEps),
                                     expected_sum_values_axis1))
        << "Sum axis 1 (keepdims=true): Data mismatch";
  }

  {
    Tensor sum_axis0 = lmp::tensor::ops::sum(tensor_f32_A_, 0);
    Tensor squeezed_sum_axis0 = sum_axis0.squeeze(0);
    EXPECT_EQ(squeezed_sum_axis0.type(), dtype_)
        << "Sum axis 0 then squeeze: Type mismatch";
    EXPECT_THAT(squeezed_sum_axis0.shape(), ::testing::ElementsAreArray({2U}))
        << "Sum axis 0 then squeeze: Shape mismatch";
    std::vector<Scalar> expected_sum_values_axis0 = {9.0F, 12.0F};
    EXPECT_THAT(getTenData(squeezed_sum_axis0),
                ::testing::Pointwise(::testing::FloatNear(kEps),
                                     expected_sum_values_axis0))
        << "Sum axis 0 then squeeze: Data mismatch";
  }
  {
    Tensor sum_axis1 = lmp::tensor::ops::sum(tensor_f32_A_, 1);
    Tensor squeezed_sum_axis1 = sum_axis1.squeeze(1);
    EXPECT_EQ(squeezed_sum_axis1.type(), dtype_)
        << "Sum axis 1 then squeeze: Type mismatch";
    EXPECT_THAT(squeezed_sum_axis1.shape(), ::testing::ElementsAreArray({3U}))
        << "Sum axis 1 then squeeze: Shape mismatch";
    std::vector<Scalar> expected_sum_values_axis1 = {3.0F, 7.0F, 11.0F};
    EXPECT_THAT(getTenData(squeezed_sum_axis1),
                ::testing::Pointwise(::testing::FloatNear(kEps),
                                     expected_sum_values_axis1))
        << "Sum axis 1 then squeeze: Data mismatch";
  }

  {
    Tensor squeezed_ax0 = tensor_f32_1x2x1_squeeze_expand_.squeeze(0);
    EXPECT_EQ(squeezed_ax0.type(), dtype_) << "Squeeze ax0: Type mismatch";
    EXPECT_THAT(squeezed_ax0.shape(), ::testing::ElementsAreArray({2U, 1U}))
        << "Squeeze ax0: Shape mismatch";
    std::vector<Scalar> expected_squeeze_data = {7.0F, 8.0F};
    EXPECT_THAT(
        getTenData(squeezed_ax0),
        ::testing::Pointwise(::testing::FloatNear(kEps), expected_squeeze_data))
        << "Squeeze ax0: Data mismatch";

    Tensor squeezed_ax2 = tensor_f32_1x2x1_squeeze_expand_.squeeze(2);
    EXPECT_EQ(squeezed_ax2.type(), dtype_) << "Squeeze ax2: Type mismatch";
    EXPECT_THAT(squeezed_ax2.shape(), ::testing::ElementsAreArray({1U, 2U}))
        << "Squeeze ax2: Shape mismatch";
    EXPECT_THAT(
        getTenData(squeezed_ax2),
        ::testing::Pointwise(::testing::FloatNear(kEps), expected_squeeze_data))
        << "Squeeze ax2: Data mismatch";

    Tensor temp_squeeze = tensor_f32_1x2x1_squeeze_expand_.squeeze(0);
    Tensor final_squeezed = temp_squeeze.squeeze(1);
    EXPECT_EQ(final_squeezed.type(), dtype_)
        << "Sequential squeeze: Type mismatch";
    EXPECT_THAT(final_squeezed.shape(), ::testing::ElementsAreArray({2U}))
        << "Sequential squeeze: Shape mismatch";
    EXPECT_THAT(
        getTenData(final_squeezed),
        ::testing::Pointwise(::testing::FloatNear(kEps), expected_squeeze_data))
        << "Sequential squeeze: Data mismatch";
  }
}
TEST_P(TensorOpTest, ToTest) {
#ifndef LMP_ENABLE_CUDA
  GTEST_SKIP();
#endif

  if (device_ == DeviceType::CPU) {
    Tensor result = tensor_f32_B_.to(DeviceType::CUDA);

    EXPECT_EQ(result.device(), DeviceType::CUDA)
        << "To: Result device mismatch";
    EXPECT_NE(result.device(), tensor_f32_B_.device())
        << "To: Result device mismatch";
    EXPECT_THAT(getTenData(result),
                ::testing::Pointwise(::testing::FloatNear(kEps),
                                     getTenData(tensor_f32_B_)))
        << "To: Result data mismatch";
  } else if (device_ == DeviceType::CUDA) {
    Tensor result = tensor_f32_B_.to(DeviceType::CPU);

    EXPECT_EQ(result.device(), DeviceType::CPU) << "To: Result device mismatch";
    EXPECT_NE(result.device(), tensor_f32_B_.device())
        << "To: Result device mismatch";
    EXPECT_THAT(getTenData(result),
                ::testing::Pointwise(::testing::FloatNear(kEps),
                                     getTenData(tensor_f32_B_)))
        << "To: Result data mismatch";
  } else {
    ASSERT_TRUE(false);
  }
}
TEST_P(TensorOpTest, CopyTest) {
#ifndef LMP_ENABLE_CUDA
  GTEST_SKIP();
#endif
  {
    Tensor tensor_copy_target =
        Tensor(std::vector<float>{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F},
               std::vector<size_t>{3U, 2U}, device_, dtype_);

    tensor_copy_target.copy(tensor_f32_A_);

    EXPECT_EQ(tensor_copy_target.type(), dtype_)
        << "Copy: Result data type mismatch";
    EXPECT_THAT(tensor_copy_target.shape(),
                ::testing::ElementsAreArray({3U, 2U}))
        << "Copy: Result shape mismatch";
    EXPECT_THAT(getTenData(tensor_copy_target),
                ::testing::Pointwise(::testing::FloatNear(kEps),
                                     getTenData(tensor_f32_A_)))
        << "Copy: Result data mismatch";
  }

  {
    std::vector<int32_t> data_i32_target = {0, 0, 0, 0, 0, 0};
    Tensor tensor_i32_target = Tensor(
        data_i32_target, std::vector<size_t>{3U, 2U}, device_, DataType::Int32);

    tensor_i32_target.copy(tensor_f32_A_);

    EXPECT_EQ(tensor_i32_target.type(), DataType::Int32)
        << "Copy with type conversion: Result data type mismatch";
    EXPECT_THAT(tensor_i32_target.shape(),
                ::testing::ElementsAreArray({3U, 2U}))
        << "Copy with type conversion: Result shape mismatch";

    std::vector<int32_t> expected_values = {1, 2, 3, 4, 5, 6};
    EXPECT_THAT(getIntegerTenData<int32_t>(tensor_i32_target),
                ::testing::ElementsAreArray(expected_values))
        << "Copy with type conversion: Result data mismatch";
  }

  if (device_ == DeviceType::CPU) {
    Tensor tensor_cuda_target =
        Tensor(std::vector<float>{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F},
               std::vector<size_t>{3U, 2U}, DeviceType::CUDA, dtype_);

    tensor_cuda_target.copy(tensor_f32_A_);

    EXPECT_EQ(tensor_cuda_target.device(), DeviceType::CUDA)
        << "Copy cross-device: Result device mismatch";
    EXPECT_THAT(getTenData(tensor_cuda_target),
                ::testing::Pointwise(::testing::FloatNear(kEps),
                                     getTenData(tensor_f32_A_)))
        << "Copy cross-device: Result data mismatch";
  } else if (device_ == DeviceType::CUDA) {
    Tensor tensor_cpu_target =
        Tensor(std::vector<float>{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F},
               std::vector<size_t>{3U, 2U}, DeviceType::CPU, dtype_);

    tensor_cpu_target.copy(tensor_f32_A_);

    EXPECT_EQ(tensor_cpu_target.device(), DeviceType::CPU)
        << "Copy cross-device: Result device mismatch";
    EXPECT_THAT(getTenData(tensor_cpu_target),
                ::testing::Pointwise(::testing::FloatNear(kEps),
                                     getTenData(tensor_f32_A_)))
        << "Copy cross-device: Result data mismatch";
  }
}
TEST_P(TensorOpTest, IndexTest) {
  {
    Scalar indexed_value = tensor_f32_A_.index({0, 0});
    EXPECT_NEAR(indexed_value, 1.0F, kEps) << "Index [0,0]: Value mismatch";

    indexed_value = tensor_f32_A_.index({0, 1});
    EXPECT_NEAR(indexed_value, 2.0F, kEps) << "Index [0,1]: Value mismatch";

    indexed_value = tensor_f32_A_.index({1, 0});
    EXPECT_NEAR(indexed_value, 3.0F, kEps) << "Index [1,0]: Value mismatch";

    indexed_value = tensor_f32_A_.index({2, 1});
    EXPECT_NEAR(indexed_value, 6.0F, kEps) << "Index [2,1]: Value mismatch";
  }

  {
    Scalar scalar_value = scalar_tensor_f32_.index({0});
    EXPECT_NEAR(scalar_value, 100.0F, kEps)
        << "Index scalar tensor: Value mismatch";
  }

  {
    Scalar indexed_3d_value = tensor_f32_1x2x1_squeeze_expand_.index({0, 0, 0});
    EXPECT_NEAR(indexed_3d_value, 7.0F, kEps)
        << "Index 3D tensor [0,0,0]: Value mismatch";

    indexed_3d_value = tensor_f32_1x2x1_squeeze_expand_.index({0, 1, 0});
    EXPECT_NEAR(indexed_3d_value, 8.0F, kEps)
        << "Index 3D tensor [0,1,0]: Value mismatch";
  }

  {
    Scalar broadcast_1x2_val = tensor_f32_1x2_broadcast_.index({0, 0});
    EXPECT_NEAR(broadcast_1x2_val, 10.0F, kEps)
        << "Index 1x2 broadcast [0,0]: Value mismatch";

    broadcast_1x2_val = tensor_f32_1x2_broadcast_.index({0, 1});
    EXPECT_NEAR(broadcast_1x2_val, 20.0F, kEps)
        << "Index 1x2 broadcast [0,1]: Value mismatch";

    Scalar broadcast_3x1_val = tensor_f32_3x1_broadcast_.index({1, 0});
    EXPECT_NEAR(broadcast_3x1_val, 20.0F, kEps)
        << "Index 3x1 broadcast [1,0]: Value mismatch";
  }
}
TEST_P(TensorOpTest, FillTest) {
  {
    Tensor tensor_to_fill =
        Tensor(std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F},
               std::vector<size_t>{3U, 2U}, device_, dtype_);

    Scalar fill_value = 42.0;
    tensor_to_fill.fill(fill_value);

    EXPECT_EQ(tensor_to_fill.type(), dtype_)
        << "Fill: Result data type mismatch";
    EXPECT_THAT(tensor_to_fill.shape(), ::testing::ElementsAreArray({3U, 2U}))
        << "Fill: Result shape mismatch";

    std::vector<Scalar> expected_values(6, fill_value);
    EXPECT_THAT(
        getTenData(tensor_to_fill),
        ::testing::Pointwise(::testing::FloatNear(kEps), expected_values))
        << "Fill: Result data mismatch";
  }

  {
    Tensor tensor_to_zero = tensor_f32_B_;
    tensor_to_zero.fill(0.0);

    EXPECT_EQ(tensor_to_zero.type(), dtype_)
        << "Fill zero: Result data type mismatch";
    EXPECT_THAT(tensor_to_zero.shape(), ::testing::ElementsAreArray({3U, 2U}))
        << "Fill zero: Result shape mismatch";

    std::vector<Scalar> expected_zeros(6, 0.0);
    EXPECT_THAT(
        getTenData(tensor_to_zero),
        ::testing::Pointwise(::testing::FloatNear(kEps), expected_zeros))
        << "Fill zero: Result data mismatch";
  }

  {
    Tensor tensor_negative_fill =
        Tensor(std::vector<float>{10.0F, 20.0F}, std::vector<size_t>{1U, 2U},
               device_, dtype_);

    Scalar negative_value = -99.0;
    tensor_negative_fill.fill(negative_value);

    EXPECT_EQ(tensor_negative_fill.type(), dtype_)
        << "Fill negative: Result data type mismatch";
    EXPECT_THAT(tensor_negative_fill.shape(),
                ::testing::ElementsAreArray({1U, 2U}))
        << "Fill negative: Result shape mismatch";

    std::vector<Scalar> expected_negative(2, negative_value);
    EXPECT_THAT(
        getTenData(tensor_negative_fill),
        ::testing::Pointwise(::testing::FloatNear(kEps), expected_negative))
        << "Fill negative: Result data mismatch";
  }

  {
    Tensor scalar_fill_test = scalar_tensor_f32_;
    scalar_fill_test.fill(777.0);

    EXPECT_EQ(scalar_fill_test.type(), dtype_)
        << "Fill scalar tensor: Result data type mismatch";
    EXPECT_THAT(scalar_fill_test.shape(), ::testing::ElementsAreArray({1}))
        << "Fill scalar tensor: Result shape mismatch";

    std::vector<Scalar> expected_scalar_value = {777.0};
    EXPECT_THAT(
        getTenData(scalar_fill_test),
        ::testing::Pointwise(::testing::FloatNear(kEps), expected_scalar_value))
        << "Fill scalar tensor: Result data mismatch";
  }

  {
    Tensor tensor_3d_fill = tensor_f32_1x2x1_squeeze_expand_;
    Scalar fill_3d_value = 123.0;
    tensor_3d_fill.fill(fill_3d_value);

    EXPECT_EQ(tensor_3d_fill.type(), dtype_)
        << "Fill 3D: Result data type mismatch";
    EXPECT_THAT(tensor_3d_fill.shape(),
                ::testing::ElementsAreArray({1U, 2U, 1U}))
        << "Fill 3D: Result shape mismatch";

    std::vector<Scalar> expected_3d_values(2, fill_3d_value);
    EXPECT_THAT(
        getTenData(tensor_3d_fill),
        ::testing::Pointwise(::testing::FloatNear(kEps), expected_3d_values))
        << "Fill 3D: Result data mismatch";
  }
}

TEST_P(TensorOpTest, PowTest) {
  if (dtype_ == DataType::Int16 || dtype_ == DataType::Int32 ||
      dtype_ == DataType::Int64) {
    Tensor int_exp =
        Tensor(std::vector<float>{1.0F, 2.0F, 1.0F, 2.0F, 1.0F, 2.0F}, {3, 2},
               device_, dtype_);
    Tensor result = lmp::tensor::ops::pow(tensor_f32_A_, int_exp);

    std::vector<Scalar> expected_values = {1.0F,  4.0F, 3.0F,
                                           16.0F, 5.0F, 36.0F};

    EXPECT_EQ(result.type(),
              lmp::tensor::type_upcast(tensor_f32_A_.type(), int_exp.type()))
        << "Pow tensor^tensor (int): Result data type mismatch";
    EXPECT_THAT(result.shape(), ::testing::ElementsAreArray({3U, 2U}))
        << "Pow tensor^tensor (int): Result shape mismatch";

  } else {
    Tensor result = lmp::tensor::ops::pow(tensor_f32_A_, tensor_f32_B_);

    std::vector<Scalar> expected_values = {
        std::pow(1.0F, 0.5F), std::pow(2.0F, 1.5F), std::pow(3.0F, 2.5F),
        std::pow(4.0F, 3.5F), std::pow(5.0F, 4.5F), std::pow(6.0F, 5.5F)};

    EXPECT_EQ(result.type(), lmp::tensor::type_upcast(tensor_f32_A_.type(),
                                                      tensor_f32_B_.type()))
        << "Pow tensor^tensor: Result data type mismatch";
    EXPECT_THAT(result.shape(), ::testing::ElementsAreArray({3U, 2U}))
        << "Pow tensor^tensor: Result shape mismatch";
    EXPECT_THAT(
        getTenData(result),
        ::testing::Pointwise(::testing::FloatNear(kEps), expected_values))
        << "Pow tensor^tensor: Result data mismatch";
  }

  Tensor scalar_exp = Tensor(std::vector<Scalar>{2.0}, {1}, device_, dtype_);
  Tensor result_scalar = lmp::tensor::ops::pow(tensor_f32_A_, scalar_exp);

  std::vector<Scalar> expected_scalar_values = {1.0F,  4.0F,  9.0F,
                                                16.0F, 25.0F, 36.0F};

  EXPECT_EQ(result_scalar.type(),
            lmp::tensor::type_upcast(tensor_f32_A_.type(), scalar_exp.type()))
      << "Pow tensor^scalar: Result data type mismatch";
  EXPECT_THAT(result_scalar.shape(), ::testing::ElementsAreArray({3U, 2U}))
      << "Pow tensor^scalar: Result shape mismatch";

  if (dtype_ == DataType::Int16 || dtype_ == DataType::Int32 ||
      dtype_ == DataType::Int64) {
    auto result_data = getTenData(result_scalar);
    for (size_t i = 0; i < expected_scalar_values.size(); ++i) {
      EXPECT_NEAR(result_data[i], expected_scalar_values[i], 1e-1)
          << "Pow tensor^scalar (int): Mismatch at index " << i;
    }
  } else {
    EXPECT_THAT(getTenData(result_scalar),
                ::testing::Pointwise(::testing::FloatNear(kEps),
                                     expected_scalar_values))
        << "Pow tensor^scalar: Result data mismatch";
  }

  Tensor ones_tensor = Tensor(std::vector<Scalar>{1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
                              {3, 2}, device_, dtype_);
  Tensor result_ones = lmp::tensor::ops::pow(tensor_f32_A_, ones_tensor);

  if (dtype_ == DataType::Int16 || dtype_ == DataType::Int32 ||
      dtype_ == DataType::Int64) {
    auto result_data = getTenData(result_ones);
    auto expected_data = getTenData(tensor_f32_A_);
  } else {
    EXPECT_THAT(getTenData(result_ones),
                ::testing::Pointwise(::testing::FloatNear(kEps),
                                     getTenData(tensor_f32_A_)))
        << "Pow tensor^ones: Should equal original tensor";
  }

  Tensor zeros_tensor =
      Tensor(std::vector<Scalar>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, {3, 2}, device_,
             dtype_);
  Tensor result_zeros = lmp::tensor::ops::pow(tensor_f32_A_, zeros_tensor);
  std::vector<Scalar> expected_ones = {1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F};

  if (dtype_ == DataType::Int16 || dtype_ == DataType::Int32 ||
      dtype_ == DataType::Int64) {
    auto result_data = getTenData(result_zeros);
    for (size_t i = 0; i < expected_ones.size(); ++i) {
      EXPECT_NEAR(result_data[i], expected_ones[i], 1e-1)
          << "Pow tensor^zeros (int): Mismatch at index " << i;
    }
  } else {
    EXPECT_THAT(getTenData(result_zeros),
                ::testing::Pointwise(::testing::FloatNear(kEps), expected_ones))
        << "Pow tensor^zeros: Should equal all ones";
  }
}

TEST_P(TensorOpTest, ComparisonTest) {
  {
    Tensor eq_tensor = Tensor(std::vector<Scalar>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
                              {3, 2}, device_, dtype_);
    Tensor result_eq = tensor_f32_A_ == eq_tensor;
    std::vector<Scalar> expected_eq = {1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F};

    EXPECT_THAT(result_eq.shape(), ::testing::ElementsAreArray({3U, 2U}))
        << "Equality: Result shape mismatch";
    EXPECT_THAT(getTenData(result_eq),
                ::testing::Pointwise(::testing::FloatNear(kEps), expected_eq))
        << "Equality: Result data mismatch";
  }

  {
    Tensor result_ne = tensor_f32_A_ != tensor_f32_B_;
    std::vector<Scalar> expected_ne = {1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F};

    EXPECT_THAT(result_ne.shape(), ::testing::ElementsAreArray({3U, 2U}))
        << "Inequality: Result shape mismatch";
    EXPECT_THAT(getTenData(result_ne),
                ::testing::Pointwise(::testing::FloatNear(kEps), expected_ne))
        << "Inequality: Result data mismatch";
  }

  {
    Tensor result_lt = tensor_f32_A_ < tensor_f32_B_;
    std::vector<Scalar> expected_lt = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};

    EXPECT_THAT(result_lt.shape(), ::testing::ElementsAreArray({3U, 2U}))
        << "Less than: Result shape mismatch";
    EXPECT_THAT(getTenData(result_lt),
                ::testing::Pointwise(::testing::FloatNear(kEps), expected_lt))
        << "Less than: Result data mismatch";
  }

  {
    Tensor result_gt = tensor_f32_A_ > tensor_f32_B_;
    std::vector<Scalar> expected_gt = {1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F};

    EXPECT_THAT(result_gt.shape(), ::testing::ElementsAreArray({3U, 2U}))
        << "Greater than: Result shape mismatch";
    EXPECT_THAT(getTenData(result_gt),
                ::testing::Pointwise(::testing::FloatNear(kEps), expected_gt))
        << "Greater than: Result data mismatch";
  }

  {
    Tensor eq_tensor = Tensor(std::vector<Scalar>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
                              {3, 2}, device_, dtype_);
    Tensor result_le = tensor_f32_A_ <= eq_tensor;
    std::vector<Scalar> expected_le = {1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F};

    EXPECT_THAT(result_le.shape(), ::testing::ElementsAreArray({3U, 2U}))
        << "Less than or equal: Result shape mismatch";
    EXPECT_THAT(getTenData(result_le),
                ::testing::Pointwise(::testing::FloatNear(kEps), expected_le))
        << "Less than or equal: Result data mismatch";
  }

  {
    Tensor eq_tensor = Tensor(std::vector<Scalar>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
                              {3, 2}, device_, dtype_);
    Tensor result_ge = tensor_f32_A_ >= eq_tensor;
    std::vector<Scalar> expected_ge = {1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F};

    EXPECT_THAT(result_ge.shape(), ::testing::ElementsAreArray({3U, 2U}))
        << "Greater than or equal: Result shape mismatch";
    EXPECT_THAT(getTenData(result_ge),
                ::testing::Pointwise(::testing::FloatNear(kEps), expected_ge))
        << "Greater than or equal: Result data mismatch";
  }
}

TEST_P(TensorOpTest, BroadcastPowCompTest) {
  {
    if (dtype_ == DataType::Int16 || dtype_ == DataType::Int32 ||
        dtype_ == DataType::Int64) {
      Tensor int_broadcast =
          Tensor(std::vector<float>{2.0F, 3.0F}, {1, 2}, device_, dtype_);
      Tensor result_pow = lmp::tensor::ops::pow(tensor_f32_A_, int_broadcast);
      std::vector<Scalar> expected_pow = {1.0F,  8.0F,  9.0F,
                                          64.0F, 25.0F, 216.0F};

      EXPECT_THAT(result_pow.shape(), ::testing::ElementsAreArray({3U, 2U}))
          << "Broadcast pow (int): Result shape mismatch";

      auto result_data = getTenData(result_pow);
      for (size_t i = 0; i < expected_pow.size(); ++i) {
        EXPECT_NEAR(result_data[i], expected_pow[i], 1e-2)
            << "Broadcast pow (int): Mismatch at index " << i;
      }
    } else {
      Tensor result_pow =
          lmp::tensor::ops::pow(tensor_f32_A_, tensor_f32_1x2_broadcast_ / 10);
      std::vector<Scalar> expected_pow = {
          std::pow(1.0F, 1.0F), std::pow(2.0F, 2.0F), std::pow(3.0F, 1.0F),
          std::pow(4.0F, 2.0F), std::pow(5.0F, 1.0F), std::pow(6.0F, 2.0F)};

      EXPECT_THAT(result_pow.shape(), ::testing::ElementsAreArray({3U, 2U}))
          << "Broadcast pow: Result shape mismatch";
      EXPECT_THAT(
          getTenData(result_pow),
          ::testing::Pointwise(::testing::FloatNear(1e-4), expected_pow))
          << "Broadcast pow: Result data mismatch";
    }
  }

  {
    Tensor result_comp = tensor_f32_A_ > tensor_f32_1x2_broadcast_;
    std::vector<Scalar> expected_comp = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};

    EXPECT_THAT(result_comp.shape(), ::testing::ElementsAreArray({3U, 2U}))
        << "Broadcast comparison: Result shape mismatch";
    EXPECT_THAT(getTenData(result_comp),
                ::testing::Pointwise(::testing::FloatNear(kEps), expected_comp))
        << "Broadcast comparison: Result data mismatch";
  }

  {
    Tensor scalar_3 = Tensor(std::vector<Scalar>{3.5}, {1}, device_, dtype_);
    Tensor result_scalar = tensor_f32_A_ >= scalar_3;
    std::vector<Scalar> expected_scalar = {0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F};

    EXPECT_THAT(result_scalar.shape(), ::testing::ElementsAreArray({3U, 2U}))
        << "Scalar comparison: Result shape mismatch";
  }

  {
    Tensor result_col = tensor_f32_A_ < tensor_f32_3x1_broadcast_;
    std::vector<Scalar> expected_col = {1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F};

    EXPECT_THAT(result_col.shape(), ::testing::ElementsAreArray({3U, 2U}))
        << "Column broadcast comparison: Result shape mismatch";
    EXPECT_THAT(getTenData(result_col),
                ::testing::Pointwise(::testing::FloatNear(kEps), expected_col))
        << "Column broadcast comparison: Result data mismatch";
  }
}

namespace {

std::vector<ParamTypes> GenerateParamCombinations() {
  std::vector<DeviceType> devices;
  devices.push_back(DeviceType::CPU);
#ifdef LMP_ENABLE_CUDA
  devices.push_back(DeviceType::CUDA);
#endif

  std::vector<ParamTypes> comb;
  for (auto dtype : {DataType::Int16, DataType::Int32, DataType::Int64,
                     DataType::Float32, DataType::Float64})
    for (auto device : devices) comb.emplace_back(device, dtype);
  return comb;
}

}  // namespace

// NOLINTNEXTLINE(misc-use-anonymous-namespace)
INSTANTIATE_TEST_SUITE_P(TensorOp, TensorOpTest,
                         testing::ValuesIn(GenerateParamCombinations()));

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
