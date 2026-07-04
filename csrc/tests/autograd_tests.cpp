#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <vector>

#include "lamp3/autograd/core.hpp"
#include "lamp3/autograd/functions/view_ops.hpp"
#include "lamp3/tensor/core.hpp"
#include "lamp3/tensor/device_type.hpp"

using lmp::autograd::Variable;
using lmp::tensor::DataType;
using lmp::tensor::DeviceType;
using lmp::tensor::Scalar;
using lmp::tensor::Tensor;

const Scalar kEps = 1e-5;

using ParamTypes = std::tuple<DeviceType>;

class VariableOpTest
    : public testing::Test,
      public testing::WithParamInterface<std::tuple<DeviceType>> {
 protected:
  VariableOpTest() = default;
  ~VariableOpTest() override = default;

  void SetUp() override {
    device_ = std::get<0>(GetParam());

    a_data_ = Tensor(std::vector<Scalar>{1.0, 2.0, 3.0, 4.0, 5.0, 2.0},
                     std::vector<size_t>{3U, 2U}, device_, DataType::Float32);
    b_data_ = Tensor(std::vector<Scalar>{-1.0, 4.0, -2.0, 0.0, 3.0, 0.5},
                     std::vector<size_t>{3U, 2U}, device_, DataType::Float32);
    a_ = Variable(a_data_, true);
    b_ = Variable(b_data_, true);
  }
  void TearDown() override {};
  static std::vector<Scalar> getTenData(const Tensor& ten) {
    return ten.to_vector<Scalar>();
  }

  Variable a_, b_;
  Tensor a_data_, b_data_;
  DeviceType device_;
};

TEST_P(VariableOpTest, AddTest) {
  Variable res = a_ + b_;
  EXPECT_THAT(getTenData(res.data()),
              ::testing::Pointwise(::testing::FloatNear(kEps),
                                   {0.0, 6.0, 1.0, 4.0, 8.0, 2.5}))
      << "Forward data mismatch";
  EXPECT_THAT(res.data().shape(), ::testing::ElementsAreArray({3U, 2U}))
      << "Forward shape mismatch";
  res.backward();
  EXPECT_THAT(getTenData(a_.grad()),
              ::testing::Pointwise(::testing::FloatNear(kEps),
                                   {1.0, 1.0, 1.0, 1.0, 1.0, 1.0}))
      << "Gradient mismatch for variable 1";
  EXPECT_THAT(getTenData(b_.grad()),
              ::testing::Pointwise(::testing::FloatNear(kEps),
                                   {1.0, 1.0, 1.0, 1.0, 1.0, 1.0}))
      << "Gradient mismatch for variable 2";
}

TEST_P(VariableOpTest, SubTest) {
  Variable res = a_ - b_;
  EXPECT_THAT(getTenData(res.data()),
              ::testing::Pointwise(::testing::FloatNear(kEps),
                                   {2.0, -2.0, 5.0, 4.0, 2.0, 1.5}))
      << "Forward data mismatch";
  EXPECT_THAT(res.data().shape(), ::testing::ElementsAreArray({3U, 2U}))
      << "Forward shape mismatch";
  res.backward();
  EXPECT_THAT(getTenData(a_.grad()),
              ::testing::Pointwise(::testing::FloatNear(kEps),
                                   {1.0, 1.0, 1.0, 1.0, 1.0, 1.0}))
      << "Gradient mismatch for variable 1";
  EXPECT_THAT(getTenData(b_.grad()),
              ::testing::Pointwise(::testing::FloatNear(kEps),
                                   {-1.0, -1.0, -1.0, -1.0, -1.0, -1.0}))
      << "Gradient mismatch for variable 2";
}

TEST_P(VariableOpTest, MulTest) {
  Variable res = a_ * b_;
  EXPECT_THAT(getTenData(res.data()),
              ::testing::Pointwise(::testing::FloatNear(kEps),
                                   {-1.0, 8.0, -6.0, 0.0, 15.0, 1.0}))
      << "Forward data mismatch";
  EXPECT_THAT(res.data().shape(), ::testing::ElementsAreArray({3U, 2U}))
      << "Forward shape mismatch";
  res.backward();
  EXPECT_THAT(getTenData(a_.grad()),
              ::testing::Pointwise(::testing::FloatNear(kEps),
                                   {-1.0, 4.0, -2.0, 0.0, 3.0, 0.5}))
      << "Gradient mismatch for variable 1";
  EXPECT_THAT(getTenData(b_.grad()),
              ::testing::Pointwise(::testing::FloatNear(kEps),
                                   {1.0, 2.0, 3.0, 4.0, 5.0, 2.0}))
      << "Gradient mismatch for variable 2";
}

TEST_P(VariableOpTest, DivTest) {
  Variable res = a_ / b_;
  EXPECT_THAT(getTenData(res.data()),
              ::testing::Pointwise(
                  ::testing::FloatNear(kEps),
                  {-1.0, 0.5, -1.5, std::numeric_limits<Scalar>::infinity(),
                   5.0 / 3.0, 4.0}))
      << "Forward data mismatch";
  EXPECT_THAT(res.data().shape(), ::testing::ElementsAreArray({3U, 2U}))
      << "Forward shape mismatch";
  res.backward();
  EXPECT_THAT(getTenData(a_.grad()),
              ::testing::Pointwise(::testing::FloatNear(kEps),
                                   {1.0 / -1.0, 1.0 / 4.0, 1.0 / -2.0,
                                    1.0 / 0.0, 1.0 / 3.0, 1.0 / 0.5}))
      << "Gradient mismatch for variable 1";
  EXPECT_THAT(getTenData(b_.grad()),
              ::testing::Pointwise(::testing::FloatNear(kEps),
                                   {-1.0 / (-1.0 * -1.0), -2.0 / (4.0 * 4.0),
                                    -3.0 / (-2.0 * -2.0), -4.0 / (0.0 * 0.0),
                                    -5.0 / (3.0 * 3.0), -2.0 / (0.5 * 0.5)}))
      << "Gradient mismatch for variable 2";
}

TEST_P(VariableOpTest, ExpTest) {
  Variable res = lmp::autograd::ops::exp(b_);
  EXPECT_THAT(getTenData(res.data()),
              ::testing::Pointwise(::testing::FloatNear(kEps),
                                   {exp(-1.0), exp(4.0), exp(-2.0), exp(0.0),
                                    exp(3.0), exp(0.5)}))
      << "Forward data mismatch";
  EXPECT_THAT(res.data().shape(), ::testing::ElementsAreArray({3U, 2U}))
      << "Forward shape mismatch";
  res.backward();
  EXPECT_THAT(getTenData(b_.grad()),
              ::testing::Pointwise(::testing::FloatNear(kEps),
                                   {exp(-1.0), exp(4.0), exp(-2.0), exp(0.0),
                                    exp(3.0), exp(0.5)}))
      << "Gradient mismatch";
}

TEST_P(VariableOpTest, LogTest) {
  Variable res = lmp::autograd::ops::log(a_);
  EXPECT_THAT(getTenData(res.data()),
              ::testing::Pointwise(
                  ::testing::FloatNear(kEps),
                  {log(1.0), log(2.0), log(3.0), log(4.0), log(5.0), log(2.0)}))
      << "Forward data mismatch";
  EXPECT_THAT(res.data().shape(), ::testing::ElementsAreArray({3U, 2U}))
      << "Forward shape mismatch";
  res.backward();
  EXPECT_THAT(getTenData(a_.grad()),
              ::testing::Pointwise(::testing::FloatNear(kEps),
                                   {1.0 / 1.0, 1.0 / 2.0, 1.0 / 3.0, 1.0 / 4.0,
                                    1.0 / 5.0, 1.0 / 2.0}))
      << "Gradient mismatch";
}

TEST_P(VariableOpTest, MatMulTest) {
  Tensor b_mat = Tensor(std::vector<Scalar>{-1.0, 4.0},
                        std::vector<size_t>{2U, 1U}, device_);
  Variable b_mat_var(b_mat, true);
  Variable res = lmp::autograd::ops::matmul(a_, b_mat_var);
  EXPECT_THAT(
      getTenData(res.data()),
      ::testing::Pointwise(::testing::FloatNear(kEps), {7.0, 13.0, 3.0}))
      << "Forward data mismatch";
  EXPECT_THAT(res.data().shape(), ::testing::ElementsAreArray({3U, 1U}))
      << "Forward shape mismatch";
  res.backward();
  EXPECT_THAT(getTenData(a_.grad()),
              ::testing::Pointwise(::testing::FloatNear(kEps),
                                   {-1.0, 4.0, -1.0, 4.0, -1.0, 4.0}))
      << "Gradient mismatch for variable 1";
  EXPECT_THAT(getTenData(b_mat_var.grad()),
              ::testing::Pointwise(::testing::FloatNear(kEps), {9.0, 8.0}))
      << "Gradient mismatch for variable 2";
}

TEST_P(VariableOpTest, TransposeTest) {
  Variable res = lmp::autograd::ops::transpose(a_);
  EXPECT_THAT(getTenData(res.data()),
              ::testing::Pointwise(::testing::FloatNear(kEps),
                                   {1.0, 3.0, 5.0, 2.0, 4.0, 2.0}))
      << "Forward data mismatch";
  EXPECT_THAT(res.data().shape(), ::testing::ElementsAreArray({2U, 3U}))
      << "Forward shape mismatch";
  res.backward();
  EXPECT_THAT(getTenData(a_.grad()),
              ::testing::Pointwise(::testing::FloatNear(kEps),
                                   {1.0, 1.0, 1.0, 1.0, 1.0, 1.0}))
      << "Gradient mismatch";
}

TEST_P(VariableOpTest, SumTest) {
  Variable res = lmp::autograd::ops::sum(a_, 1);
  EXPECT_THAT(getTenData(res.data()),
              ::testing::Pointwise(::testing::FloatNear(kEps), {3.0, 7.0, 7.0}))
      << "Forward data mismatch";
  EXPECT_THAT(res.data().shape(), ::testing::ElementsAreArray({3U, 1U}))
      << "Forward shape mismatch";
  res.backward();
  EXPECT_THAT(getTenData(a_.grad()),
              ::testing::Pointwise(::testing::FloatNear(kEps),
                                   {1.0, 1.0, 1.0, 1.0, 1.0, 1.0}))
      << "Gradient mismatch";
}

TEST_P(VariableOpTest, MaxTest) {
  Variable res = lmp::autograd::ops::max(a_, 1);
  EXPECT_THAT(getTenData(res.data()),
              ::testing::Pointwise(::testing::FloatNear(kEps), {2.0, 4.0, 5.0}))
      << "Forward data mismatch";
  EXPECT_THAT(res.data().shape(), ::testing::ElementsAreArray({3U, 1U}))
      << "Forward shape mismatch";
  res.backward();
  EXPECT_THAT(
      getTenData(a_.grad()),
      ::testing::Pointwise(::testing::FloatNear(kEps), {0, 1, 0, 1, 1, 0}))
      << "Gradient mismatch";
}

TEST_P(VariableOpTest, MinTest) {
  Variable res = lmp::autograd::ops::min(a_, 1);
  EXPECT_THAT(getTenData(res.data()),
              ::testing::Pointwise(::testing::FloatNear(kEps), {1.0, 3.0, 2.0}))
      << "Forward data mismatch";
  EXPECT_THAT(res.data().shape(), ::testing::ElementsAreArray({3U, 1U}))
      << "Forward shape mismatch";
  res.backward();
  EXPECT_THAT(
      getTenData(a_.grad()),
      ::testing::Pointwise(::testing::FloatNear(kEps), {1, 0, 1, 0, 0, 1}))
      << "Gradient mismatch";
}

TEST_P(VariableOpTest, ReshapeTest) {
  Variable res = lmp::autograd::ops::reshape(a_, {2, 1, 3});
  EXPECT_THAT(getTenData(res.data()),
              ::testing::Pointwise(::testing::FloatNear(kEps),
                                   {1.0, 2.0, 3.0, 4.0, 5.0, 2.0}))
      << "Forward data mismatch";
  EXPECT_THAT(res.data().shape(), ::testing::ElementsAreArray({2U, 1U, 3U}))
      << "Forward shape mismatch";
  res.backward();
  EXPECT_THAT(
      getTenData(a_.grad()),
      ::testing::Pointwise(::testing::FloatNear(kEps), {1, 1, 1, 1, 1, 1}))
      << "Gradient data mismatch";
  EXPECT_THAT(a_.grad().shape(), ::testing::ElementsAreArray({3U, 2U}))
      << "Gradient shape mismatch";
}

TEST_P(VariableOpTest, ExpandDimsTest) {
  Variable res = lmp::autograd::ops::expand_dims(a_, 0);
  EXPECT_THAT(getTenData(res.data()),
              ::testing::Pointwise(::testing::FloatNear(kEps),
                                   {1.0, 2.0, 3.0, 4.0, 5.0, 2.0}))
      << "Forward data mismatch";
  EXPECT_THAT(res.data().shape(), ::testing::ElementsAreArray({1U, 3U, 2U}))
      << "Forward shape mismatch";
  res.backward();
  EXPECT_THAT(
      getTenData(a_.grad()),
      ::testing::Pointwise(::testing::FloatNear(kEps), {1, 1, 1, 1, 1, 1}))
      << "Gradient data mismatch";
  EXPECT_THAT(a_.grad().shape(), ::testing::ElementsAreArray({3U, 2U}))
      << "Gradient shape mismatch";
}

TEST_P(VariableOpTest, SqueezeTest) {
  Tensor squeeze_data =
      Tensor(std::vector<Scalar>{1.0, 2.0, 3.0}, std::vector<size_t>{3U, 1U},
             device_, DataType::Float32);
  Variable squeeze_var = Variable(squeeze_data, true);
  Variable res = lmp::autograd::ops::squeeze(squeeze_var, 1);
  EXPECT_THAT(getTenData(res.data()),
              ::testing::Pointwise(::testing::FloatNear(kEps), {1.0, 2.0, 3.0}))
      << "Forward data mismatch";
  EXPECT_THAT(res.data().shape(), ::testing::ElementsAreArray({3U}))
      << "Forward shape mismatch";
  res.backward();
  EXPECT_THAT(getTenData(squeeze_var.grad()),
              ::testing::Pointwise(::testing::FloatNear(kEps), {1, 1, 1}))
      << "Gradient data mismatch";
  EXPECT_THAT(squeeze_var.grad().shape(), ::testing::ElementsAreArray({3U, 1U}))
      << "Gradient shape mismatch";
}

TEST_P(VariableOpTest, BroadcastAddTest) {
  Tensor broadcast_data =
      Tensor(std::vector<Scalar>{10.0, 20.0}, std::vector<size_t>{1U, 2U},
             device_, DataType::Float32);
  Variable broadcast_var = Variable(broadcast_data, true);

  Variable res = a_ + broadcast_var;
  EXPECT_THAT(getTenData(res.data()),
              ::testing::Pointwise(::testing::FloatNear(kEps),
                                   {11.0, 22.0, 13.0, 24.0, 15.0, 22.0}))
      << "Broadcast forward data mismatch";
  EXPECT_THAT(res.data().shape(), ::testing::ElementsAreArray({3U, 2U}))
      << "Broadcast forward shape mismatch";

  res.backward();
  EXPECT_THAT(getTenData(a_.grad()),
              ::testing::Pointwise(::testing::FloatNear(kEps),
                                   {1.0, 1.0, 1.0, 1.0, 1.0, 1.0}))
      << "Gradient mismatch for variable a in broadcast";
  EXPECT_THAT(getTenData(broadcast_var.grad()),
              ::testing::Pointwise(::testing::FloatNear(kEps), {3.0, 3.0}))
      << "Gradient mismatch for broadcast variable";
  EXPECT_THAT(broadcast_var.grad().shape(),
              ::testing::ElementsAreArray({1U, 2U}))
      << "Gradient shape mismatch for broadcast variable";
}

TEST_P(VariableOpTest, ConstructorTest) {
  Variable zeros_var =
      lmp::autograd::zeros({2, 3}, true, device_, DataType::Float32);
  EXPECT_THAT(getTenData(zeros_var.data()),
              ::testing::Pointwise(::testing::FloatNear(kEps),
                                   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}))
      << "Zeros constructor data mismatch";
  EXPECT_THAT(zeros_var.data().shape(), ::testing::ElementsAreArray({2U, 3U}))
      << "Zeros constructor shape mismatch";
  EXPECT_TRUE(zeros_var.requires_grad())
      << "Zeros constructor requires_grad mismatch";

  Variable ones_var =
      lmp::autograd::ones({2, 2}, false, device_, DataType::Float32);
  EXPECT_THAT(
      getTenData(ones_var.data()),
      ::testing::Pointwise(::testing::FloatNear(kEps), {1.0, 1.0, 1.0, 1.0}))
      << "Ones constructor data mismatch";
  EXPECT_THAT(ones_var.data().shape(), ::testing::ElementsAreArray({2U, 2U}))
      << "Ones constructor shape mismatch";
  EXPECT_FALSE(ones_var.requires_grad())
      << "Ones constructor requires_grad mismatch";

  Variable rand_var =
      lmp::autograd::rand({100}, true, device_, DataType::Float32);
  std::vector<Scalar> rand_data = getTenData(rand_var.data());

  for (Scalar val : rand_data) {
    EXPECT_GE(val, 0.0) << "Random value below 0";
    EXPECT_LE(val, 1.0) << "Random value above 1";
  }

  Scalar sum = 0.0;
  for (Scalar val : rand_data) {
    sum += val;
  }
  Scalar avg = sum / rand_data.size();
  EXPECT_NEAR(avg, 0.5, 0.1) << "Random values average not near 0.5";

  EXPECT_THAT(rand_var.data().shape(), ::testing::ElementsAreArray({100U}))
      << "Rand constructor shape mismatch";
  EXPECT_TRUE(rand_var.requires_grad())
      << "Rand constructor requires_grad mismatch";
}

TEST_P(VariableOpTest, CopyFillTest) {
  Tensor source_data =
      Tensor(std::vector<Scalar>{100.0, 200.0, 300.0, 400.0, 500.0, 600.0},
             std::vector<size_t>{3U, 2U}, device_, DataType::Float32);
  Variable source_var = Variable(source_data, true);

  Variable temp_res = a_ * 2.0;

  a_.copy(source_var);
  EXPECT_THAT(getTenData(a_.data()),
              ::testing::Pointwise(::testing::FloatNear(kEps),
                                   {100.0, 200.0, 300.0, 400.0, 500.0, 600.0}))
      << "Copy data mismatch";

  a_.fill(42.0);
  EXPECT_THAT(getTenData(a_.data()),
              ::testing::Pointwise(::testing::FloatNear(kEps),
                                   {42.0, 42.0, 42.0, 42.0, 42.0, 42.0}))
      << "Fill data mismatch";
}

TEST_P(VariableOpTest, ZeroGradTest) {
  Variable res1 = a_ * b_;
  Variable res2 = res1 + a_;
  res2.backward();

  std::vector<Scalar> a_grad_before = getTenData(a_.grad());
  std::vector<Scalar> b_grad_before = getTenData(b_.grad());

  bool a_has_nonzero = false;
  bool b_has_nonzero = false;
  for (Scalar val : a_grad_before) {
    if (std::abs(val) > kEps) a_has_nonzero = true;
  }
  for (Scalar val : b_grad_before) {
    if (std::abs(val) > kEps) b_has_nonzero = true;
  }
  EXPECT_TRUE(a_has_nonzero)
      << "Variable 'a' should have non-zero gradients before zero_grad";
  EXPECT_TRUE(b_has_nonzero)
      << "Variable 'b' should have non-zero gradients before zero_grad";

  a_.zero_grad();
  b_.zero_grad();

  EXPECT_THAT(getTenData(a_.grad()),
              ::testing::Pointwise(::testing::FloatNear(kEps),
                                   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}))
      << "Variable 'a' gradients should be zero after zero_grad";
  EXPECT_THAT(getTenData(b_.grad()),
              ::testing::Pointwise(::testing::FloatNear(kEps),
                                   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}))
      << "Variable 'b' gradients should be zero after zero_grad";
}

TEST_P(VariableOpTest, GradModeDisablesRecordingTest) {
  ASSERT_TRUE(lmp::autograd::is_grad_enabled());
  {
    lmp::autograd::GradModeGuard guard(false);
    EXPECT_FALSE(lmp::autograd::is_grad_enabled());

    Variable res = a_ * b_;
    EXPECT_FALSE(res.requires_grad())
        << "Ops inside GradModeGuard(false) must not require grad";
    EXPECT_EQ(res.grad_fn().lock(), nullptr)
        << "Ops inside GradModeGuard(false) must not record a grad_fn";
    EXPECT_THAT(getTenData(res.data()),
                ::testing::Pointwise(::testing::FloatNear(kEps),
                                     {-1.0, 8.0, -6.0, 0.0, 15.0, 1.0}))
        << "Forward data must still be computed under GradModeGuard(false)";

    // explicit leaf construction still honors the requires_grad flag
    Variable leaf = Variable(a_data_, true);
    EXPECT_TRUE(leaf.requires_grad())
        << "Explicit leaf creation must be unaffected by grad mode";
  }
  EXPECT_TRUE(lmp::autograd::is_grad_enabled())
      << "GradModeGuard must restore the previous mode";

  Variable res = a_ * b_;
  EXPECT_TRUE(res.requires_grad())
      << "Recording must resume after the guard is destroyed";
  EXPECT_NE(res.grad_fn().lock(), nullptr);
}

TEST_P(VariableOpTest, GradModeGuardNestingTest) {
  lmp::autograd::GradModeGuard outer(false);
  EXPECT_FALSE(lmp::autograd::is_grad_enabled());
  {
    lmp::autograd::GradModeGuard inner(true);
    EXPECT_TRUE(lmp::autograd::is_grad_enabled());
  }
  EXPECT_FALSE(lmp::autograd::is_grad_enabled())
      << "Inner guard must restore the outer guard's mode, not the default";
}

TEST_P(VariableOpTest, GradModeBackwardUnaffectedTest) {
  Variable res = a_ * b_;  // graph built with grad enabled
  {
    lmp::autograd::GradModeGuard guard(false);
    res.backward();
  }
  EXPECT_THAT(getTenData(a_.grad()),
              ::testing::Pointwise(::testing::FloatNear(kEps),
                                   {-1.0, 4.0, -2.0, 0.0, 3.0, 0.5}))
      << "backward() of a pre-built graph must work inside GradModeGuard";
  EXPECT_THAT(getTenData(b_.grad()),
              ::testing::Pointwise(::testing::FloatNear(kEps),
                                   {1.0, 2.0, 3.0, 4.0, 5.0, 2.0}))
      << "backward() of a pre-built graph must work inside GradModeGuard";
}

namespace {

std::vector<ParamTypes> GenerateParams() {
  std::vector<ParamTypes> devices;
  devices.emplace_back(DeviceType::CPU);
#ifdef LMP_ENABLE_CUDA
  devices.emplace_back(DeviceType::CUDA);
#endif
  return devices;
}

}  // namespace

// NOLINTNEXTLINE(misc-use-anonymous-namespace)
INSTANTIATE_TEST_SUITE_P(VariableOp, VariableOpTest,
                         testing::ValuesIn(GenerateParams()));

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
