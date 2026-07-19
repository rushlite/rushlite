#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "lamp3/tensor/core.hpp"

namespace {

using lmp::tensor::DataType;
using lmp::tensor::DeviceType;
using lmp::tensor::Tensor;
using lmp::tensor::TensorImpl;

class EagerLazyBackend final : public lmp::tensor::lazy::LazyBackend {
 public:
  void realize(TensorImpl* impl) override {
    ++realize_count;
    impl->lazy_op()->run_eager(*impl);
  }

  size_t realize_count = 0;
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

Tensor make_tensor(const std::vector<size_t>& shape) {
  const size_t numel =
      shape.empty()
          ? 0
          : std::accumulate(shape.begin(), shape.end(), size_t{1},
                            std::multiplies<>());
  std::vector<float> data(numel);
  std::iota(data.begin(), data.end(), 0.0F);
  return Tensor(data, shape, DeviceType::CPU, DataType::Float32);
}

Tensor make_deferred_tensor(const std::vector<size_t>& shape) {
  Tensor source = make_tensor(shape);
  auto function = std::make_shared<lmp::tensor::lazy::NegFn>(
      std::vector<std::shared_ptr<TensorImpl>>{
          lmp::tensor::detail::UnsafeTensorAccessor::getImpl(source)});
  return lmp::tensor::detail::UnsafeTensorAccessor::fromImpl(
      lmp::tensor::lazy::record(std::move(function)));
}

TEST(TensorLayoutTest, ConstructorsHaveCanonicalStrides) {
  Tensor tensor = make_tensor({2, 3, 4});
  EXPECT_THAT(tensor.strides(), testing::ElementsAre(12, 4, 1));
  EXPECT_TRUE(tensor.is_contiguous());

  Tensor size_one = make_tensor({2, 1, 3});
  EXPECT_THAT(size_one.strides(), testing::ElementsAre(3, 3, 1));
  EXPECT_TRUE(size_one.is_contiguous());

  Tensor empty = make_tensor({2, 0, 3});
  EXPECT_THAT(empty.strides(), testing::ElementsAre(0, 3, 1));
  EXPECT_TRUE(empty.is_contiguous());
}

TEST(TensorLayoutTest, TransposeSwapsOnlyShapeAndStrides) {
  Tensor tensor = make_tensor({2, 3, 4});
  void* storage = tensor.data();

  for (size_t dim0 = 0; dim0 < tensor.shape().size(); ++dim0) {
    for (size_t dim1 = 0; dim1 < tensor.shape().size(); ++dim1) {
      Tensor view = tensor.transpose(dim0, dim1);
      std::vector<size_t> expected_shape = tensor.shape();
      std::vector<lmp::tensor::detail::stride_t> expected_strides =
          tensor.strides();
      std::swap(expected_shape[dim0], expected_shape[dim1]);
      std::swap(expected_strides[dim0], expected_strides[dim1]);

      EXPECT_THAT(view.shape(), testing::ElementsAreArray(expected_shape));
      EXPECT_THAT(view.strides(), testing::ElementsAreArray(expected_strides));
      EXPECT_EQ(view.data(), storage);
    }
  }

  Tensor view = tensor.transpose(0, 2);
  EXPECT_FALSE(view.is_contiguous());
  EXPECT_FLOAT_EQ(view.index({3, 2, 1}), 23.0);
}

TEST(TensorLayoutTest, TransposeValidatesDimensionsAndIsReversible) {
  Tensor tensor = make_tensor({2, 3, 4});
  EXPECT_THROW(tensor.transpose(3, 0), std::runtime_error);
  EXPECT_THROW(tensor.transpose(0, 3), std::runtime_error);

  Tensor restored = tensor.transpose(0, 2).transpose(0, 2);
  EXPECT_THAT(restored.shape(), testing::ElementsAreArray(tensor.shape()));
  EXPECT_THAT(restored.strides(), testing::ElementsAreArray(tensor.strides()));
  EXPECT_TRUE(restored.is_contiguous());
  EXPECT_EQ(restored.data(), tensor.data());
}

TEST(TensorLayoutTest, PermuteReordersShapeAndStrides) {
  Tensor tensor = make_tensor({2, 3, 4});
  Tensor view = tensor.permute({2, 0, 1});

  EXPECT_THAT(view.shape(), testing::ElementsAre(4, 2, 3));
  EXPECT_THAT(view.strides(), testing::ElementsAre(1, 12, 4));
  EXPECT_FALSE(view.is_contiguous());
  EXPECT_EQ(view.data(), tensor.data());
  EXPECT_FLOAT_EQ(view.index({3, 1, 2}), 23.0);

  Tensor restored = view.permute({1, 2, 0});
  EXPECT_THAT(restored.shape(), testing::ElementsAreArray(tensor.shape()));
  EXPECT_THAT(restored.strides(), testing::ElementsAreArray(tensor.strides()));
  EXPECT_TRUE(restored.is_contiguous());
}

TEST(TensorLayoutTest, PermuteValidatesCompletePermutation) {
  Tensor tensor = make_tensor({2, 3, 4});
  EXPECT_THROW(tensor.permute({0, 1}), std::runtime_error);
  EXPECT_THROW(tensor.permute({0, 1, 2, 3}), std::runtime_error);
  EXPECT_THROW(tensor.permute({0, 1, 3}), std::runtime_error);
  EXPECT_THROW(tensor.permute({0, 1, 1}), std::runtime_error);
}

TEST(TensorLayoutTest, SizeOnePermutationCanRemainContiguous) {
  Tensor tensor = make_tensor({2, 1, 3});
  Tensor view = tensor.permute({1, 0, 2});

  EXPECT_THAT(view.shape(), testing::ElementsAre(1, 2, 3));
  EXPECT_THAT(view.strides(), testing::ElementsAre(3, 3, 1));
  EXPECT_TRUE(view.is_contiguous());
}

TEST(TensorLayoutTest, SqueezePreservesRemainingStrides) {
  Tensor tensor = make_tensor({2, 3, 1});
  Tensor view = tensor.transpose(0, 1).squeeze(2);

  EXPECT_THAT(view.shape(), testing::ElementsAre(3, 2));
  EXPECT_THAT(view.strides(), testing::ElementsAre(1, 3));
  EXPECT_FALSE(view.is_contiguous());
  EXPECT_EQ(view.data(), tensor.data());
  EXPECT_FLOAT_EQ(view.index({2, 1}), 5.0);
}

TEST(TensorLayoutTest, ExpandDimsInsertsLayoutPreservingStride) {
  Tensor tensor = make_tensor({2, 3}).transpose(0, 1);

  Tensor front = tensor.expand_dims(0);
  EXPECT_THAT(front.shape(), testing::ElementsAre(1, 3, 2));
  EXPECT_THAT(front.strides(), testing::ElementsAre(3, 1, 3));

  Tensor middle = tensor.expand_dims(1);
  EXPECT_THAT(middle.shape(), testing::ElementsAre(3, 1, 2));
  EXPECT_THAT(middle.strides(), testing::ElementsAre(1, 6, 3));

  Tensor back = tensor.expand_dims(2);
  EXPECT_THAT(back.shape(), testing::ElementsAre(3, 2, 1));
  EXPECT_THAT(back.strides(), testing::ElementsAre(1, 3, 1));

  EXPECT_EQ(front.data(), tensor.data());
  EXPECT_EQ(middle.data(), tensor.data());
  EXPECT_EQ(back.data(), tensor.data());
}

TEST(TensorLayoutTest, IndexValidatesEveryDimension) {
  Tensor tensor = make_tensor({2, 3});
  EXPECT_THROW(tensor.index({0}), std::runtime_error);
  EXPECT_THROW(tensor.index({2, 0}), std::runtime_error);
  EXPECT_THROW(tensor.index({0, 3}), std::runtime_error);
}

TEST(TensorLayoutTest, StorageSharingViewsRealizeDeferredInputsFirst) {
  EagerLazyBackend backend;
  ScopedCpuLazyBackend registration(&backend);

  auto check_realized = [](Tensor deferred, Tensor view) {
    EXPECT_FALSE(
        lmp::tensor::detail::UnsafeTensorAccessor::getImpl(deferred)
            ->is_deferred());
    EXPECT_FALSE(lmp::tensor::detail::UnsafeTensorAccessor::getImpl(view)
                     ->is_deferred());
    EXPECT_EQ(view.data(), deferred.data());
  };

  {
    Tensor deferred = make_deferred_tensor({2, 3});
    check_realized(deferred, deferred.transpose(0, 1));
  }
  {
    Tensor deferred = make_deferred_tensor({2, 3});
    check_realized(deferred, deferred.permute({1, 0}));
  }
  {
    Tensor deferred = make_deferred_tensor({2, 1, 3});
    check_realized(deferred, deferred.squeeze(1));
  }
  {
    Tensor deferred = make_deferred_tensor({2, 3});
    check_realized(deferred, deferred.expand_dims(1));
  }
  {
    Tensor deferred = make_deferred_tensor({2, 3});
    check_realized(deferred, deferred.reshape({3, 2}));
  }

  EXPECT_EQ(backend.realize_count, 5);
}

}  // namespace
