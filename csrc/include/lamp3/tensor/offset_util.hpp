#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <type_traits>

#include "lamp3/common/assert.hpp"
#include "lamp3/tensor/utils/align_utils.hpp"

#ifdef __CUDACC__
#define LMP_OFFSET_HOST_DEVICE __host__ __device__
#else
#define LMP_OFFSET_HOST_DEVICE
#endif

namespace lmp::tensor {

class TensorImpl;

namespace detail {

struct OperandLayout {
  shape_list shape;
  stride_list strides;
};

template <size_t NArgs>
struct OffsetValues {
  stride_t values[NArgs]{};

  LMP_OFFSET_HOST_DEVICE stride_t& operator[](size_t index) noexcept {
    return values[index];
  }

  LMP_OFFSET_HOST_DEVICE const stride_t& operator[](size_t index) const
      noexcept {
    return values[index];
  }
};

template <size_t NArgs>
using offsets_t = OffsetValues<NArgs>;

/**
 * A fixed-capacity logical-index-to-input-offset mapping.
 *
 * This is deliberately an owning value with no pointers so the same lowered
 * representation can be used concurrently on CPU and passed to CUDA kernels
 * by value.
 */
template <size_t NArgs>
struct OffsetCalculator {
  static_assert(NArgs > 0, "OffsetCalculator requires at least one operand");

  size_t rank{};
  stride_t logical_strides[LMP_MAX_DIMS]{};
  stride_t effective_strides[NArgs][LMP_MAX_DIMS]{};

  LMP_OFFSET_HOST_DEVICE offsets_t<NArgs> get(
      size_t logical_index) const noexcept {
    offsets_t<NArgs> offsets{};
    stride_t remaining = static_cast<stride_t>(logical_index);

    for (size_t dim = 0; dim < rank; ++dim) {
      const stride_t logical_stride = logical_strides[dim];
      if (logical_stride == 0) {
        // Empty iteration spaces are never meant to call get(). Keep the
        // value representation safe if a caller violates that contract.
        return offsets;
      }

      const stride_t coordinate = remaining / logical_stride;
      remaining %= logical_stride;
      for (size_t operand = 0; operand < NArgs; ++operand) {
        offsets[operand] +=
            coordinate * effective_strides[operand][dim];
      }
    }

    return offsets;
  }
};

template <size_t NArgs>
OffsetCalculator<NArgs> make_offset_calculator(
    const shape_list& iteration_shape,
    const std::array<OperandLayout, NArgs>& operands) {
  LMP_CHECK(iteration_shape.size() <= LMP_MAX_DIMS)
      << "Offset iteration rank " << iteration_shape.size()
      << " exceeds LMP_MAX_DIMS (" << LMP_MAX_DIMS << ')';

  OffsetCalculator<NArgs> calculator{};
  calculator.rank = iteration_shape.size();

  stride_t logical_stride = 1;
  for (size_t i = iteration_shape.size(); i > 0; --i) {
    const size_t dim = i - 1;
    calculator.logical_strides[dim] = logical_stride;
    logical_stride *= static_cast<stride_t>(iteration_shape[dim]);
  }

  for (size_t operand = 0; operand < NArgs; ++operand) {
    const OperandLayout& layout = operands[operand];
    LMP_CHECK(layout.shape.size() == layout.strides.size())
        << "Operand " << operand << " shape rank " << layout.shape.size()
        << " does not match stride rank " << layout.strides.size();
    LMP_CHECK(layout.shape.size() <= LMP_MAX_DIMS)
        << "Operand " << operand << " rank " << layout.shape.size()
        << " exceeds LMP_MAX_DIMS (" << LMP_MAX_DIMS << ')';
    LMP_CHECK(layout.shape.size() <= iteration_shape.size())
        << "Operand " << operand << " rank " << layout.shape.size()
        << " exceeds iteration rank " << iteration_shape.size();

    const size_t leading_missing_dims =
        iteration_shape.size() - layout.shape.size();
    for (size_t operand_dim = 0; operand_dim < layout.shape.size();
         ++operand_dim) {
      const size_t iteration_dim = leading_missing_dims + operand_dim;
      if (layout.shape[operand_dim] == 1 &&
          iteration_shape[iteration_dim] > 1) {
        calculator.effective_strides[operand][iteration_dim] = 0;
      } else {
        calculator.effective_strides[operand][iteration_dim] =
            layout.strides[operand_dim];
      }
    }
  }

  return calculator;
}

static_assert(std::is_trivially_copyable_v<OffsetCalculator<1>>);
static_assert(std::is_trivially_copyable_v<OffsetCalculator<2>>);
static_assert(std::is_trivially_copyable_v<OffsetCalculator<3>>);

class OffsetUtil {
 public:
  virtual ~OffsetUtil() = default;

  size_t arity() const noexcept { return arity_; }

 protected:
  explicit OffsetUtil(size_t arity) : arity_(arity) {}

 private:
  size_t arity_;
};

OperandLayout operand_layout(const TensorImpl& tensor);
OperandLayout leading_operand_layout(const TensorImpl& tensor,
                                     size_t trailing_dims = 2);

template <size_t NArgs>
using offset_util_fn = std::unique_ptr<OffsetUtil> (*)(
    const shape_list& iteration_shape,
    const std::array<OperandLayout, NArgs>& operands);

}  // namespace detail
}  // namespace lmp::tensor

#undef LMP_OFFSET_HOST_DEVICE
