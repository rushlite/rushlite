#pragma once

#include <array>
#include <memory>

#include "lamp3/tensor/cpu/offset_util.hpp"

namespace lmp::tensor::detail::cuda {

/// @internal
/**
 * Host-side dispatch owner for a CUDA-safe calculator value.
 */
template <size_t NArgs>
class CUDAOffsetUtil final : public OffsetUtil {
 public:
  CUDAOffsetUtil(const shape_list& iteration_shape,
                 const std::array<OperandLayout, NArgs>& operands)
      : OffsetUtil(NArgs),
        calculator_(make_offset_calculator(iteration_shape, operands)) {}

  const OffsetCalculator<NArgs>& calculator() const noexcept {
    return calculator_;
  }

 private:
  const OffsetCalculator<NArgs> calculator_;
};

template <size_t NArgs>
std::unique_ptr<OffsetUtil> offset_util_cuda(
    const shape_list& iteration_shape,
    const std::array<OperandLayout, NArgs>& operands) {
  return std::make_unique<CUDAOffsetUtil<NArgs>>(iteration_shape, operands);
}
/// @endinternal

}  // namespace lmp::tensor::detail::cuda
