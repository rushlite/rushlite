#pragma once

#include <array>
#include <memory>

#include "lamp3/tensor/dispatch_stub.hpp"
#include "lamp3/tensor/offset_util.hpp"

namespace lmp::tensor::detail {

namespace cpu {

/// @internal
template <size_t NArgs>
class CPUOffsetUtil final : public OffsetUtil {
 public:
  CPUOffsetUtil(const shape_list& iteration_shape,
                const std::array<OperandLayout, NArgs>& operands)
      : OffsetUtil(NArgs),
        calculator_(make_offset_calculator(iteration_shape, operands)) {}

  offsets_t<NArgs> get(size_t logical_index) const noexcept {
    return calculator_.get(logical_index);
  }

  const OffsetCalculator<NArgs>& calculator() const noexcept {
    return calculator_;
  }

 private:
  const OffsetCalculator<NArgs> calculator_;
};
/// @endinternal

template <size_t NArgs>
std::unique_ptr<OffsetUtil> offset_util_cpu(
    const shape_list& iteration_shape,
    const std::array<OperandLayout, NArgs>& operands) {
  return std::make_unique<CPUOffsetUtil<NArgs>>(iteration_shape, operands);
}

}  // namespace cpu

LMP_DECLARE_DISPATCH(offset_util_fn<1>, offset_util_stub_1);
LMP_DECLARE_DISPATCH(offset_util_fn<2>, offset_util_stub_2);
LMP_DECLARE_DISPATCH(offset_util_fn<3>, offset_util_stub_3);

}  // namespace lmp::tensor::detail
