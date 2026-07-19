#pragma once

#include <memory>
#include <vector>
#include "lamp3/tensor/cpu/offset_util.hpp"
#include "lamp3/tensor/data_type.hpp"
#include "lamp3/tensor/tensor_impl.hpp"

namespace lmp::tensor::detail {

using tensor_list = std::vector<const lmp::tensor::TensorImpl*>;

template <typename... Args>
class TensorMetaHandler {
 public:
  static constexpr std::size_t kNumElem =
      (0 + ... + std::size_t{std::is_same_v<const TensorImpl*, Args>});
  explicit TensorMetaHandler(Args... args);

  TensorImpl& out() noexcept { return *outTen_; }
  tensor_list& in() noexcept { return inTens_; }
  const OffsetUtil* offset() const {
    LMP_INTERNAL_ASSERT(outOffset_ != nullptr)
        << "Offset utility was not constructed for this route";
    return outOffset_.get();
  }
  bool has_offset() const noexcept { return outOffset_ != nullptr; }
  bool expand() const noexcept { return expand_; }

 private:
  DataType outDtype_;
  size_t outSize_;
  std::vector<size_t> outShape_;

  bool expand_{false};
  std::unique_ptr<OffsetUtil> outOffset_;
  std::unique_ptr<TensorImpl> outTen_;
  tensor_list inTens_;
};

using UnaryMetaHandler = TensorMetaHandler<const TensorImpl*>;
using BinaryMetaHandler =
    TensorMetaHandler<const TensorImpl*, const TensorImpl*>;
using ReductMetaHandler = TensorMetaHandler<const TensorImpl*, size_t>;

}  // namespace lmp::tensor::detail
