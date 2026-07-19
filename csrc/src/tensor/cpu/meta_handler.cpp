#include "lamp3/tensor/cpu/meta_handler.hpp"

#include "lamp3/common/assert.hpp"
#include "lamp3/tensor/device_type.hpp"
#include "lamp3/tensor/infer_meta.hpp"

namespace lmp::tensor::detail {

template <>
UnaryMetaHandler::TensorMetaHandler(const TensorImpl* a) : inTens_({a}) {
  OpMeta m = infer_unary(a);
  outDtype_ = m.dtype;
  outSize_ = m.size;
  outShape_ = m.shape;
  expand_ = m.expand;
  LMP_DISPATCH_ALL_TYPES(outDtype_, [&] {
    using out_dtype_t = scalar_t;
    LMP_DISPATCH_ALL_TYPES(a->type(), [&] {
      using arg_dtype_t = scalar_t;
      Storage out_st(outSize_ * sizeof(out_dtype_t), a->device());
      outTen_ = std::make_unique<TensorImpl>(out_st, outShape_, outDtype_);
    });
  });
  if (!a->is_contiguous()) {
    outOffset_ = offset_util_stub_1()(
        a->device(), outShape_,
        std::array<OperandLayout, UnaryMetaHandler::kNumElem>{
            operand_layout(*a)});
  }
}

template <>
// NOLINTNEXTLINE(readability-function-size,google-readability-function-size)
BinaryMetaHandler::TensorMetaHandler(const TensorImpl* a, const TensorImpl* b)
    : inTens_({a, b}) {
  OpMeta m = infer_binary(a, b);
  outDtype_ = m.dtype;
  outSize_ = m.size;
  outShape_ = m.shape;
  expand_ = m.expand;

  LMP_DISPATCH_ALL_TYPES(outDtype_, [&] {
    using out_dtype_t = scalar_t;
    LMP_DISPATCH_ALL_TYPES(a->type(), [&] {
      using arg1_dtype_t = scalar_t;
      LMP_DISPATCH_ALL_TYPES(b->type(), [&] {
        using arg2_dtype_t = scalar_t;
        Storage out_st(outSize_ * sizeof(out_dtype_t), a->device());
        outTen_ = std::make_unique<TensorImpl>(out_st, outShape_, outDtype_);
      });
    });
  });
  outOffset_ = offset_util_stub_2()(
      a->device(), outShape_,
      std::array<OperandLayout, BinaryMetaHandler::kNumElem>{
          operand_layout(*a), operand_layout(*b)});
}

template <>
ReductMetaHandler::TensorMetaHandler(const TensorImpl* a, size_t axis)
    : inTens_({a}) {
  OpMeta m = infer_reduct(a, axis);
  outDtype_ = m.dtype;
  outSize_ = m.size;
  outShape_ = m.shape;
  expand_ = m.expand;
  LMP_DISPATCH_ALL_TYPES(outDtype_, [&] {
    using out_dtype_t = scalar_t;
    LMP_DISPATCH_ALL_TYPES(a->type(), [&] {
      using arg_dtype_t = scalar_t;
      Storage out_st(outSize_ * sizeof(out_dtype_t), a->device());
      outTen_ = std::make_unique<TensorImpl>(out_st, outShape_, outDtype_);
    });
  });
  if (!a->is_contiguous()) {
    outOffset_ = offset_util_stub_1()(
        a->device(), outShape_,
        std::array<OperandLayout, ReductMetaHandler::kNumElem>{
            operand_layout(*a)});
  }
}

}  // namespace lmp::tensor::detail
