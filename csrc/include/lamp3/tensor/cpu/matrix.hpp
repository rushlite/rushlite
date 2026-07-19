#pragma once

#include <cstddef>

#include "lamp3/tensor/offset_util.hpp"

namespace lmp::tensor::detail::cpu {

/// @internal
template <typename U, typename V, typename OutType>
void cpuMatMul(const U* A, const V* B, OutType* C, size_t m, size_t n,
               size_t k, size_t batch_count, stride_t a_row_stride,
               stride_t a_contract_stride, stride_t b_contract_stride,
               stride_t b_col_stride,
               const OffsetCalculator<2>& batch_offsets);

template <typename T>
void cpuTranspose(const T* in, T* out, size_t m, size_t n);
/// @endinternal

}  // namespace lmp::tensor::detail::cpu
