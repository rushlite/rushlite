#include "lamp3/tensor/cpu/matrix.hpp"

#include "lamp3/common/macros.hpp"
#include "lamp3/tensor/data_type.hpp"

namespace lmp::tensor::detail::cpu {

template <typename U, typename V, typename OutType>
void cpuMatmulKernel(const U* A, const V* B, OutType* C, size_t m, size_t n,
                     size_t k, size_t batch, size_t i, size_t j,
                     stride_t a_base, stride_t b_base, stride_t a_row_stride,
                     stride_t a_contract_stride, stride_t b_contract_stride,
                     stride_t b_col_stride) {
  OutType sum = 0;
  for (size_t t = 0; t < k; ++t) {
    const stride_t a_offset =
        a_base + (static_cast<stride_t>(i) * a_row_stride) +
        (static_cast<stride_t>(t) * a_contract_stride);
    const stride_t b_offset =
        b_base + (static_cast<stride_t>(t) * b_contract_stride) +
        (static_cast<stride_t>(j) * b_col_stride);
    sum += static_cast<OutType>(A[a_offset]) *
           static_cast<OutType>(B[b_offset]);
  }
  C[(batch * m * n) + (i * n) + j] = sum;
}

template <typename T>
void cpuTransposeKernel(const T* in, T* out, size_t m, size_t n, size_t i,
                        size_t j) {
  out[(j * m) + i] = in[(i * n) + j];
}

template <typename U, typename V, typename OutType>
void cpuMatMul(const U* A, const V* B, OutType* C, size_t m, size_t n,
               size_t k, size_t batch_count, stride_t a_row_stride,
               stride_t a_contract_stride, stride_t b_contract_stride,
               stride_t b_col_stride,
               const OffsetCalculator<2>& batch_offsets) {
#pragma omp parallel for collapse(2) schedule(static)
  for (size_t batch = 0; batch < batch_count; ++batch) {
    for (size_t i = 0; i < m; ++i) {
      const offsets_t<2> bases = batch_offsets.get(batch);
      for (size_t j = 0; j < n; ++j) {
        cpuMatmulKernel<U, V, OutType>(
            A, B, C, m, n, k, batch, i, j, bases[0], bases[1], a_row_stride,
            a_contract_stride, b_contract_stride, b_col_stride);
      }
    }
  }
}

template <typename T>
void cpuTranspose(const T* in, T* out, size_t m, size_t n) {
#pragma omp parallel for collapse(2) schedule(static)
  for (size_t i = 0; i < m; i++)
    for (size_t j = 0; j < n; j++) cpuTransposeKernel<T>(in, out, m, n, i, j);
}

#define INSTANTIATE_MATMUL(arg1_type, arg2_type, out_type)              \
  template void cpuMatMul<arg1_type, arg2_type, out_type>(              \
      const arg1_type*, const arg2_type*, out_type(*), size_t, size_t,  \
      size_t, size_t, stride_t, stride_t, stride_t, stride_t,           \
      const OffsetCalculator<2>&);
#define INSTANTIATE_TRANSPOSE(type) \
  template void cpuTranspose<type>(const type*, type(*), size_t, size_t);

LMP_FOR_EACH_CARTESIAN_PRODUCT(INSTANTIATE_MATMUL, LMP_LIST_TYPES,
                               LMP_LIST_TYPES, LMP_LIST_TYPES);
LMP_FOR_EACH_CARTESIAN_PRODUCT(INSTANTIATE_TRANSPOSE, LMP_LIST_TYPES);

#undef INSTANTIATE_MATMUL
#undef INSTANTIATE_TRANSPOSE

}  // namespace lmp::tensor::detail::cpu
