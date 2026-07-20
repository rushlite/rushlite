#include <cstdint>
#include <type_traits>

#include <algorithm>

#include "lamp3/common/macros.hpp"
#include "lamp3/tensor/cuda/matrix.cuh"
#include "lamp3/tensor/cuda/vec.cuh"
#include "lamp3/tensor/data_type.hpp"

namespace lmp::tensor::detail::cuda {

namespace {

// Vec4<T> / kHasVec4<T> live in cuda/vec.cuh (shared with the elementwise
// kernels).

// copied from here: https://github.com/clay-arras/sgemm-cublas-bench
// this could probably be faster with better locality and warp magic but meh
constexpr int kT = 8;
constexpr int kChD = 128;
constexpr int kChK = 16;

template <typename U, typename V, typename OutType>
__global__ void cudaMatmulVecKernel(const U* A, const V* B, OutType* C,
                                    size_t m, size_t n, size_t k,
                                    size_t batch_count,
                                    stride_t a_row_stride,
                                    stride_t a_contract_stride,
                                    stride_t b_contract_stride,
                                    stride_t b_col_stride,
                                    OffsetCalculator<2> batch_offsets,
                                    OutType alpha, OutType beta,
                                    bool b_vec_switch,
                                    bool output_vec_switch) {
  static_assert(kChK * kChD == kChD * kChD / kT);
  static_assert(kT % 4 == 0);

  __align__(4 * sizeof(U)) __shared__ U sA[kChD * kChK];
  __align__(4 * sizeof(V)) __shared__ V sB[kChD * kChK];

  __align__(4 * sizeof(OutType)) OutType tmp[kT * kT];
  __align__(4 * sizeof(U)) U sA_tmp[kT];
  __align__(4 * sizeof(V)) V sB_tmp[kT];

  for (size_t batch = blockIdx.z; batch < batch_count; batch += gridDim.z) {
    const offsets_t<2> bases = batch_offsets.get(batch);
    const stride_t a_base = bases[0];
    const stride_t b_base = bases[1];

    for (int i = 0; i < kT * kT; ++i) tmp[i] = OutType(0);

    for (int slideIdx = 0; slideIdx < static_cast<int>(k);
         slideIdx += kChK) {
      int Aidx_x = threadIdx.x * kT + blockIdx.x * kChK * kT;
      int Aidx_y = slideIdx + threadIdx.y;

      int Bidx_x = threadIdx.x + slideIdx;
      int Bidx_y = threadIdx.y * kT + blockIdx.y * kChK * kT;

#pragma unroll
      for (int i = 0; i < kT; i++) {
        if (Aidx_x + i < static_cast<int>(m) &&
            Aidx_y < static_cast<int>(k)) {
          const stride_t offset =
              a_base +
              (static_cast<stride_t>(Aidx_x + i) * a_row_stride) +
              (static_cast<stride_t>(Aidx_y) * a_contract_stride);
          sA[threadIdx.y * kChD + (threadIdx.x * kT + i)] = A[offset];
        } else {
          sA[threadIdx.y * kChD + (threadIdx.x * kT + i)] = U(0);
        }
      }

#pragma unroll
      for (int i = 0; i < kT; i += 4) {
        int sB_base = threadIdx.x * kChD + (threadIdx.y * kT + i);
        if constexpr (kHasVec4<V>) {
          using V4 = typename Vec4<V>::type;
          if (b_vec_switch && Bidx_x < static_cast<int>(k) &&
              Bidx_y + i + 3 < static_cast<int>(n)) {
            const stride_t b_offset =
                b_base +
                (static_cast<stride_t>(Bidx_x) * b_contract_stride) +
                (static_cast<stride_t>(Bidx_y + i) * b_col_stride);
            const V* address = &B[b_offset];
            const bool aligned =
                reinterpret_cast<std::uintptr_t>(address) % alignof(V4) == 0;
            if (aligned) {
              reinterpret_cast<V4*>(&sB[sB_base])[0] =
                  reinterpret_cast<const V4*>(address)[0];
              continue;
            }
          }
        }
        for (int kk = 0; kk < 4; kk++) {
          if (Bidx_x < static_cast<int>(k) &&
              Bidx_y + i + kk < static_cast<int>(n)) {
            const stride_t scalar_offset =
                b_base +
                (static_cast<stride_t>(Bidx_x) * b_contract_stride) +
                (static_cast<stride_t>(Bidx_y + i + kk) * b_col_stride);
            sB[sB_base + kk] = B[scalar_offset];
          } else {
            sB[sB_base + kk] = V(0);
          }
        }
      }

      __syncthreads();

      for (int kIdx = 0; kIdx < kChK; kIdx++) {
        if constexpr (kHasVec4<U>) {
          using U4 = typename Vec4<U>::type;
#pragma unroll
          for (int r = 0; r < kT / 4; r++) {
            reinterpret_cast<U4*>(sA_tmp)[r] = reinterpret_cast<const U4*>(
                &sA[kIdx * kChD + (threadIdx.x * kT)])[r];
          }
        } else {
#pragma unroll
          for (int r = 0; r < kT; r++) {
            sA_tmp[r] = sA[kIdx * kChD + (threadIdx.x * kT) + r];
          }
        }

        if constexpr (kHasVec4<V>) {
          using V4 = typename Vec4<V>::type;
#pragma unroll
          for (int r = 0; r < kT / 4; r++) {
            reinterpret_cast<V4*>(sB_tmp)[r] = reinterpret_cast<const V4*>(
                &sB[kIdx * kChD + (threadIdx.y * kT)])[r];
          }
        } else {
#pragma unroll
          for (int r = 0; r < kT; r++) {
            sB_tmp[r] = sB[kIdx * kChD + (threadIdx.y * kT) + r];
          }
        }

#pragma unroll
        for (int i = 0; i < kT; i++) {
          for (int j = 0; j < kT; j++) {
            tmp[i * kT + j] += static_cast<OutType>(sA_tmp[i]) *
                               static_cast<OutType>(sB_tmp[j]);
          }
        }
      }
      __syncthreads();
    }

    const size_t c_base = batch * m * n;
    for (int i = 0; i < kT; i++) {
      int resIdx_x =
          blockIdx.x * blockDim.x * kT + threadIdx.x * kT + i;
      for (int j = 0; j < kT; j += 4) {
        int resIdx_y =
            blockIdx.y * blockDim.y * kT + threadIdx.y * kT + j;

        if constexpr (kHasVec4<OutType>) {
          using O4 = typename Vec4<OutType>::type;
          if (output_vec_switch && resIdx_x < static_cast<int>(m) &&
              resIdx_y + 3 < static_cast<int>(n)) {
            OutType* cptr = &C[c_base + (resIdx_x * n) + resIdx_y];
            O4 c = reinterpret_cast<O4*>(cptr)[0];
            c.x = alpha * tmp[i * kT + j + 0] + beta * c.x;
            c.y = alpha * tmp[i * kT + j + 1] + beta * c.y;
            c.z = alpha * tmp[i * kT + j + 2] + beta * c.z;
            c.w = alpha * tmp[i * kT + j + 3] + beta * c.w;
            reinterpret_cast<O4*>(cptr)[0] = c;
            continue;
          }
        }
        for (int kk = 0; kk < 4; kk++) {
          if (resIdx_x < static_cast<int>(m) &&
              resIdx_y + kk < static_cast<int>(n)) {
            const size_t c_offset =
                c_base + (resIdx_x * n) + resIdx_y + kk;
            C[c_offset] =
                alpha * tmp[i * kT + j + kk] + beta * C[c_offset];
          }
        }
      }
    }
    __syncthreads();
  }
}

template <typename U, typename V, typename OutType>
__global__ void cudaMatmulKernel(const U* A, const V* B, OutType* C, size_t m,
                                 size_t n, size_t k, size_t batch_count,
                                 stride_t a_row_stride,
                                 stride_t a_contract_stride,
                                 stride_t b_contract_stride,
                                 stride_t b_col_stride,
                                 OffsetCalculator<2> batch_offsets) {
  for (size_t batch = blockIdx.z; batch < batch_count; batch += gridDim.z) {
    const offsets_t<2> bases = batch_offsets.get(batch);
    for (size_t i = (blockIdx.x * blockDim.x) + threadIdx.x; i < m;
         i += gridDim.x * blockDim.x) {
      for (size_t j = (blockIdx.y * blockDim.y) + threadIdx.y; j < n;
           j += gridDim.y * blockDim.y) {
        OutType sum = 0;
        for (size_t t = 0; t < k; t++) {
          const stride_t a_offset =
              bases[0] + (static_cast<stride_t>(i) * a_row_stride) +
              (static_cast<stride_t>(t) * a_contract_stride);
          const stride_t b_offset =
              bases[1] + (static_cast<stride_t>(t) * b_contract_stride) +
              (static_cast<stride_t>(j) * b_col_stride);
          sum += static_cast<OutType>(A[a_offset]) *
                 static_cast<OutType>(B[b_offset]);
        }
        C[(batch * m * n) + (i * n) + j] = sum;
      }
    }
  }
}

template <typename T>
__global__ void cudaTransposeKernel(const T* in, T* out, size_t m, size_t n) {
  for (size_t i = (blockIdx.x * blockDim.x) + threadIdx.x; i < m;
       i += gridDim.x * blockDim.x) {
    for (size_t j = (blockIdx.y * blockDim.y) + threadIdx.y; j < n;
         j += gridDim.y * blockDim.y) {
      out[(j * m) + i] = in[(i * n) + j];
    }
  }
}

}  // namespace

template <typename U, typename V, typename OutType>
void cudaMatMul(const U* A, const V* B, OutType* C, size_t m, size_t n,
                size_t k, size_t batch_count, stride_t a_row_stride,
                stride_t a_contract_stride, stride_t b_contract_stride,
                stride_t b_col_stride, OffsetCalculator<2> batch_offsets) {
  if (batch_count == 0 || m == 0 || n == 0) return;

  constexpr size_t kTileDim = kChK * kT;
  constexpr size_t kMinK = kChK;

  int device = 0;
  int max_grid_z = 1;
  LMP_CUDA_CHECK(cudaGetDevice(&device));
  LMP_CUDA_CHECK(
      cudaDeviceGetAttribute(&max_grid_z, cudaDevAttrMaxGridDimZ, device));
  const size_t grid_z =
      std::min(batch_count, static_cast<size_t>(max_grid_z));

  if (m >= kTileDim && n >= kTileDim && k > kMinK) {
    LMP_CUDA_CHECK(
        cudaMemset(C, 0, batch_count * m * n * sizeof(OutType)));

    const bool output_vec_switch = (n % 4 == 0);
    const bool b_vec_switch = output_vec_switch && b_col_stride == 1;
    dim3 block(kChK, kChK);
    dim3 grid((m + kTileDim - 1) / kTileDim,
              (n + kTileDim - 1) / kTileDim, grid_z);
    cudaMatmulVecKernel<U, V, OutType>
        <<<grid, block>>>(
            A, B, C, m, n, k, batch_count, a_row_stride, a_contract_stride,
            b_contract_stride, b_col_stride, batch_offsets, OutType(1),
            OutType(0), b_vec_switch, output_vec_switch);
  } else {
    dim3 threads(16, 16);
    dim3 blocks((m + threads.x - 1) / threads.x,
                (n + threads.y - 1) / threads.y, grid_z);
    cudaMatmulKernel<U, V, OutType><<<blocks, threads>>>(
        A, B, C, m, n, k, batch_count, a_row_stride, a_contract_stride,
        b_contract_stride, b_col_stride, batch_offsets);
  }
  LMP_CUDA_INTERNAL_ASSERT(cudaGetLastError())
      << "cudaMatMul: kernel launch failed.";
}

template <typename T>
void cudaTranspose(const T* in, T* out, size_t m, size_t n) {
  dim3 threads(16, 16);
  dim3 blocks((m + threads.x - 1) / threads.x, (n + threads.y - 1) / threads.y);
  cudaTransposeKernel<T><<<blocks, threads>>>(in, out, m, n);
}

#define INSTANTIATE_MATMUL(arg1_type, arg2_type, out_type)                 \
  template void cudaMatMul<arg1_type, arg2_type, out_type>(                \
      const arg1_type*, const arg2_type*, out_type*, size_t, size_t,       \
      size_t, size_t, stride_t, stride_t, stride_t, stride_t,              \
      OffsetCalculator<2>);
#define INSTANTIATE_TRANSPOSE(type) \
  template void cudaTranspose<type>(const type*, type*, size_t, size_t);

LMP_FOR_EACH_CARTESIAN_PRODUCT(INSTANTIATE_MATMUL, LMP_LIST_TYPES,
                               LMP_LIST_TYPES, LMP_LIST_TYPES);
LMP_FOR_EACH_CARTESIAN_PRODUCT(INSTANTIATE_TRANSPOSE, LMP_LIST_TYPES);

#undef INSTANTIATE_MATMUL
#undef INSTANTIATE_TRANSPOSE

}  // namespace lmp::tensor::detail::cuda
