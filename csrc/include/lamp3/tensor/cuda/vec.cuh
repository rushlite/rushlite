#pragma once

#include <cuda_runtime.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace lmp::tensor::detail::cuda {

/// @internal
/**
 * @brief Single source of truth for width-4 vectorization.
 * @details Each supported dtype maps to a 4-wide vector type so a single thread
 * can move a 16-byte-style packet (float4/int4/...) per load/store. This cuts
 * the number of memory instructions ~4x on the bandwidth-bound elementwise
 * kernels and is also reused by the matmul kernel for its shared-memory loads.
 */

/// bool has no builtin 4-vector; mirror the {x,y,z,w} layout of the CUDA types.
struct alignas(4) bool4 {
  bool x, y, z, w;
};

template <typename T>
struct Vec4;
template <>
struct Vec4<bool> {
  using type = bool4;
};
template <>
struct Vec4<int16_t> {
  using type = short4;
};
template <>
struct Vec4<int> {
  using type = int4;
};
template <>
struct Vec4<int64_t> {
  using type = longlong4;
};
template <>
struct Vec4<float> {
  using type = float4;
};
template <>
struct Vec4<double> {
  using type = double4;
};

template <typename T>
using vec4_t = typename Vec4<T>::type;

/// matmul opts bool out of vectorized shared-memory loads.
template <typename T>
inline constexpr bool kHasVec4 = !std::is_same_v<T, bool>;

namespace internal {

constexpr int kVecWidth = 4;
constexpr size_t kElemwiseThreads = 256;
constexpr size_t kElemwiseMaxBlocks = 65535;

/// Grid sizing for a 1D grid-stride elementwise launch.
inline size_t elemwise_blocks(size_t work_items) {
  size_t blocks = (work_items + kElemwiseThreads - 1) / kElemwiseThreads;
  return std::max(std::min(blocks, kElemwiseMaxBlocks), size_t{1});
}

/// Whether `p` is safe to reinterpret as a packet with the given alignment.
inline bool is_aligned(const void* p, size_t alignment) {
  return (reinterpret_cast<uintptr_t>(p) & (alignment - 1)) == 0;
}

/// Apply a unary functor to each lane of a packet.
template <typename T, typename OpFn>
__device__ inline vec4_t<T> apply_unary4(const vec4_t<T>& a, OpFn& fn) {
  vec4_t<T> r;
  r.x = fn(a.x);
  r.y = fn(a.y);
  r.z = fn(a.z);
  r.w = fn(a.w);
  return r;
}

/// Apply a binary functor to each lane of two packets.
template <typename T, typename OpFn>
__device__ inline vec4_t<T> apply_binary4(const vec4_t<T>& a,
                                          const vec4_t<T>& b, OpFn& fn) {
  vec4_t<T> r;
  r.x = fn(a.x, b.x);
  r.y = fn(a.y, b.y);
  r.z = fn(a.z, b.z);
  r.w = fn(a.w, b.w);
  return r;
}

}  // namespace internal
/// @endinternal

}  // namespace lmp::tensor::detail::cuda
