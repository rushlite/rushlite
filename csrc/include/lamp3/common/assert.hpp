#pragma once

#include <cassert>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include "stream.hpp"

#ifdef LMP_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace lmp {
namespace detail {

template <typename Derived>
class BaseStream {
  BaseStream() = default;

 public:
  template <class T>
  Derived& operator<<(T&& v) {
    os_ << std::forward<T>(v);
    return static_cast<Derived&>(*this);
  }

 protected:
  std::ostringstream os_;
  friend Derived;
};

class CheckStream : public BaseStream<CheckStream> {
 public:
  CheckStream(const char* file, int line, const char* func, const char* expr) {
    os_ << "Lamp3: Runtime error thrown at " << file << ':' << line << " in "
        << func << ". CHECK(" << expr << ") failed: ";
  }

  [[noreturn]] void trigger() const {
    std::cerr << os_.str() << std::endl;
    throw std::runtime_error(os_.str());
  }
};

#ifdef LMP_ENABLE_CUDA
class CudaAssertStream : public BaseStream<CudaAssertStream> {
 public:
  CudaAssertStream(const char* file, int line, const char* func,
                   cudaError_t err) {
    os_ << "Lamp3: CUDA error at " << file << ':' << line << " in " << func
        << ". Error: " << cudaGetErrorString(err) << ". ";
  }

  [[noreturn]] void trigger() const {
    std::cerr << os_.str() << std::endl;
    std::terminate();
  }
};
#endif

#ifdef LMP_DEBUG

class AssertStream : public BaseStream<AssertStream> {
 public:
  AssertStream(const char* file, int line, const char* func, const char* expr) {
    os_ << "Lamp3: Internal assertion failure at " << file << ':' << line
        << " in " << func << ". ASSERT(" << expr << ") failed: ";
  }

  [[noreturn]] void trigger() const {
    std::cerr << os_.str() << std::endl;
    std::terminate();
  }
};

#endif

struct Voidify {
  template <class T>
  void operator&(T&& stream) const {
    stream.trigger();
  }
};

}  // namespace detail
}  // namespace lmp

// clang-format off
#define LMP_CHECK(cond) \
    (cond) ? (void)0 : ::lmp::detail::Voidify() & ::lmp::detail::CheckStream(__FILE__, __LINE__, __func__, #cond)  // NOLINT(bugprone-macro-parentheses)
// clang-format on

#ifdef LMP_ENABLE_CUDA
// TODO(clay-arras): Separate CUDA check and internal assertion failure handling.
#define LMP_CUDA_CHECK(call)                                          \
  if (const cudaError_t LMP_CUDA_ERROR = (call);                      \
      LMP_CUDA_ERROR == cudaSuccess)                                  \
    ;                                                                 \
  else                                                                \
    ::lmp::detail::Voidify() &                                        \
        ::lmp::detail::CudaAssertStream(__FILE__, __LINE__, __func__, \
                                        LMP_CUDA_ERROR)
#endif

#ifdef LMP_DEBUG

// clang-format off
#define LMP_INTERNAL_ASSERT(cond) \
    (cond) ? (void)0 : ::lmp::detail::Voidify() & ::lmp::detail::AssertStream(__FILE__, __LINE__, __func__, #cond)  // NOLINT(bugprone-macro-parentheses)
// clang-format on

#ifdef LMP_ENABLE_CUDA

namespace lmp::detail {

inline cudaError_t& cuda_assert_last_error() {
  static thread_local cudaError_t last{};
  return last;
}

template <class F>
inline bool cuda_assert_once(F&& f) {
  cuda_assert_last_error() = f();
  return cuda_assert_last_error() != cudaSuccess;
}

}  // namespace lmp::detail

#define LMP_CUDA_INTERNAL_ASSERT(call)                                \
  if (const cudaError_t LMP_CUDA_ERROR = (call);                      \
      LMP_CUDA_ERROR == cudaSuccess)                                  \
    ;                                                                 \
  else                                                                \
    ::lmp::detail::Voidify() &                                        \
        ::lmp::detail::CudaAssertStream(__FILE__, __LINE__, __func__, \
                                        LMP_CUDA_ERROR)

#endif

#define LMP_PRINT(fmt, ...)                                              \
  fprintf(stderr, "[%s:%d] %s: " fmt "\n", __FILE__, __LINE__, __func__, \
          ##__VA_ARGS__)

#else

namespace lmp::detail {
struct NullStream {
  template <typename T>
  NullStream& operator<<(T&&) {
    return *this;
  }
  void trigger() const {}
};

}  // namespace lmp::detail
#define LMP_INTERNAL_ASSERT(cond) \
  true ? (void)0 : ::lmp::detail::Voidify() & ::lmp::detail::NullStream()
#define LMP_CUDA_INTERNAL_ASSERT(call) \
  true ? (void)0 : ::lmp::detail::Voidify() & ::lmp::detail::NullStream()

#endif
