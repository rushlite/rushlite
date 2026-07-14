#pragma once

#include <benchmark/benchmark.h>

#include <array>
#include <chrono>
#include <functional>
#include <string>

#include "lamp3/lamp3.hpp"

#ifdef LMP_ENABLE_CUDA
#include <cuda_runtime_api.h>
#endif

// Benchmark harness.
//
// Measures synchronized end-to-end per-op time using the same protocol as
// PyTorch's operator_benchmark framework (benchmarks/torch/all_tests.py), so
// the two CSVs are directly comparable:
//
//  - Inputs (and, for backward, the graph) are built once, outside the timed
//    region.
//  - Each timed iteration runs a batch of kOpsPerRound ops followed by ONE
//    device synchronize, all inside the timed region; the reported time is
//    elapsed / kOpsPerRound. Batching amortizes the fixed sync latency the
//    same way operator_benchmark's inner loop does. Without the in-loop sync,
//    CUDA benchmarks would only time host-side kernel enqueue and return
//    before the GPU has done the work.
//  - Backward is called repeatedly on the retained graph with no reduction:
//    Lamp3's backward() seeds ones_like(out) internally, so the PyTorch side
//    must match with `out.backward(torch.ones_like(out), retain_graph=True)`.
//    Gradients accumulate across calls on both sides (no zero_grad).

template <size_t N>
using OperatorFunction = std::function<lmp::autograd::Variable(
    const std::array<lmp::autograd::Variable, N>&)>;
template <size_t N>
using InitializerFunction =
    std::function<std::array<lmp::autograd::Variable, N>(bool)>;

const size_t kOpsPerRound = 100;
// Accumulated manual time per benchmark. The manual time is per-op (round
// elapsed / kOpsPerRound), so each benchmark's wall time is roughly
// kMinTimeSeconds * kOpsPerRound.
const double kMinTimeSeconds = 0.005;

inline void device_sync(bool is_cuda) {
#ifdef LMP_ENABLE_CUDA
  if (is_cuda) {
    cudaDeviceSynchronize();
  }
#else
  (void)is_cuda;
#endif
}

inline void run_timed_loop(benchmark::State& state,
                           const std::function<void()>& run_op, bool is_cuda) {
  for (auto _ : state) {
    auto start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < kOpsPerRound; ++i) {
      run_op();
    }
    device_sync(is_cuda);
    auto end = std::chrono::steady_clock::now();
    state.SetIterationTime(std::chrono::duration<double>(end - start).count() /
                           static_cast<double>(kOpsPerRound));
  }
}

template <size_t N>
void register_forward(const std::string& name, const OperatorFunction<N>& op_fn,
                      const InitializerFunction<N>& init_fn, bool is_cuda) {
  benchmark::RegisterBenchmark(
      name + "Forward",
      [op_fn, init_fn, is_cuda](benchmark::State& state) {
        std::array<lmp::autograd::Variable, N> inputs = init_fn(false);
        auto run_op = [op_fn, inputs]() {
          lmp::autograd::Variable result = op_fn(inputs);
          benchmark::DoNotOptimize(result);
        };
        run_op();  // warm-up: prime allocations and caches
        device_sync(is_cuda);
        run_timed_loop(state, run_op, is_cuda);
      })
      ->UseManualTime()
      ->MinTime(kMinTimeSeconds);
}

template <size_t N>
void register_backward(const std::string& name,
                       const OperatorFunction<N>& op_fn,
                       const InitializerFunction<N>& init_fn, bool is_cuda) {
  benchmark::RegisterBenchmark(
      name + "Backward",
      [op_fn, init_fn, is_cuda](benchmark::State& state) {
        std::array<lmp::autograd::Variable, N> inputs = init_fn(true);
        lmp::autograd::Variable out = op_fn(inputs);
        auto run_op = [out]() mutable { out.backward(); };
        run_op();  // warm-up: prime allocations and caches
        device_sync(is_cuda);
        run_timed_loop(state, run_op, is_cuda);
      })
      ->UseManualTime()
      ->MinTime(kMinTimeSeconds);
}
