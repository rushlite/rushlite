#include <cuda_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include "lamp3/tensor/cuda/memory.cuh"

namespace {

using lmp::tensor::detail::DataPtr;
using lmp::tensor::detail::cuda::empty_cuda;

void check_cuda(cudaError_t error, const char* operation) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(error));
  }
}

#define CHECK(expression)                                                \
  do {                                                                   \
    if (!(expression)) {                                                 \
      std::cerr << __func__ << ':' << __LINE__                           \
                << ": check failed: " #expression << '\n';              \
      std::exit(1);                                                      \
    }                                                                    \
  } while (false)

__global__ void write_value(int* values, std::size_t size, int value) {
  for (std::size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
       index < size; index += gridDim.x * blockDim.x) {
    values[index] = value;
  }
}

__global__ void add_value(int* values, std::size_t size, int value) {
  for (std::size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
       index < size; index += gridDim.x * blockDim.x) {
    values[index] += value;
  }
}

void test_address_reuse_and_stream_ordering() {
  constexpr std::size_t kElementCount = 1024 * 1024;
  constexpr std::size_t kByteSize = kElementCount * sizeof(int);

  DataPtr first = empty_cuda(kByteSize);
  void* first_address = first.data();
  write_value<<<256, 256>>>(static_cast<int*>(first.data()), kElementCount, 7);
  check_cuda(cudaGetLastError(), "first test kernel launch");

  first = DataPtr{};
  DataPtr second = empty_cuda(kByteSize);
  CHECK(second.data() == first_address);

  add_value<<<256, 256>>>(static_cast<int*>(second.data()), kElementCount, 5);
  check_cuda(cudaGetLastError(), "second test kernel launch");

  int result = 0;
  check_cuda(cudaMemcpy(&result, second.data(), sizeof(result),
                        cudaMemcpyDeviceToHost),
             "test result copy");
  CHECK(result == 12);
}

bool test_deallocation_uses_the_owning_device() {
  int device_count = 0;
  check_cuda(cudaGetDeviceCount(&device_count), "get device count");
  if (device_count < 2) {
    std::cout << "SKIP owning_device (requires two CUDA devices)\n";
    return false;
  }

  check_cuda(cudaSetDevice(0), "select device zero");
  DataPtr allocation = empty_cuda(4096);
  void* first_address = allocation.data();

  check_cuda(cudaSetDevice(1), "select device one");
  allocation = DataPtr{};

  int current_device = -1;
  check_cuda(cudaGetDevice(&current_device), "get current device");
  CHECK(current_device == 1);

  check_cuda(cudaSetDevice(0), "restore device zero");
  DataPtr reused = empty_cuda(4096);
  CHECK(reused.data() == first_address);
  return true;
}

}  // namespace

int main() {
  test_address_reuse_and_stream_ordering();
  std::cout << "PASS address_reuse_and_stream_ordering\n";
  if (test_deallocation_uses_the_owning_device()) {
    std::cout << "PASS owning_device\n";
  }
}
