#include <functional>

#include "lamp3/lamp3.hpp"
#include "lamp3/nets/layers/linear.hpp"
#include "lamp3/nets/parameter.hpp"

// NOLINTNEXTLINE(bugprone-exception-escape)
int main() {
  auto input = lmp::tensor::Tensor(
      std::vector<float>{1, -4, 2, 7, 9, 10, 4, -2, 27, 0, 0, 24, 2,
                         2, 10, 1, 4, 0, 2,  1, 1,  4,  0, 2, 1},
      std::vector<size_t>{5, 5}, lmp::DeviceType::CUDA);

  std::cout << input << std::endl;
}