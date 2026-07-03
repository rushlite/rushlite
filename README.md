<div align="left">
  <img src="https://github.com/user-attachments/assets/52f467bf-bc40-4e01-8389-358d74777731" alt="neural_bulb_svg (3)" width="600">
</div>

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/rushlite/rushlite) <!-- Placeholder -->
[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/rushlite/rushlite/blob/main/LICENSE) <!-- Placeholder -->
[![Version](https://img.shields.io/badge/version-0.0.0-blue)](https://github.com/rushlite/rushlite) <!-- Placeholder -->

# Rushlite

Rushlite is a lightweight machine learning library built from scratch in C++/CUDA. It provides:
- efficient tensor operations on GPU hardware, with automatic differentiation
- reusable modules for building deep neural networks

Highlights:
- **Intuitive to use**: the API closely mirrors PyTorch
- **Fast backpropagation**: on single-op CUDA benchmarks, the backward pass is up to 3.5x faster than PyTorch ([benchmarks](benchmarks/README.md))
- **Kernel fusion**: the backend (Lamp3) is one of the only machine learning libraries that exposes kernel fusion directly to C++ users
- **Written from scratch**: the C++/CUDA core has zero dependencies beyond the CUDA Toolkit, and the Python layer depends only on pybind11 (to bind the C++ code)

> [!WARNING]
> Rushlite is still in early development and updates may include breaking changes.

## Contents

- [Installation](#installation)
- [Examples](#examples)
  - [Kernel fusion](#kernel-fusion)
  - [Autograd](#autograd)
  - [A simple MLP](#a-simple-mlp)
- [Building from source](#building-from-source)
  - [Using Lamp3](#using-lamp3)
- [License](#license)
- [Contributing](#contributing)

## Installation

Create and activate a virtual environment, then install **Rushlite**:

```bash
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install rushlite
```

To install with CUDA acceleration, point pip at the wheels attached to a GitHub release:

```bash
$ pip install rushlite -f https://github.com/rushlite/rushlite/releases/expanded_assets/v0.1.0
```

Change the release tag to the latest version, or whichever one you want. Unfortunately, the only supported CUDA versions is CUDA 12 right now.

## Examples

### Kernel fusion

To fuse a group of operations, use the `capture_on` context manager or the `capture` decorator:

```py
import rushlite as rsl

a = rsl.rand([5, 5])
b = rsl.rand([5, 1])

with rsl.capture_on():
    c = a * a + b  # recorded lazily and compiled into a fused kernel
```

Note that `capture_on` limits where operations are *captured*, not where they are *realized*: a lazily captured graph can be realized anywhere, including outside the `with` block, the first time its values are needed.

<!-- TODO: create a guide for realize(), when data() is called -->

### Autograd

In Rushlite, gradient-tracked tensors are called `Variable`:

```py
import rushlite as rsl

a = rsl.rand([5, 5], requires_grad=True)
b = rsl.rand([5, 1], requires_grad=True)

c = a + b  # broadcasts
c.backward()
```

`a.grad` and `b.grad` now hold the accumulated gradients. Unlike PyTorch, a `Variable` does *not* need to be a scalar to call `backward()` on it.

### A simple MLP

Putting it together: `rushlite.nets` provides reusable modules for building neural nets, and you can fuse the forward pass with `capture`.

```py
import rushlite as rsl
from rushlite.nets import layers

model = layers.Sequential([
    layers.Linear(784, 128),
    layers.ReLU(),
    layers.Linear(128, 10),
    layers.Softmax(dim=-1),
])

x = rsl.rand([32, 784])  
y = rsl.rand([32, 10])  

@rsl.capture()  
def forward(x):
    return model(x)

probs = forward(x)
loss = rsl.sum(rsl.sum(-y * rsl.log(probs + 1e-3), 0), 1)  # cross-entropy
loss.backward()

for name, param in model.named_parameters():
    print(name, param.grad)
```


## Building from source

You will need:

- CMake 3.27+
- A C++20 compiler (most common GCC 11+, Clang 14+)
- Python 3.11–3.14
- CUDA Toolkit 12.x (optional, for GPU support)

We use [`uv`](https://docs.astral.sh/uv/) as the package manager below, but any one works:

```sh
$ git clone https://github.com/rushlite/rushlite.git
$ cd rushlite
$ uv sync
$ uv pip install -e .
```

This builds the C++ core and the Python bindings from scratch. To enable CUDA:

```sh
$ uv pip install -e . -C cmake.define.LMP_ENABLE_CUDA=ON
```

### Using Lamp3 

Lamp3 is Rushlite's C++ backend (the name comes from lamp++ → lamppp → lamp3), and it's a full standalone library -- the Python layer is a thin pybind11 wrapper around it. 

To build it without the Python bindings:

```sh
$ cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLMP_ENABLE_CUDA=ON
$ cmake --build build
```

The easiest way to consume it from your own CMake project is `add_subdirectory` (or `FetchContent`) and linking the umbrella target:

```cmake
add_subdirectory(rushlite)
target_link_libraries(my_app PRIVATE lamp3)
```

A minimal program:

```cpp
#include "lamp3/lamp3.hpp"
#include <iostream>

int main() {
  lmp::Tensor data({1.0F, 2.0F, 3.0F, 4.0F}, {2, 2},
                   lmp::DeviceType::CPU, lmp::DataType::Float32);
  lmp::Variable a(data, /*requires_grad=*/true);

  lmp::Variable loss = lmp::sum(a * a, 0);
  loss.backward();

  std::cout << a.grad() << std::endl;  // 2a
}
```

## License

Rushlite is licensed under the [MIT License](LICENSE).

## Contributing

Contributions are welcome.
A contributing guide is coming soon! In the meantime, feel free to [open an issue](https://github.com/rushlite/rushlite/issues) for bugs, questions, or feature requests.
