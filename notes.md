<!-- START HERE:  -->

<div align="left">
  <img src="https://github.com/user-attachments/assets/52f467bf-bc40-4e01-8389-358d74777731" alt="neural_bulb_svg (3)" width="600">
</div>

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/rushlite/rushlite) <!-- Placeholder -->
[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/rushlite/rushlite/blob/main/LICENSE) <!-- Placeholder -->
[![Version](https://img.shields.io/badge/version-0.1.0-blue)](https://github.com/rushlite/rushlite) <!-- Placeholder -->

# Rushlite

Rushlite is a lightweight machine learning library built from scratch in C++/CUDA that provides the following features:
- efficient tensor operations on GPU hardware with automatic differentiation
- reusable modules for deep neural networks

Some highlighted features are:
- **Intuitive to use**: the API is very similar to Pytorch
- **Fast backpropogation**: the backward pass of single-op CUDA benchmarks are 3.5x faster than Pytorch
- **Kernel fusion**: the backend (Lamp3) is one of the only machine learning libraries that support direct kernel fusion for C++ users
- **Written from scratch**: there are no dependencies for the C++ / CUDA code other than the out of the box CUDA Toolkit, and the only dependency for the Python code is Pybind11 (to bind the C++ code).

> [!WARNING]
> Rushlite is still in early development and updates may include breaking changes



- Installation ... link the actual headings
- Examples
  - Kernel fusion
  - Autograd
  - Simple MLP
- Build from source
  - Using the C++ library
- Licensing
- Feedback and support



## Installation
Create and activate a virtual environment, then install **Rushlite**
```bash
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install rushlite
```

To install with CUDA acceleration, do:
```bash
$ pip install rushlite -f https://github.com/rushlite/rushlite/releases/tag/v2.12.1/expanded-assets
```

Change the release tag to the latest version, or whichever one you want. Unfortunately, the only supported CUDA versions is CUDA 12 right now.


## Examples

### Kernel Fusion
To fuse a group of operations, use the `capture_on` context manager or the `capture` decorator.
```py
import rushlite as rsl

a = rsl.rand([5, 5])
b = rsl.rand([5, 1])

with rsl.capture_on():
  c = a * a + b  # this will capture the tensors lazily
```
A small note: the tag limits *capture*, not *realization*. A lazily captured tensor graph can be realized anywhere.


### Autograd
In Rushlite, gradient-tracked tensors are called `Variable`
```py
import rushlite as rsl

a = rsl.rand([5, 5])
b = rsl.rand([5, 1])

c = a + b
c.backward()
```
Now, `a.grad` and `b.grad` contain the accumulated gradients. Note that a Variable *doesn't* need to be a scalar value to call backward() on. 


### Simple MLP
Now lets put it all together to build something slightly more complex! Rushlite defines reusable modules you can use to build neural nets!
... example. use backwards + kernel fusion maybe.


## Building from source
To build the library from source, make sure you have the following installed: 
- CMake
- C++ ...what else do we need? 

We will use `uv` as the package manager, but you can use anything

```sh
$ git clone https://github.com/rushlite/rushlite.git 
$ cd rushlite
$ uv lock && uv sync 
$ LMP_ENABLE_CUDA=OFF uv pip install -e .
```
This will build the library from scratch. To enable CUDA, set the flag to ON.

### Using Lamp3

Lamp3 is the backend for Rushlite (the name comes from lamp++ --> lamppp --> lamp3). It contains most of the business logic for Rushlite. ...???

In order to use the C++ library, you can do the following:
```sh
$ cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLMP_ENABLE_CUDA=ON
$ cmake --build build
```

... add more details on how to run code or set up CMake 


## Licensing
Rushlite is licensed under MIT License. By using, distributing, or contributing to this project, you agree the terms and conditions of this license.


## Contributing
Coming soon!