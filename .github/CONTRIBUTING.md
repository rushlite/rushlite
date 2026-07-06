# Contributing to Rushlite

Thank you for your interest in contributing to Rushlite!

All kinds of contributions are welcome: bug reports, documentation, tests, benchmarks, and code on either the Python or C++/CUDA side. If you're unsure whether an idea fits, [open an issue](https://github.com/rushlite/rushlite/issues) first and we can talk it through before you write any code.

Please note that this project has a [Code of Conduct](.github/CODE_OF_CONDUCT.md); by participating, you are expected to uphold it.

## Contents

- [Prerequisites](#prerequisites)
- [Python](#python)
  - [Setup](#setup)
  - [Running the tests](#running-the-tests)
  - [Formatting and linting](#formatting-and-linting)
- [C++/CUDA](#ccuda)
  - [Building](#building)
  - [Running the C++ tests](#running-the-c-tests)
  - [Formatting and linting](#formatting-and-linting-1)
- [Submitting changes](#submitting-changes)

## Prerequisites

- CMake 3.27+
- A C++20 compiler (GCC 11+, Clang 14+)
- Python 3.11–3.14
- CUDA Toolkit 12.x — **optional**; everything builds and the test suite runs on CPU-only machines (see [Running the tests](#running-the-tests))
- [`uv`](https://docs.astral.sh/uv/), our Python package manager

## Python

If you are interested in contributing to the Python side:

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate

uv lock
uv sync --extra cpu     # dev tools are included by default
uv pip install -e .
```

A few notes:

- `--extra cpu` installs the CPU build of PyTorch, which the test suite uses as a reference implementation. On a machine with an NVIDIA GPU, use `--extra cu128` instead (the two extras conflict, pick one).
- The editable install does **not** automatically rebuild the C++ code. If you touch anything under `csrc/`, you'll need to re-run `uv pip install -e .`.
- To build the Python package with CUDA enabled:

  ```bash
  uv pip install -e . -C cmake.define.LMP_ENABLE_CUDA=ON
  ```

### Running the tests

To run the full test suite:

```bash
uv run pytest tests/
```

Tests are parametrized over devices and determine at collection time whether the CUDA backend is usable. GPU coverage for pull requests is handled by CI, so it's fine to develop and test locally on CPU only.

Tests automatically run when you push to a branch. We use [Modal](https://modal.com/) for GPU testing on CI. 


### Formatting and linting

We use `black` for formatting and `ruff` for linting:

```bash
uv run black .
uv run ruff check --fix
```

CI checks both, and the `autofix.ci` bot will push formatting fixes to your PR automatically if you forget.

## C++/CUDA

The C++/CUDA core is a standalone library called **Lamp3** (`lmp::` namespace, `LMP_` CMake options), which lives in `csrc/`. The Python layer is a thin pybind11 wrapper around it, so most low level changes happen here.

### Building

We use CMake as our build system. On Debian/Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake
```

Configure and build (CPU-only, with tests):

```bash
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_COLOR_DIAGNOSTICS=ON \
    -DLMP_ENABLE_CUDA=OFF \
    -DLMP_ENABLE_TEST=ON \
    -DLMP_ENABLE_BENCH=OFF \
    -DLMP_ENABLE_COVERAGE=OFF

cmake --build build --parallel
```

- Set `-DLMP_ENABLE_CUDA=ON` to build the CUDA backend (requires CUDA
  Toolkit 12.x). 
- `LMP_ENABLE_BENCH=ON` builds the Google Benchmark suites in `benchmarks/`.

We recommend adding `-G Ninja` to the configure step to speed up builds
(`sudo apt-get install ninja-build`).

### Running the C++ tests

The test binaries land in `build/csrc/tests/`:

```bash
build/csrc/tests/tensor_tests
build/csrc/tests/autograd_tests
```

or equivalently, through CTest:

```bash
ctest --test-dir build
```

If OpenMP is available on your system, CMake picks it up automatically, which speeds up the CPU kernels by roughly 3x (the CPU kernels are otherwise very unoptimized :p).

### Formatting and linting

C++ formatting is clang-format (Google style, see `.clang-format`) and linting is clang-tidy (see `.clang-tidy`). Both are wired up as CMake targets:

```bash
cmake --build build --target format      
cmake --build build --target lint        # check only
cmake --build build --target lint-fix    
```

The `lint` targets need `compile_commands.json`, so make sure you configured with `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`. As with the Python linting and formatting, this is covered by the CI.

### Coverage (optional)

Configure with `-DLMP_ENABLE_COVERAGE=ON` and use the
`coverage` target to generate a combined C++/Python HTML coverage report in
`build/coverage/`.


## Submitting changes

1. Fork the repo and create a branch from `main`. Branch format: `<gh-username>/<desc-of-changes>`
2. Make your changes, and add tests where it makes sense.
3. Run the formatters/linters and the tests locally (CPU is fine).
4. Open a pull request with a short, descriptive title
5. Keep PRs focused: one feature or fix per PR is much easier to review.
