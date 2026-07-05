# Contributing to Rushlite

Thank you for your interest in contributing to Rushlite!

All kinds of contributions are welcome: bug reports, documentation, tests,
benchmarks, and code on either the Python or C++/CUDA side. If you're unsure
whether an idea fits, [open an issue](https://github.com/rushlite/rushlite/issues)
first and we can talk it through before you write any code.

Please note that this project has a [Code of Conduct](.github/CODE_OF_CONDUCT.md);
by participating, you are expected to uphold it.

## Contents

- [Prerequisites](#prerequisites)
- [Python](#python)
  - [Setup](#setup)
  - [Running the tests](#running-the-tests)
  - [Test layout](#test-layout)
  - [Formatting and linting](#formatting-and-linting)
- [C++/CUDA](#ccuda)
  - [Building](#building)
  - [Running the C++ tests](#running-the-c-tests)
  - [Formatting and linting](#formatting-and-linting-1)
- [Continuous integration](#continuous-integration)
- [Repo organization](#repo-organization)
- [Submitting changes](#submitting-changes)

## Prerequisites

- CMake 3.27+
- A C++20 compiler (GCC 11+, Clang 14+)
- Python 3.11–3.14
- CUDA Toolkit 12.x — **optional**; everything builds and the full test suite
  runs on CPU-only machines (see [Running the tests](#running-the-tests))
- [`uv`](https://docs.astral.sh/uv/), our Python package manager

## Python

If you are interested in contributing to the Python side:

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate

uv lock
uv sync --extra cpu     # dev tools (pytest, black, ruff) are included by default
uv pip install -e .
```

A few notes:

- `--extra cpu` installs the CPU build of PyTorch, which the test suite uses as
  a reference implementation. On a machine with an NVIDIA GPU, use
  `--extra cu128` instead (the two extras conflict, pick one).
- `uv pip install -e .` compiles the C++ core and the pybind11 bindings, so the
  first install takes a few minutes. The editable install does **not**
  automatically rebuild the C++ code — if you touch anything under `csrc/`,
  re-run `uv pip install -e .`.
- To build the Python package with CUDA enabled:

  ```bash
  uv pip install -e . -C cmake.define.LMP_ENABLE_CUDA=ON
  ```

### Running the tests

To run the full test suite:

```bash
uv run pytest tests/
```

**The tests work on CPU** — no GPU required. Tests are parametrized over
devices and probe at collection time whether the CUDA backend is usable; on a
CPU-only machine only the `cpu` variants run, and on a CUDA-enabled build the
same tests run again on `cuda`. GPU coverage for pull requests is handled by
CI (see below), so it's fine to develop and test locally on CPU only.

Tests automatically run when you push to a branch. We use
[Modal](https://modal.com/) for GPU testing on CI. Because Modal can be
expensive, we recommend cancelling the `ci-gpu` workflow if your changes don't
touch the CUDA files.

### Test layout

Tests are separated into unit tests, operation (stress) tests, and graph tests:

- `tests/unit/` — plain unit tests for Python-level features (modules, grad
  mode, ...).
- `tests/stress/` — single-operation stress tests. Each op is run against
  PyTorch on randomly generated matrices for many iterations, comparing both
  the forward values and the gradients within per-op tolerances.
- `tests/graphs/` — chain-level tests that check groups of operations and
  kernel fusion. Templates describe shape-agnostic op chains with typed slots;
  the runner samples concrete ops per slot and diffs forward + backward against
  PyTorch, with and without fusion. To add a new graph test, add a template in
  `tests/graphs/templates.py`.
- `tests/ci/` — the Modal entrypoint used by the GPU CI workflow, not part of
  the local pytest run.

### Formatting and linting

We use `black` for formatting and `ruff` for linting:

```bash
uv run black .
uv run ruff check --fix
```

CI checks both, and the `autofix.ci` bot will push formatting fixes to your PR
automatically if you forget.

## C++/CUDA

The C++/CUDA core is a standalone library called **Lamp3** (`lmp::` namespace,
`LMP_` CMake options), which lives in `csrc/`. The Python layer is a thin
pybind11 wrapper around it, so most "real" changes happen here.

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

- `CMAKE_EXPORT_COMPILE_COMMANDS=ON` generates `build/compile_commands.json`,
  which clang-tidy and most editor tooling need.
- `LMP_ENABLE_TEST=ON` builds the GoogleTest suites. GoogleTest is fetched
  automatically via `FetchContent` (or found on the system if installed), so
  you don't need to install it yourself.
- Set `-DLMP_ENABLE_CUDA=ON` to build the CUDA backend (requires CUDA
  Toolkit 12.x). Everything compiles fine with it `OFF`.
- `LMP_ENABLE_BENCH=ON` builds the Google Benchmark suites in `benchmarks/`.
- If a `.venv` exists at the repo root, CMake picks up its Python for the
  bindings — create the venv first (see the Python setup above) to avoid
  version mismatches.

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

There's also `build/csrc/tests/playground` (from `csrc/tests/playground.cpp`),
a scratch executable that's handy for quickly poking at the C++ API while
developing.

The C++ tests are just unit tests and aren't very comprehensive — the Python
stress/graph suites (which compare against PyTorch) are the main correctness
net, so it's recommended to push to origin and let CI run the full matrix.

If OpenMP is available on your system, CMake picks it up automatically, which
speeds up the CPU kernels by roughly 3x (the CPU kernels are otherwise very
unoptimized — GPU is the optimization target).

### Formatting and linting

C++ formatting is clang-format (Google style, see `.clang-format`) and linting
is clang-tidy (see `.clang-tidy`). Both are wired up as CMake targets:

```bash
cmake --build build --target format      # clang-format -i on all sources
cmake --build build --target lint        # clang-tidy (check only)
cmake --build build --target lint-fix    # clang-tidy with -fix-errors
```

The `lint` targets need `compile_commands.json`, so make sure you configured
with `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`.

### Coverage (optional)

Configure with `-DLMP_ENABLE_COVERAGE=ON` (requires `lcov`) and use the
`coverage` target to generate a combined C++/Python HTML coverage report in
`build/coverage/`.

## Continuous integration

Every pull request runs these workflows:

| Workflow | What it does |
|---|---|
| `ci` | Builds the C++ core (CPU-only), runs the GTest suites, then installs the Python package and runs the full pytest suite on CPU. |
| `ci-gpu` | Runs the whole pytest suite on an NVIDIA T4 via Modal, with `LMP_ENABLE_CUDA=ON`. This is the only place the CUDA paths get exercised — but it costs money, so please cancel it if your change can't affect the CUDA code. |
| `lint` | clang-format + clang-tidy on the C++ side, black + ruff on the Python side. |
| `autofix.ci` | Auto-commits formatting/lint fixes back to your PR branch. |
| `docs` | (main only) Builds the Doxygen docs and deploys them to GitHub Pages. |

## Repo organization

```
csrc/                  Lamp3, the C++/CUDA core
  include/lamp3/       public headers
  src/
    tensor/            tensor storage + kernels (cpu/, cuda/, native/, lazy/)
    autograd/          Variable, backward functions
    inductor/          IR graph + kernel fusion (NVRTC JIT backend)
    nets/              neural-net modules and layers
  bindings/            pybind11 bindings (the `_rushlite` extension module)
  tests/               GoogleTest suites + playground
src/rushlite/          the Python package (thin layer over the bindings)
tests/                 pytest suites (unit/, stress/, graphs/, ci/)
benchmarks/            Google Benchmark suites + PyTorch comparison scripts
docs/                  Doxygen config and C++ examples
```

Miscellaneous notes:

- All major C++ structures (Tensor, Variable, modules, ...) are PImpl'd: the
  public headers in `csrc/include/lamp3/` expose stable handles, and the
  implementations live in `csrc/src/`.
- The name Lamp3 comes from lamp++ → lamppp → lamp3. The C++ library is fully
  usable standalone — see "Using Lamp3" in the [README](README.md).
- The Doxygen docs (built from `docs/` and the header comments) are published
  from `main` to GitHub Pages. You can build them locally with
  `cmake --build build --target docs` if Doxygen is installed.

## Submitting changes

1. Fork the repo and create a branch from `main`.
2. Make your changes, and add tests where it makes sense (a stress-test op
   entry, a graph template, a GTest case, or a unit test).
3. Run the formatters/linters and the tests locally (CPU is fine).
4. Open a pull request with a short, descriptive title — we loosely follow
   `feat:`/`fix:`/`docs:`/`chore:` prefixes — and describe *what* changed and
   *why* in the body.
5. Keep PRs focused: one feature or fix per PR is much easier to review.

For anything non-trivial (new ops, API changes, backend work), please open an
issue first so we can discuss the design before you invest a lot of time.
