# CUDA allocator tests

The host model and the CUDA implementation share the production
`BlockArena` in:

```text
csrc/src/tensor/cuda/allocator_core.hpp
```

The model adds only a fake segment provider and the segment-acquisition policy,
so its fast tests exercise the same splitting, global best-fit selection,
coalescing, and whole-segment extraction used by CUDA.

## Host-only suite

```sh
c++ -std=c++20 -O2 -Wall -Wextra -Wpedantic \
  prototypes/cuda_allocator/allocator_model_test.cpp \
  -o /tmp/cuda_allocator_model_test
/tmp/cuda_allocator_model_test
```

The suite also runs cleanly with:

```sh
c++ -std=c++20 -O1 -g -Wall -Wextra -Wpedantic \
  -fsanitize=address,undefined \
  prototypes/cuda_allocator/allocator_model_test.cpp \
  -o /tmp/cuda_allocator_model_test_sanitized
/tmp/cuda_allocator_model_test_sanitized
```

## Focused CUDA suite

After building `tensor_core`, compile and run:

```sh
nvcc -std=c++17 --default-stream=legacy -DLMP_ENABLE_CUDA \
  -I csrc/include -I build/csrc/include \
  prototypes/cuda_allocator/cuda_allocator_test.cu \
  -L build/csrc/src/tensor -ltensor_core -lcudart \
  -o /tmp/cuda_allocator_test
LD_LIBRARY_PATH=build/csrc/src/tensor /tmp/cuda_allocator_test
```

This checks exact address reuse, legacy-default-stream ordering across a
logical free/reuse, and owning-device deallocation when two CUDA devices are
available.
