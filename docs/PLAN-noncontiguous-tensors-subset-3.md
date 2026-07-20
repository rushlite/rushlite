# Plan: Non-Contiguous Tensors V1 — Subset 3

## Source and baseline

This plan implements subset 3 from
`docs/SCOPE-noncontiguous-tensors-v1.md` on
`nlin/noncontiguous-tensors` at `6a37fea`.

Implementation location:

- Branch: `codex/noncontiguous-tensors-subset-1`
- Base: `main` at `008c2f1`
- Worktree: `.worktrees/noncontiguous-tensors-subset-1`
- Subset 1 checkpoint: `439e639`
- Subset 2 checkpoint: `4fc1ff5`

The scope document remains the high-level contract on
`nlin/noncontiguous-tensors`. It is intentionally not copied into this
worktree.

The requested subset-3 plan did not exist at this path when implementation
started. This reviewed plan reconstructs the operator contract from the scope
and the subset-2 handoff before changing any subset-3 code.

## Outcome

Subset 3 is complete when the eager CPU and CUDA implementations of unary,
binary, reduction, and matrix multiplication:

- Read dense non-contiguous inputs in logical order.
- Keep fresh computational outputs contiguous.
- Preserve input storage and layout metadata.
- Use the shared subset-2 offset calculator rather than introducing another
  coordinate mapping.
- Support elementwise and matmul broadcasting without materializing expanded
  operands.
- Preserve existing contiguous behavior.
- Keep non-contiguous operands out of flat generated fusion kernels.
- Produce correct gradients for stride-aware backward helpers, reductions,
  and broadcasted batched matmul.

Subset 3 does not introduce strided writes, storage offsets, slicing, negative
strides, vector promotion for one-dimensional matmul, or generated
stride-aware fusion kernels.

## Review decisions

1. **Unary and reduction retain contiguous fast paths.** Their current flat
   addressing is valid when `input.is_contiguous()`. A handler-owned
   `OffsetUtil<1>` selects the strided route only for a non-contiguous input.
2. **Binary has one execution path.** Every `BinaryMetaHandler` already owns
   `OffsetUtil<2>` after subset 2. Arithmetic, comparisons, and internal
   backward helpers all consume it. The temporary `expand_` route and the
   separate CPU/CUDA expand kernels are removed.
3. **Matmul has dedicated metadata.** Complete matrix shapes must not pass
   through elementwise `AlignUtil`. `infer_matmul` validates rank and
   contraction dimensions, broadcasts leading dimensions only, and reports
   batch shape/count plus `M`, `N`, `K`.
4. **Matmul offsets describe batches only.** The iteration shape is the
   output batch shape. Each operand contributes only leading shape/stride
   metadata. Final matrix strides are passed explicitly to the matrix kernel.
5. **Rank-two matmul is a batch of one.** An empty batch shape has product one
   and both batch bases are zero. This is distinct from an output batch shape
   containing a zero dimension, which launches no work.
6. **CUDA kernels receive values, not ownership.** Unary, binary, reduction,
   and matmul kernels receive `OffsetCalculator<N>` by value. No polymorphic
   owner, host pointer, or temporary device metadata allocation crosses a
   launch boundary.
7. **The CUDA optimized matmul remains available.** It is extended with
   batch grid-stride iteration and explicit matrix strides. Vectorized global
   `B` loads are permitted only when the physical column stride is one and
   the selected address is aligned; otherwise scalar loads are used.
8. **Matmul backward reduces broadcast axes.** The logical formulas use
   metadata-only final-two-dimension transposes, then
   `sum_broadcast_axis` reduces every leading dimension introduced or
   expanded by batch broadcasting before gradient accumulation.
9. **Fusion gating is checked at both boundaries.** Native unary/binary
   entry points record only contiguous, shape-compatible leaves. The NVRTC
   fused graph asserts that every realized leaf is contiguous before code
   generation can emit flat `input[i]` reads.
10. **Zero-work launches are skipped.** Elementwise and reduction CUDA
    launchers return when the output has zero elements. Matmul also skips
    launches when the batch count, `M`, or `N` is zero.

## Implementation sequence

### 1. Route unary kernels through arity-one offsets

Primary files:

- `csrc/include/lamp3/tensor/cpu/unary.hpp`
- `csrc/src/tensor/cpu/unary.cpp`
- `csrc/include/lamp3/tensor/cuda/unary.cuh`
- `csrc/src/tensor/cuda/unary.cu`
- `csrc/src/tensor/cpu/meta_handler.cpp`

Changes:

- Keep the existing direct CPU loop and aligned CUDA vec4 route for
  contiguous inputs.
- Add CPU and CUDA kernels that map each output logical index through
  `OffsetUtil<1>` before reading the input.
- Select the strided route with `meta.has_offset()`.
- Apply the route to negation, exp, log, sqrt, abs, sin, cos, tan, and clamp.
- Keep output addressing flat because handler outputs are canonical.

Checkpoint:

- Every unary functor and clamp match a logical reference for rank-two
  transposes and higher-rank permutations on CPU and CUDA.

### 2. Replace binary/expand splitting with one offset-aware path

Primary files:

- `csrc/include/lamp3/tensor/cpu/binary.hpp`
- `csrc/src/tensor/cpu/binary.cpp`
- `csrc/include/lamp3/tensor/cuda/binary.cuh`
- `csrc/src/tensor/cuda/binary.cu`
- `csrc/src/tensor/cpu/kernels.cpp`
- `csrc/src/tensor/cuda/kernels.cu`
- `csrc/src/tensor/CMakeLists.txt`
- `csrc/include/lamp3/tensor/cpu/expand.hpp`
- `csrc/src/tensor/cpu/expand.cpp`
- `csrc/include/lamp3/tensor/cuda/expand.cuh`
- `csrc/src/tensor/cuda/expand.cu`

Changes:

- Make the binary launcher require the arity-two calculator and use the two
  returned input offsets for every output index.
- Remove the same-shape flat/vectorized route, the `meta.expand()` execution
  branch, and dead expand sources/headers/CMake entries.
- Route arithmetic, comparisons, absolute-value backward, and clamp backward
  through the same implementation.
- Retain `OpMeta::expand` temporarily only if another metadata consumer still
  needs it; it no longer controls a kernel choice.

Checkpoint:

- Matching non-contiguous layouts, mixed contiguous/non-contiguous layouts,
  different permutations, broadcasting, mixed dtypes, and backward helpers
  pass on both backends.

### 3. Add strided reduction base addressing

Primary files:

- `csrc/include/lamp3/tensor/cpu/reduct.hpp`
- `csrc/src/tensor/cpu/reduct.cpp`
- `csrc/include/lamp3/tensor/cuda/reduct.cuh`
- `csrc/src/tensor/cuda/reduct.cu`
- `csrc/src/tensor/infer_meta.cpp`

Changes:

- Keep the existing contiguous reduction formula for contiguous inputs.
- For a non-contiguous input, map each keep-dimension output index through
  `OffsetUtil<1>` to an input base.
- Walk the reduced dimension with the input's real
  `input.strides()[axis]`.
- Pass only reduced extent and reduced stride to the strided CUDA kernel;
  remove temporary device copies of complete shape/stride arrays from that
  route.
- Continue validating the axis before indexing input metadata.

Checkpoint:

- Sum, min, max, and product pass for every axis of transposed rank-two and
  permuted rank-three inputs, including the discriminating
  `[3,4,2]` / `[4,1,12]` layout.

### 4. Add dedicated matmul metadata and CPU batch execution

Primary files:

- `csrc/include/lamp3/tensor/infer_meta.hpp`
- `csrc/src/tensor/infer_meta.cpp`
- `csrc/include/lamp3/tensor/cpu/matrix.hpp`
- `csrc/src/tensor/cpu/matrix.cpp`
- `csrc/src/tensor/cpu/kernels.cpp`

Changes:

- Add `MatmulMeta` and `infer_matmul`.
- Require both ranks to be at least two and matching contraction dimensions.
- Right-align and broadcast leading batch shapes, including missing
  dimensions.
- Construct a batch-only `OffsetUtil<2>` from leading operand layouts.
- Parallelize CPU work over batch/row/column without a nested OpenMP
  reduction per output element.
- Address `A` and `B` using batch bases and their final-two-dimension physical
  strides.
- Allocate one canonical output with
  `output_batch_shape + [M,N]`.

Checkpoint:

- Rank-two transposes, shared rank-two operands, multi-axis batch broadcast,
  batch permutations, mixed matrix/batch permutations, and validation cases
  pass on CPU.

### 5. Extend simple and optimized CUDA matmul

Primary files:

- `csrc/include/lamp3/tensor/cuda/matrix.cuh`
- `csrc/src/tensor/cuda/matrix.cu`
- `csrc/src/tensor/cuda/kernels.cu`

Changes:

- Pass batch count, the batch offset calculator, and four matrix strides to
  both CUDA kernels.
- Assign batches through `grid.z` with a grid-stride loop and cap `grid.z` at
  the device maximum.
- Reset per-thread accumulation state for every optimized-kernel batch.
- Synchronize shared-memory tiles independently for each batch iteration.
- Clear the complete batched output if the optimized beta path still relies
  on a pre-clear.
- Gate vectorized `B` loads on physical column stride and per-batch address
  alignment; retain scalar loads otherwise.
- Keep contiguous vector output stores within each matrix.

Checkpoint:

- Small and large batched cases exercise simple and optimized routes, a
  batch count beyond `grid.z`, and transposed/unaligned `B`.

### 6. Complete fusion and autograd integration

Primary files:

- `csrc/src/tensor/native/unary_ops.cpp`
- `csrc/src/tensor/native/expand_ops.cpp`
- `csrc/src/inductor/nvrtc/fused_graph.cpp`
- `csrc/src/autograd/functions/matrix_ops.cpp`
- `csrc/src/autograd/utils/grad_utils.cpp`

Changes:

- Prevent lazy recording when a unary input is non-contiguous.
- Prevent binary lazy recording when either input is non-contiguous or the
  operation requires broadcasting.
- Assert realized fused leaves are contiguous.
- Use final-two-dimension metadata transposes in matmul backward.
- Reduce all broadcasted batch axes back to each original operand shape
  before `incr_grad`.
- Keep produced gradients contiguous before flat gradient accumulation.

Checkpoint:

- Non-contiguous eager consumers are contiguous fusion barriers, internal
  backward helpers honor strides, and broadcasted batched matmul gradients
  have the original operand shapes and values.

### 7. Add integration tests and cleanup

Primary files:

- `csrc/tests/noncontiguous_ops_tests.cpp`
- `csrc/tests/CMakeLists.txt`
- All touched operator headers and sources

Changes:

- Add one CPU/CUDA parameterized integration target for the four operator
  families.
- Use logical references independent of physical storage order.
- Assert result shapes, canonical strides, values, and input preservation.
- Cover mixed dtypes and empty-output no-launch behavior.
- Remove stale expand declarations/includes and temporary route checks.
- Run formatting when available and `git diff --check`.

## Test matrix

### Unary

- Every public unary functor on a rank-two transpose.
- Higher-rank permutation.
- Clamp.
- Canonical output and unchanged input layout/storage.

### Binary

- Every arithmetic and comparison functor on matching transposes.
- One contiguous and one non-contiguous operand.
- Different physical permutations with one logical shape.
- Broadcasted contiguous and non-contiguous operands.
- Multiple leading broadcast dimensions.
- Mixed dtypes.
- Absolute-value and clamp backward helpers.

### Reduction

- Sum, min, max, and product on every rank-two axis.
- Every axis of several rank-three permutations.
- Size-one reduction dimensions.
- The `[3,4,2]` / `[4,1,12]` discriminating layout.
- Invalid axis.

### Matmul

- Existing contiguous rank-two regression.
- Transposed `A`, transposed `B`, and both transposed.
- Matching and broadcasted multidimensional batches.
- Shared rank-two left and right operands.
- Batch permutations and permutations crossing matrix/batch roles.
- Mixed dtypes.
- Invalid rank, contraction, and batch compatibility.
- Small/simple and large/optimized CUDA routes.
- Batch count greater than CUDA `grid.z`.

### Autograd and fusion

- Unary backward helpers with transposed inputs.
- Reduction backward over transposed inputs.
- Rank-two matmul backward regression.
- Batched and broadcasted matmul backward.
- Multiple broadcast axes and repeated accumulation.
- A non-contiguous input executes eagerly and produces a contiguous result.

## Verification commands

CPU:

```bash
cmake -S . -B build/subset3-cpu \
  -DCMAKE_BUILD_TYPE=Debug \
  -DLMP_ENABLE_TEST=ON \
  -DLMP_ENABLE_CUDA=OFF
cmake --build build/subset3-cpu -j2
ctest --test-dir build/subset3-cpu --output-on-failure
```

CUDA:

```bash
CUDACXX=/usr/local/cuda/bin/nvcc \
cmake -S . -B build/subset3-cuda \
  -DCMAKE_BUILD_TYPE=Debug \
  -DLMP_ENABLE_TEST=ON \
  -DLMP_ENABLE_CUDA=ON
cmake --build build/subset3-cuda -j2
ctest --test-dir build/subset3-cuda --output-on-failure
```

Also run `git diff --check` and search for stale `expand_dispatch_handler`,
`meta.expand()` routing, and flat binary input indexing before handoff.

## Completion criteria

Subset 3 is ready for handoff when:

- Unary non-contiguous reads use arity-one offsets on CPU and CUDA.
- Binary has one arity-two offset-aware route for every functor.
- Reduction non-contiguous reads use mapped bases and the real reduced-axis
  stride.
- Matmul performs leading-dimension broadcast inference independently of
  matrix contraction.
- Rank-two, batched, broadcasted, and non-contiguous matmul pass on CPU and
  CUDA.
- CUDA simple and optimized kernels both honor batch bases and matrix strides.
- Matmul backward reduces broadcasted batch axes to each operand shape.
- Flat fusion kernels cannot receive non-contiguous leaves.
- Every computational output is canonical and contiguous.
- Existing CPU and CUDA regressions pass.
