# Plan: Non-Contiguous Tensors V1 — Subset 2

## Source and baseline

This plan implements subset 2 from
`docs/SCOPE-noncontiguous-tensors-v1.md` on
`nlin/noncontiguous-tensors` at `6a37fea`.

Implementation location:

- Branch: `codex/noncontiguous-tensors-subset-1`
- Base: `main` at `008c2f1`
- Worktree: `.worktrees/noncontiguous-tensors-subset-1`

The worktree already contains the first implementation slice from subset 1:
layout metadata, metadata-only transpose and permute, stride-preserving
squeeze and expand-dimensions, contiguity checks, and layout tests. Those
changes are intentionally preserved. Subset 2 is planned in the same worktree
because subset 1's contiguous materialization will be its first direct
consumer.

The scope document remains the high-level contract on
`nlin/noncontiguous-tensors`; this document maps that contract to the code
currently on `main` plus the in-progress subset-1 layout changes.

## Outcome

Subset 2 is complete when Lamp3 has one tested transformation from a logical
row-major iteration index to the physical storage offsets of one or more input
operands:

- The logical iteration shape is explicit and independent of an output
  `TensorImpl`.
- Each operand contributes its own shape and physical strides.
- Ranks align from the right.
- Missing and broadcasted dimensions have effective stride zero.
- Contiguous, transposed, permuted, broadcasted, reduction-projection, and
  matmul-batch mappings use the same rules.
- `OffsetUtil<1>`, `OffsetUtil<2>`, and the existing `OffsetUtil<3>` are
  available through CPU and CUDA dispatch.
- CPU `get()` is const and safe in concurrent OpenMP loops.
- CUDA kernels consume a trivially copyable device representation whose
  metadata and ownership are valid for the whole launch.
- Direct CPU and CUDA offset tests prove parity without depending on
  subset-3 computational kernels.
- Existing contiguous and broadcasted binary behavior remains green while
  callers migrate.

Subset 2 does not make unary, binary, reduction, materialization, or matmul
kernels layout-aware. It supplies the mapping primitive and updates ownership
at the metadata-handler boundary. Subset 1 consumes arity one for contiguous
materialization; subset 3 consumes the remaining routes.

## Contracts to lock before implementation

1. **Offsets describe inputs only.** `get(logical_index)` returns one
   `stride_t` physical offset per operand in operand order. The output write
   position is the caller's original `logical_index`, because V1
   computational outputs are contiguous. The current extra result slot for
   the output is removed, and the existing expand kernel is adjusted from
   indices `[1]`/`[2]` to `[0]`/`[1]`.
2. **Construction is layout-based, not tensor-output-based.** The factory
   accepts an explicit iteration shape and operand shape/stride pairs. It does
   not infer logical strides from `outs.strides()` and does not require a
   complete output tensor. This is required for matmul's synthetic batch-only
   iteration space.
3. **The utility maps; operations validate semantics.** It validates
   shape/stride rank agreement, operand rank no greater than iteration rank,
   and `LMP_MAX_DIMS`. Binary broadcasting compatibility, reduction axes, and
   matmul contraction and batch compatibility stay in metadata inference or
   the native operation.
4. **Rank-zero and zero-element iteration spaces are distinct.** A rank-zero
   matmul batch shape represents one matrix and maps index zero to all-zero
   bases. An iteration shape containing a zero-sized dimension has zero
   logical elements; callers launch no work and never call `get()`. The
   current `ndim > 0` assertion is removed.
5. **Retain arity three.** There is no active arity-three caller in the
   current tree, but its dispatch symbols and explicit instantiations already
   exist. Preserve them to avoid an unrelated internal API regression.
6. **Do not copy ownership objects into CUDA kernels.** CUDA kernels receive a
   plain fixed-capacity calculator or device view by value. It contains rank,
   canonical logical strides, and effective operand strides, but no
   `DataPtr`, `ListDevicePtr`, host pointer, virtual state, or owning object.
7. **Keep the current binary route during this subset.** Binary metadata may
   construct `OffsetUtil<2>` unconditionally, as required by the target
   contract, but equal-shape inputs continue through the current flat binary
   kernel until subset 3 unifies the execution path. The `expand` flag can
   remain temporarily as a routing compatibility field; it is not an offset
   accessor invariant.
8. **No strided writes or storage offsets are introduced.** V1 offsets begin
   at storage offset zero and address read operands only.

## Proposed internal interface

The exact spelling may adapt to compilation constraints, but the API should
have these roles:

```cpp
struct OperandLayout {
  shape_list shape;
  stride_list strides;
};

template <size_t NArgs>
using offsets_t = std::array<stride_t, NArgs>;

template <size_t NArgs>
using offset_util_fn =
    std::unique_ptr<OffsetUtil> (*)(
        const shape_list& iteration_shape,
    const std::array<OperandLayout, NArgs>& operands);
```

Implementation review decision: place the pointer-free calculator and layout
types in a backend-neutral internal header,
`csrc/include/lamp3/tensor/offset_util.hpp`. Use a small trivially copyable
`OffsetValues<NArgs>` result instead of `std::array` when CUDA's host/device
annotations require it. CPU and CUDA dispatch wrappers own this common value;
only the common value is passed to a CUDA kernel.

Callers may use a helper that snapshots a full `TensorImpl` layout. Matmul can
instead construct `OperandLayout` values from only the leading shape and
stride entries. The factory must copy or fully lower the supplied vectors
during construction; it must not retain references to temporary leading-dim
vectors.

The calculation state should have a fixed-capacity, trivially copyable form
bounded by `LMP_MAX_DIMS`:

```text
rank
logical_strides[LMP_MAX_DIMS]
effective_strides[NArgs][LMP_MAX_DIMS]
```

Using one plain calculator for both hosts is preferred because it prevents CPU
and CUDA rules from drifting. If compiler separation requires a host owner and
a CUDA device view, both are built by one shared lowering routine and the
device view remains the sole value copied into a kernel.

The dispatch surface becomes:

```text
offset_util_stub_1
offset_util_stub_2
offset_util_stub_3
```

with CPU and CUDA registrations for each arity when that backend is enabled.

## Mapping algorithm

### Canonical logical strides

Build row-major logical strides from the explicit iteration shape:

```text
expected = 1
for dim from last to first:
    logical_stride[dim] = expected
    expected *= iteration_shape[dim]
```

These strides describe coordinate decomposition only. They are not copied
from an output tensor and are unrelated to any operand's physical layout.

For a zero-element shape, some leading logical strides can be zero. That is
valid construction state because no caller may invoke `get()` for an empty
iteration space.

### Effective operand strides

For every operand, right-align its rank to the iteration rank. For each
iteration dimension:

```text
operand dimension missing:
    effective stride = 0

operand shape is 1 and iteration shape is greater than 1:
    effective stride = 0

otherwise:
    effective stride = operand's real physical stride
```

A size-one dimension aligned with another size-one dimension may keep its real
stride; its only coordinate is zero, so either value maps identically.

The builder does not silently replace a real stride merely because the
operand is non-contiguous. Transpose and permutation strides must reach the
original physical storage.

### Logical decomposition

For a non-empty iteration space:

```text
remaining = logical_index
offsets = all zero

for dim from first to last:
    coordinate = remaining / logical_stride[dim]
    remaining %= logical_stride[dim]

    for operand in operands:
        offsets[operand] +=
            coordinate * effective_stride[operand][dim]
```

All intermediate and returned offsets use `stride_t`. `get()` performs no
mutation or allocation.

## Implementation sequence

### 1. Separate layout description from output tensors

Primary files:

- `csrc/include/lamp3/tensor/offset_util.hpp`
- `csrc/include/lamp3/tensor/cpu/offset_util.hpp`
- `csrc/src/tensor/cpu/offset_util.cpp`
- `csrc/include/lamp3/tensor/utils/align_utils.hpp`
- `csrc/src/tensor/utils/align_utils.cpp`

Changes:

- Add the internal operand-layout and input-offset result types.
- Replace the base class's `ndim` plus `init_padded_strides(shape, stride)`
  model with a builder that receives the explicit iteration shape.
- Generate canonical logical strides locally rather than using a physical
  output tensor's strides.
- Generate right-aligned effective strides for every operand.
- Snapshot all supplied metadata during construction.
- Add structural validation before indexing or copying metadata:
  - operand shape and stride ranks match;
  - operand rank does not exceed iteration rank;
  - iteration rank does not exceed `LMP_MAX_DIMS`;
  - no operand rank exceeds `LMP_MAX_DIMS`.
- Permit rank-zero construction and return all-zero offsets from `get(0)`.
- Keep semantic compatibility checks out of the utility.
- Centralize canonical-stride generation if `AlignUtil` and the offset builder
  would otherwise duplicate it. Do not make a physical output layout part of
  the offset API.

Checkpoint:

- A CPU-only unit can construct mappings directly from shape/stride fixtures,
  including synthetic layouts that are not backed by a `TensorImpl`.

### 2. Implement and register the CPU calculators

Primary files:

- `csrc/include/lamp3/tensor/cpu/offset_util.hpp`
- `csrc/src/tensor/cpu/offset_util.cpp`

Changes:

- Implement const, allocation-free `get()` for `CPUOffsetUtil<1>`,
  `CPUOffsetUtil<2>`, and `CPUOffsetUtil<3>`.
- Return input offsets only, indexed from zero.
- Add `offset_util_stub_1`; retain and update stubs two and three.
- Add explicit template instantiations and CPU dispatch registrations for all
  supported arities.
- Make the lowered state immutable after construction so one instance is safe
  to share across an OpenMP loop.
- Test rank-zero directly and ensure no zero-element test calls `get()`.

Checkpoint:

- Direct CPU tests cover unary, binary, reduction-projection, and batch-base
  layouts, and a parallel loop returns deterministic offsets.

### 3. Migrate metadata-handler ownership without changing operator scope

Primary files:

- `csrc/include/lamp3/tensor/cpu/meta_handler.hpp`
- `csrc/src/tensor/cpu/meta_handler.cpp`
- `csrc/include/lamp3/tensor/infer_meta.hpp`
- `csrc/src/tensor/infer_meta.cpp`
- `csrc/src/tensor/cpu/expand.cpp`
- `csrc/include/lamp3/tensor/cpu/expand.hpp`

Changes:

- Construct binary `OffsetUtil<2>` from `OpMeta::shape` and the two input
  layouts instead of from `outTen_`.
- Construct it for both equal-shape and broadcasted binary metadata. Keep the
  current `expand()` execution branch only as a temporary subset-3 migration
  aid.
- Change `offset()` to assert that `outOffset_` exists, not that
  `expand_ == true`; add `has_offset()` if route selection needs a
  non-asserting query.
- Update the current CPU expand kernel for the input-only result ordering.
- Prepare unary and reduction metadata to own `OffsetUtil<1>` on their
  non-contiguous routes. Subset 2 does not switch their kernels to that route;
  subset 3 consumes the stored calculator.
- For reduction, use the keep-dimension output shape as the iteration shape
  and preserve the input's real reduced-axis stride separately for the
  operation loop.
- Validate a reduction axis before `infer_reduct` indexes the input shape.
- Continue allocating every handler output with canonical row-major strides.
- Do not add matmul metadata inference here. Subset 3 will construct a
  batch-only `OffsetUtil<2>` independently from the elementwise handler.

Checkpoint:

- Existing same-shape and broadcasted binary tests pass on CPU, while direct
  inspection tests confirm every binary handler owns an offset calculator.

### 4. Replace CUDA pointer ownership with a device value

Primary files:

- `csrc/include/lamp3/tensor/cuda/offset_util.cuh`
- `csrc/src/tensor/cuda/offset_util.cu`
- `csrc/include/lamp3/tensor/cuda/list_ptr.cuh`
- `csrc/include/lamp3/tensor/cuda/expand.cuh`
- `csrc/src/tensor/cuda/expand.cu`

Changes:

- Add a trivially copyable CUDA calculator/device-view representation bounded
  by `LMP_MAX_DIMS`.
- Make its `get()` device-callable and identical to the CPU decomposition.
- Have `CUDAOffsetUtil<NArgs>` own or expose that plain value on the host.
  Kernel launchers pass only the value, never the polymorphic wrapper.
- Remove the current pattern that builds multiple `ListDevicePtr<stride_t>`
  allocations and then raw-copies a `CUDAOffsetUtil` containing those owning
  wrappers to device memory.
- If the calculator is passed by value, eliminate offset-metadata device
  allocations entirely. If a device allocation is retained for a justified
  launch constraint, copy only the plain view and keep its host owner alive
  through completion.
- Update the current CUDA expand launcher for input-only result ordering and
  value-based calculator passing.
- Add `OffsetUtil<1>`, `<2>`, and `<3>` explicit instantiations and CUDA
  dispatch registrations.
- Preserve the existing stream/synchronization behavior in this subset unless
  a change is required for metadata lifetime correctness.

Checkpoint:

- CUDA compilation verifies the calculator is trivially copyable and contains
  no owning or host-only pointer members; existing broadcast operations remain
  correct.

### 5. Add direct CPU/CUDA parity tests

Primary files:

- `CMakeLists.txt`
- `csrc/tests/offset_util_tests.cpp`
- `csrc/tests/offset_util_cuda_tests.cu`
- optionally `csrc/tests/offset_util_test_cases.hpp`
- `csrc/tests/CMakeLists.txt`

Changes:

- Add a dedicated CPU offset-utility test target linked to `tensor_core`.
- Enable CTest when `LMP_ENABLE_TEST` is on so the documented regression
  commands discover all test executables.
- Under `LMP_ENABLE_CUDA`, add a CUDA test target with a small test-only kernel
  that calls the device calculator and copies only returned offsets back for
  comparison.
- Share fixture definitions between CPU and CUDA where practical so the same
  logical indices and expected offsets exercise both implementations.
- Use distinct dimensions, strides, coordinates, and expected offsets so a
  reversed or shifted dimension cannot accidentally pass.
- Add CTest discovery and coverage flags consistently with existing targets.
- Keep utility tests independent of unary/binary/reduction/matmul numerical
  kernels. Existing tensor tests remain the integration regression suite for
  the current binary-broadcast consumer.

Checkpoint:

- Every scope acceptance case has a direct CPU assertion and, when CUDA is
  enabled, an identical device assertion.

### 6. Publish stable adapters for subsets 1 and 3

Primary files:

- `csrc/include/lamp3/tensor/cpu/offset_util.hpp`
- `csrc/include/lamp3/tensor/cuda/offset_util.cuh`
- `csrc/include/lamp3/tensor/cpu/meta_handler.hpp`

Changes:

- Document how contiguous materialization constructs arity one over the full
  input logical shape.
- Document how unary and reduction kernels retrieve arity one from their
  handler.
- Document that binary retrieves arity two for every route after subset 3
  removes the flat/expand split.
- Expose a direct factory path for matmul to provide leading-only operand
  layouts and an output batch iteration shape without fabricating tensors.
- Document that rank-two matmul can bypass the utility with zero batch bases;
  a rank-zero utility remains valid for tests and generic callers.
- Keep these adapters internal to the tensor implementation. No C++ or Python
  public API is added by subset 2.

Checkpoint:

- Subset 1 can implement contiguous materialization without another offset
  algorithm, and subset 3 can consume the same representation without an API
  redesign.

## Test plan

### Core unary mapping

- Canonical contiguous layout across rank one, rank two, and higher rank.
- Rank-two transpose.
- Higher-rank permutation with deliberately non-canonical strides.
- Size-one dimensions with arbitrary real strides.
- Maximum rank `LMP_MAX_DIMS`.
- Arity-one construction through both direct layouts and a subset-1
  transposed/permuted `TensorImpl`.

### Binary mapping and broadcasting

- Two equal contiguous layouts.
- Two matching non-contiguous layouts.
- One contiguous and one non-contiguous layout with the same logical shape.
- Row broadcasting.
- Column broadcasting.
- Size-one scalar broadcasting using the repository's `{1}` scalar shape.
- Missing leading dimensions.
- Multiple leading broadcast dimensions.
- Broadcasting combined with a higher-rank permutation.
- Operand order is preserved in the returned offsets.
- An arity-three smoke case preserves the existing supported instantiation.

### Reduction projection

- Use the keep-dimension output shape as the iteration shape.
- Confirm the reduced coordinate contributes zero to the base offset.
- Confirm non-reduced coordinates use the input's real physical strides.
- Exercise every axis of at least one permuted rank-three layout.
- Include the scope regression layout:
  `shape=[3,4,2]`, `strides=[4,1,12]`, reducing axis 1.
- Keep the reduced-axis stride outside the utility and verify a test loop can
  walk `base + j * input_stride[axis]`.

### Matmul batch-base mapping

- Empty batch shape: both bases are zero and batch count is conceptually one.
- Matching one-dimensional batch shapes.
- Matching multidimensional batch shapes.
- Rank-two operand shared across all batches through missing leading
  dimensions.
- Size-one batch dimensions broadcast over larger output dimensions.
- Multiple operands broadcasting different leading dimensions, such as
  `[4,1]` with `[1,7]`.
- Non-canonical physical batch strides after permutation.
- Only leading shape/stride entries are provided; final matrix strides are not
  part of the batch utility.

### Structural validation and edge cases

- Reject an operand whose shape and stride ranks differ.
- Reject an operand rank greater than the iteration rank.
- Reject iteration and operand ranks greater than `LMP_MAX_DIMS`.
- Permit rank-zero construction.
- Permit construction for an iteration shape containing a zero-sized
  dimension but launch no test kernel and never call `get()`.
- Verify repeated and parallel CPU calls do not mutate calculator state.
- Verify returned values and intermediate accumulation use `stride_t`.

### CUDA parity and representation

- Run all core mapping tables through a test-only CUDA kernel.
- Compare every returned operand offset with CPU and hand-computed expected
  values.
- Exercise maximum rank and all supported arities.
- Add compile-time checks that the kernel argument/device view is trivially
  copyable.
- Run repeated launches to expose metadata lifetime errors.
- Avoid a zero-block launch for empty iteration spaces.

### Regression commands

CPU:

```bash
cmake -S . -B build/subset2-cpu \
  -DLMP_ENABLE_TEST=ON \
  -DLMP_ENABLE_CUDA=OFF \
  -DCMAKE_BUILD_TYPE=Debug
cmake --build build/subset2-cpu -j
ctest --test-dir build/subset2-cpu --output-on-failure
```

CUDA, on a CUDA runner:

```bash
cmake -S . -B build/subset2-cuda \
  -DLMP_ENABLE_TEST=ON \
  -DLMP_ENABLE_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Debug
cmake --build build/subset2-cuda -j
ctest --test-dir build/subset2-cuda --output-on-failure
```

Also run the repository formatter/linter over touched C++ and CUDA files and
use `git diff --check` before handoff.

## Reviewable change slices

Keep each slice green for the behavior it introduces:

1. Explicit iteration-shape API, common lowering, CPU arities 1/2/3, and
   direct CPU tests.
2. Metadata-handler migration plus existing CPU binary-broadcast regressions.
3. Trivially copyable CUDA representation, CUDA arities 1/2/3, and parity
   tests.
4. Stable subset-1/subset-3 adapters, edge-case coverage, and cleanup of the
   old output-tensor/pointer-owning API.

The first slice is immediately useful to subset 1. No slice should claim that
ordinary non-contiguous operators work until subset 3 connects their kernels.

## Risks and mitigations

- **An output tensor accidentally remains the source of logical strides.**
  Remove it from the constructor signature and test synthetic batch-only
  shapes that cannot be represented by a complete matmul output tensor.
- **Broadcasting erases a real non-contiguous stride.** Zero only missing
  dimensions and dimensions actually broadcast from size one; test
  permutation plus broadcasting together.
- **CPU and CUDA implementations drift.** Lower through one shared builder
  and run the same fixture table on both backends.
- **CUDA copies host ownership state into a kernel.** Separate the host
  wrapper from a plain fixed-capacity device value and enforce trivial-copy
  properties at compile time.
- **Asynchronous metadata dies before use.** Prefer a by-value kernel
  argument. If device allocation is unavoidable, keep the owning launcher
  state alive through kernel completion.
- **Rank-zero batches are mistaken for zero work.** Treat the product of an
  empty matmul batch shape as one in matmul metadata; keep zero-element tensor
  iteration as a separate no-launch case.
- **The migration breaks existing binary broadcasting before subset 3.**
  Update the expand consumer in the same slice as the result layout and keep
  the temporary `expand()` route until the unified binary kernel lands.
- **Type erasure hides a wrong arity downcast.** Keep arity in the factory,
  handler alias, and typed accessor, and cover all registered arities with
  construction tests.
- **Subset 2 absorbs operator semantics.** Limit it to coordinate
  transformation, structural validation, ownership, and compatibility
  plumbing; leave numerical kernel routing and matmul inference to subset 3.

## Completion criteria

Subset 2 is ready for handoff when:

- No offset utility constructor accepts an output `TensorImpl`.
- Logical strides always come from the explicit iteration shape.
- Effective operand strides follow right-aligned broadcast rules.
- Arity-one, arity-two, and retained arity-three factories are registered for
  every enabled backend.
- CPU `get()` is immutable and OpenMP-safe.
- CUDA kernels receive only device-safe, trivially copyable offset state.
- Existing binary broadcasting still passes.
- Direct CPU tests cover every scoped mapping and validation boundary.
- CUDA tests, when available, produce identical offsets for the same cases.
- Subset 1 can consume arity one for materialization, and subset 3 can consume
  unary, binary, reduction, and batch-matmul mappings without changing the
  core API.
