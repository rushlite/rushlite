# Plan: Non-Contiguous Tensors V1 — Subset 1

## Source and baseline

This plan implements subset 1 from
`docs/SCOPE-noncontiguous-tensors-v1.md` on
`nlin/noncontiguous-tensors` at `6a37fea`.

Implementation branch:

- Branch: `codex/noncontiguous-tensors-subset-1`
- Base: `main` at `008c2f1`
- Worktree: `.worktrees/noncontiguous-tensors-subset-1`

The scope document is intentionally not copied into this branch. It remains
the high-level contract on `nlin/noncontiguous-tensors`; this document maps
that contract to the implementation currently on `main`.

## Outcome

Subset 1 is complete when Lamp3 has a coherent layout lifecycle:

- Shape and stride metadata can describe canonical tensors and dense
  permutation views.
- `transpose`, `permute`, `squeeze`, and `expand_dims` are metadata-only,
  storage-sharing views.
- `reshape` shares storage for contiguous inputs and materializes
  non-contiguous inputs.
- `is_contiguous` and `contiguous` are available in C++, Python, and the
  tensor-facing native operation layer.
- Logical exports, printing, and device transfer preserve view order.
- Unsupported non-contiguous mutation fails explicitly.
- Creating a view from deferred data is safe, and non-contiguous tensors
  cannot enter a flat generated fusion kernel.
- View autograd formulas preserve logical gradients.

Subset 1 does not make the normal unary, binary, reduction, or matmul kernels
stride-aware. That is subset 3. Until subset 3 lands, non-contiguous views are
valid for metadata, explicit materialization, logical export, device transfer,
and view autograd tests, but are not a complete computational feature.

## Contracts to lock before implementation

1. Interpret the scope sentence about copy and in-place operations as
   rejecting **non-contiguous** operands.
2. Keep the no-dimension `transpose(a)` API and `.T` property as a
   rank-at-least-two convenience that swaps the final two dimensions.
3. Reject non-contiguous `fill` in subset 1 unless the V1 aliasing behavior is
   explicitly approved. This is the conservative default: it keeps all
   public mutation behind one contiguous-only rule and avoids establishing a
   behavior that will change when slicing introduces partial-storage views.
4. Treat subset 2's generalized `OffsetUtil<1>` CPU/CUDA implementation as a
   prerequisite for the materialization kernel. Subset 1 should consume that
   API, not introduce a second logical-index-to-storage-offset algorithm.

## Implementation sequence

### 1. Establish layout invariants in `TensorImpl`

Primary files:

- `csrc/include/lamp3/tensor/tensor_impl.hpp`
- `csrc/src/tensor/tensor_impl.cpp`
- `csrc/include/lamp3/tensor/tensor.hpp`
- `csrc/src/tensor/tensor.cpp`

Changes:

- Add `TensorImpl::is_contiguous() const noexcept` using the scope algorithm:
  empty tensors are contiguous, size-one dimensions are ignored, and every
  other stride must match the expected row-major stride.
- Restrict the current `update_strides()` helper to construction of a
  known-contiguous layout. Rename it if that makes the invariant clearer, and
  remove the current comment that requires recalculation after every shape
  change.
- Keep all allocation constructors canonical and preserve the invariant that
  realized storage contains exactly `numel` elements at storage offset zero.
- Add metadata operations for `transpose(dim0, dim1)` and `permute(dims)`.
  Validate dimensions/permutations before copying the impl, then reorder both
  `shape_` and `strides_` while continuing to share `Storage`.
- Change `squeeze(dim)` to erase the matching shape and stride entries.
- Change `expand_dims(dim)` to insert shape `1` and the scoped stride:
  `old_stride[dim] * old_shape[dim]` before an existing dimension, otherwise
  `1`.
- Before any storage-sharing impl copy, realize a deferred source. This
  applies to transpose, permute, squeeze, expand-dimensions, and contiguous
  reshape.
- Make reshape layout-dependent: a contiguous input gets canonical strides
  over shared storage; a non-contiguous input first uses `contiguous()` and
  then reshapes that materialized impl.
- Add public `Tensor` wrappers for `is_contiguous`, `contiguous`,
  `transpose(dim0, dim1)`, and `permute(dims)`.
- Make public `Tensor::contiguous()` return the existing `Tensor` handle when
  it is already contiguous. This preserves deferred-impl ownership instead of
  copying a pending `LazyFunction`.
- Keep `index()` stride-based and add per-dimension bounds validation while
  touching the path.

Checkpoint:

- Constructor, contiguity, metadata validation, storage aliasing, and
  realize-before-view tests pass without using computational kernels.

### 2. Replace physical transpose with layout views

Primary files:

- `csrc/include/lamp3/tensor/native/shape_ops.hpp`
- `csrc/src/tensor/native/shape_ops.cpp`
- `csrc/include/lamp3/tensor/native/matrix_ops.hpp`
- `csrc/src/tensor/native/matrix_ops.cpp`
- `csrc/include/lamp3/tensor/cpu/kernels.hpp`
- `csrc/src/tensor/cpu/kernels.cpp`
- `csrc/include/lamp3/tensor/cuda/kernels.cuh`
- `csrc/src/tensor/cuda/kernels.cu`

Changes:

- Add native free functions for explicit-axis transpose, permute,
  contiguity, and contiguous materialization.
- Keep `ops::transpose(a)` as a compatibility wrapper that validates rank and
  forwards to the final-two-axis metadata transpose.
- Remove `transpose_stub`, its dispatch type, and the CPU/CUDA physical
  transpose registrations and kernels once no call site uses them.
- Keep matrix multiplication routing unchanged in this subset; only its
  transpose helper changes from an allocation to a view.

Checkpoint:

- Rank-two legacy transpose and every valid higher-rank axis pair have view
  semantics, and no physical transpose kernel remains on the public path.

### 3. Add stride-aware contiguous materialization

Primary files:

- `csrc/include/lamp3/tensor/native/memory_ops.hpp`
- `csrc/src/tensor/native/memory_ops.cpp`
- `csrc/include/lamp3/tensor/cpu/memory.hpp`
- `csrc/src/tensor/cpu/memory.cpp`
- `csrc/include/lamp3/tensor/cuda/memory.cuh`
- `csrc/src/tensor/cuda/memory.cu`
- `csrc/src/tensor/CMakeLists.txt` only if implementation is split into new
  source files

Dependency:

- Generalized `OffsetUtil<1>` from subset 2, constructed from the input
  layout and the input's logical iteration shape.

Changes:

- Add a dtype-preserving strided-copy dispatch path that receives layout
  metadata; do not widen the raw pointer-only `copy_stub`.
- For a non-contiguous input, allocate same-device/same-dtype storage,
  construct canonical output metadata with the same shape, and copy
  `input[offset_util.get(logical_index)]` to `output[logical_index]`.
- Implement the CPU path as a read-only, OpenMP-safe loop.
- Implement the CUDA path as a grid-stride kernel using the subset 2 device
  offset representation, with metadata lifetime extending through kernel
  completion.
- Handle zero elements without launching a kernel.
- Ensure internal `TensorImpl::contiguous()` realizes a deferred contiguous
  impl before returning an impl copy; public `Tensor::contiguous()` keeps its
  same-handle fast path.

Checkpoint:

- Materializing rank-two transposes and higher-rank permutations produces
  canonical strides, correct logical order, independent storage, and
  unchanged source storage on CPU and CUDA.

### 4. Route all public logical reads through layout-aware behavior

Primary files:

- `csrc/include/lamp3/tensor/tensor.hpp`
- `csrc/src/tensor/tensor_impl.cpp`
- `csrc/src/tensor/native/memory_ops.cpp`
- `csrc/bindings/tensor.hpp`
- `csrc/bindings/variable.hpp`

Changes:

- Make `Tensor::to_vector<T>()` read from a contiguous logical
  materialization rather than copying physical storage directly.
- Python `tolist` then inherits logical order for both `_Tensor` and
  `Variable`.
- Make tensor printing read logical order. Use RAII storage for the temporary
  host buffer while modifying this code.
- Make `ops::to` first obtain a logical contiguous source, then perform the
  existing cross-device flat copy into a fresh canonical tensor.
- Preserve current same-device `to` behavior unless separately changed; this
  subset only changes layout correctness.

Checkpoint:

- `to_vector`, `tolist`, `operator<<`, and CPU↔CUDA transfer agree on logical
  order for transpose and a nontrivial rank-three permutation.

### 5. Enforce the V1 mutation boundary

Primary files:

- `csrc/src/tensor/tensor_impl.cpp`
- `csrc/src/tensor/native/memory_ops.cpp`
- `csrc/src/autograd/variable.cpp`

Changes:

- Require contiguous source and destination in `Tensor::copy`; retain dtype
  conversion and cross-device behavior for supported contiguous copies.
- Require contiguous source and destination in `ops::add_inplace`, in
  addition to its existing device, shape, and dtype checks.
- Realize deferred operands before raw-pointer mutation or copying.
- Apply the locked `fill` decision. Under this plan's conservative default,
  reject a non-contiguous destination with an error that names the operation
  and suggests `contiguous()` when a detached packed tensor is intended.
- Keep `zeros_like`/`ones_like`/`full_like` fresh and canonical, even when the
  source is a view.

Checkpoint:

- Every unsupported source/destination combination throws before writing,
  and existing contiguous copy, fill, and repeated gradient accumulation
  tests remain green.

### 6. Add autograd-aware transpose and permute

Primary files:

- `csrc/include/lamp3/autograd/functions/view_ops.hpp`
- `csrc/src/autograd/functions/view_ops.cpp`
- `csrc/include/lamp3/autograd/functions/matrix_ops.hpp`
- `csrc/src/autograd/functions/matrix_ops.cpp`

Changes:

- Extend transpose forward/backward state with `dim0` and `dim1`; backward
  applies the same swap.
- Add permute forward state and inverse-permutation backward state.
- Preserve the no-dimension transpose overload by deriving the final two
  dimensions after rank validation.
- Before passing a transpose/permute view to `Variable::incr_grad`,
  materialize it because gradient accumulation is contiguous-only in V1.
- Leave reshape backward as reshape to the saved input shape; a forward
  materialization is logically an identity.
- Retain squeeze/expand-dimensions as inverse operations, now relying on
  stride-preserving tensor implementations.

Checkpoint:

- Transpose of arbitrary axes, permute/inverse-permute, squeeze or
  expand-dimensions after permutation, and reshape after permutation produce
  correct fresh contiguous leaf gradients.

### 7. Expose the layout API in Python

Primary files:

- `csrc/bindings/tensor.hpp`
- `csrc/bindings/variable.hpp`
- `csrc/bindings/functions/view.hpp`
- `csrc/bindings/functions/matrix.hpp`

Changes:

- Expose read-only `strides` and `is_contiguous` properties on `_Tensor`.
- Bind `_Tensor.contiguous()`, `_Tensor.transpose(dim0, dim1)`,
  `_Tensor.permute(dims)`, and `_Tensor.T`.
- Bind autograd-aware `Variable.transpose(dim0, dim1)` and
  `Variable.permute(dims)`, while retaining `Variable.T` and its no-argument
  final-two-dimension behavior.
- Expose module-level explicit-axis transpose and permute overloads without
  making pybind choose between ambiguous C++ overloads implicitly.
- Keep `tolist` flat and row-major, but make its values follow logical view
  order through step 4.

Checkpoint:

- C++ and Python surfaces report identical shape, strides, contiguity,
  logical values, and validation errors.

### 8. Install fusion barriers and defensive checks

Primary files:

- `csrc/src/tensor/native/unary_ops.cpp`
- `csrc/src/tensor/native/expand_ops.cpp`
- `csrc/src/inductor/nvrtc/fused_graph.cpp`
- `csrc/src/inductor/nvrtc/codegen.cpp`

Changes:

- Gate unary capture on `a.is_contiguous()`.
- Gate binary capture on all would-be fused leaves being contiguous, in
  addition to the current device/shape fusibility rules.
- Let a non-contiguous consumer take the eager path. Correct strided eager
  kernels are delivered by subset 3; subset 1's responsibility is that the
  flat generated path is never selected.
- Assert at fused-graph construction/code-generation boundaries that every
  realized input leaf is contiguous before emitting `input[i]`.
- Do not add stride metadata or offset logic to generated NVRTC kernels.

Checkpoint:

- A view of a deferred tensor realizes before sharing storage.
- A non-contiguous leaf cannot be emitted into a flat fused kernel.
- After subset 3 lands, add the integration assertion that an eager strided
  consumer returns a contiguous tensor from which a new fusion graph can
  begin.

### 9. Update user documentation

Primary file:

- `docs/using_tensor.dox`

Changes:

- Replace statements that Lamp3 does not support non-contiguous tensors.
- Document shape/stride layout, size-one contiguity, view aliasing,
  `is_contiguous`, `contiguous`, arbitrary-axis transpose, permute, and the
  reshape copy boundary.
- Correct the current statement that transpose allocates.
- Document that exports and device transfer preserve logical order, while
  unsupported mutation of non-contiguous tensors fails.
- State the V1 limits: no slicing, negative/zero strides, storage offsets,
  overlapping views, or non-contiguous write destinations.

## Test plan

Prefer dedicated layout test executables/files rather than adding all cases
to the already broad `tensor_tests.cpp` and `autograd_tests.cpp`:

- `csrc/tests/tensor_layout_tests.cpp`
- `csrc/tests/autograd_layout_tests.cpp`
- `tests/unit/test_tensor_layout.py`
- corresponding targets in `csrc/tests/CMakeLists.txt`

Run all storage-access cases on CPU and CUDA; metadata-only validation can run
once on CPU.

### Metadata and aliasing

- Canonical constructor strides across ranks, empty dimensions, and
  size-one dimensions.
- Every transpose axis pair on a rank-four tensor.
- Invalid transpose dimensions.
- Double transpose restores shape and strides.
- Valid permute and inverse permute.
- Missing, repeated, extra, and out-of-range permutation dimensions.
- Shared storage for transpose, permute, squeeze, expand-dimensions, and
  contiguous reshape.
- Exact stride preservation/removal/insertion for squeeze and expand.
- Nontrivial permutation is non-contiguous; size-one-only permutation stays
  contiguous.

### Materialization and reads

- Contiguous fast path returns the same public handle/storage.
- Transpose and higher-rank permute materialize into canonical strides and
  logical row-major values.
- Materialization allocates independent storage and does not modify aliases.
- Contiguous reshape shares storage; non-contiguous reshape allocates and
  preserves logical order.
- `index`, C++ `to_vector`, Python `tolist`, printing, and device transfer all
  observe logical order.
- Empty tensors avoid invalid division or kernel launches.

### Mutation

- Contiguous same-device, cross-device, and dtype-converting copy regressions.
- Non-contiguous source and destination copy rejection.
- Non-contiguous source and destination `add_inplace` rejection.
- Locked `fill` behavior, including alias preservation after rejection.
- Failure paths leave both source and destination bytes unchanged.

### Lazy/fusion lifecycle

- Transpose, permute, squeeze, expand-dimensions, and contiguous reshape of a
  deferred tensor realize the original impl before creating the view.
- Already-contiguous public `contiguous()` does not fork a deferred impl.
- Capture enabled plus a non-contiguous input never constructs a generated
  flat leaf.
- Backend invariant check rejects any manually or accidentally introduced
  non-contiguous leaf.
- Cross-subset integration, enabled with subset 3: the eager strided result is
  contiguous and a following operation can start a new fusion region.

### Autograd

- Arbitrary-axis transpose backward.
- Permute backward with a non-self-inverse permutation.
- Squeeze and expand-dimensions backward after permutation.
- Reshape after non-contiguous view backward.
- Gradients and gradient buffers are canonical and repeated accumulation
  remains correct.

### Regression commands

CPU:

```bash
cmake -S . -B build/subset1-cpu \
  -DLMP_ENABLE_TEST=ON \
  -DLMP_ENABLE_CUDA=OFF
cmake --build build/subset1-cpu -j
ctest --test-dir build/subset1-cpu --output-on-failure
pytest tests/unit/test_tensor_layout.py tests/unit
```

CUDA, on a CUDA runner:

```bash
cmake -S . -B build/subset1-cuda \
  -DLMP_ENABLE_TEST=ON \
  -DLMP_ENABLE_CUDA=ON
cmake --build build/subset1-cuda -j
ctest --test-dir build/subset1-cuda --output-on-failure
pytest tests/unit/test_tensor_layout.py tests/graphs/test_graphs.py
```

Also build documentation and run the repository's formatter/linter over
touched C++ and CUDA files before handoff.

## Reviewable change slices

Keep each slice green for the behavior it introduces:

1. Layout invariants, metadata views, and metadata tests.
2. Native/public APIs and removal of physical transpose dispatch.
3. CPU/CUDA contiguous materialization and logical-read routing.
4. Mutation guards and regression tests.
5. Autograd functions and bindings.
6. Fusion gates, lifecycle tests, and documentation.

The materialization slice starts only after subset 2's `OffsetUtil<1>` contract
is available. The final fusion-consumer integration assertion becomes
required when subset 3 is merged.

## Risks and mitigations

- **Deferred impl copies can orphan realization state.** Realize before every
  storage-sharing impl copy and return the same public handle from the
  contiguous fast path.
- **A stray stride reset silently destroys a view.** Limit canonical stride
  recomputation to allocation and contiguous reshape, and test exact strides
  after chained views.
- **Flat raw copies expose physical order.** Centralize logical packing
  through `contiguous()` for export, printing, and transfer.
- **Fusion correctness can regress at a future call site.** Gate capture at
  native entry points and assert contiguity again in the fused backend.
- **Autograd views conflict with contiguous-only accumulation.** Pack view
  gradients immediately before `incr_grad`, while keeping gradient buffers
  freshly allocated and canonical.
- **Subset boundaries can hide incomplete operator behavior.** Keep
  subset-3-dependent tests labeled as cross-subset integration and do not
  claim the full V1 feature complete after subset 1 alone.
