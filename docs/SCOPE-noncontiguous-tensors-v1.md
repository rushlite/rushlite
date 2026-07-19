# Scope: Non-Contiguous Tensors V1

## Purpose

This document scopes the first version of layout-aware tensors in Lamp3.

The work is divided into three subsets:

1. Tensor metadata, views, contiguous materialization, and layout lifecycle.
2. A generalized logical-index-to-storage-offset utility.
3. Layout-aware unary, binary, reduction, and matrix multiplication kernels.

The three subsets are separated so that their contracts and tests can be
reviewed independently. They still form one feature: metadata-only views are
not usable until their consumers honor strides, and strided kernels need a
well-defined tensor layout model.

This document supersedes the non-contiguous-tensor portions of
`HANDOFF-batched-matmul-noncontiguous-tensors.md`. The batched matmul context
from that handoff remains relevant.

## Terminology

### Logical order

The row-major order implied by a tensor's shape, independent of how its
elements are arranged in storage.

For a tensor with shape `{d0, d1, ..., dn}`, a logical linear index is first
decomposed into coordinates under that shape.

### Physical offset

The storage element reached by applying a tensor's strides to logical
coordinates:

```text
physical_offset = sum(coordinate[dim] * stride[dim])
```

V1 always uses storage offset zero.

### Contiguous tensor

A tensor whose logical row-major order matches its physical storage order.
Strides on size-one dimensions do not affect contiguity because the only
coordinate on those dimensions is zero.

### Non-contiguous tensor

In V1, a dense view whose shape and strides describe a permutation of the
entire underlying storage. Every storage element is still reachable exactly
once.

### Lazy realization

Execution of a deferred operation or fused graph so that a tensor has
physical storage.

### Contiguous materialization

Allocation of new storage followed by copying a tensor's values from logical
order into canonical row-major physical order.

Lazy realization and contiguous materialization are different operations.
A realized tensor can still be non-contiguous.

## V1 Contract

### In scope

- Metadata-only transpose between any two axes.
- Arbitrary full-dimension permutation.
- Dense non-contiguous views produced by transpose or permutation.
- Stride-preserving `squeeze` and `expand_dims`.
- `is_contiguous`.
- Explicit contiguous materialization.
- Automatic contiguous materialization when reshaping a non-contiguous
  tensor.
- Logical reads from non-contiguous inputs in:
  - Unary operations.
  - Binary operations.
  - Reductions.
  - Rank-two matrix multiplication.
  - Batched matrix multiplication.
- Broadcasting in binary operations.
- Broadcasting over matmul batch dimensions.
- Contiguous outputs from all computational operations.
- CPU and CUDA implementations.
- Non-contiguous tensors as lazy-fusion barriers.
- Correct autograd behavior for the operations above.

### Out of scope

- Slicing and narrowing.
- A per-tensor storage offset.
- Views with gaps in their reachable storage.
- Negative strides.
- Zero-stride expanded views.
- Overlapping views.
- Non-contiguous write destinations.
- General reshape-viewability analysis for non-contiguous layouts.
- Stride-aware generated fusion kernels.
- One-dimensional `torch.matmul` promotion rules.
- Materializing broadcasted inputs before an operation.

### Output layout policy

The following operations return fresh contiguous tensors:

- Unary operations.
- Binary operations.
- Reductions.
- Matmul.
- `contiguous()` when its input is non-contiguous.
- `reshape()` when its input is non-contiguous.
- Device transfer.

The following operations return storage-sharing views:

- `transpose`.
- `permute`.
- `squeeze`.
- `expand_dims`.
- `reshape` when its input is already contiguous.

### Mutation policy

V1 does not add generalized strided writes.

Operations that require elementwise correspondence between a source and a
destination, including `copy` and `add_inplace`, require contiguous operands.
They must not silently interpret a non-contiguous tensor in physical storage
order.

`Tensor::copy` is currently public in both C++ and Python even though it is
used as an internal primitive. Its rejection of non-contiguous operands is
therefore observable behavior.

The policy for `fill` must be locked before implementation. A flat fill is
logically correct for V1 dense permutation views because every storage element
is represented exactly once, but it also mutates every storage alias. Slicing
would invalidate that shortcut later.

### Fusion policy

Generated fusion kernels continue to assume flat contiguous inputs.

- Constructing a storage-sharing view from a deferred tensor first realizes
  the deferred tensor.
- A non-contiguous input is not recorded as a leaf in a fused elementwise
  graph.
- An operation consuming a non-contiguous tensor executes through the eager
  stride-aware path.
- That operation produces a contiguous tensor.
- Subsequent operations may start a new fusion graph from the contiguous
  result.

The barrier check is based on `is_contiguous()`, not on whether a transpose or
permute operation occurred. Permutations involving only size-one dimensions
can still be contiguous.

## Subset 1: Tensor Layout, Views, and Materialization

### Goal

Make shape and stride metadata independently meaningful, expose metadata-only
views, and define when a tensor is packed into contiguous storage.

### Primary files

- `csrc/include/lamp3/tensor/tensor_impl.hpp`
- `csrc/src/tensor/tensor_impl.cpp`
- `csrc/include/lamp3/tensor/tensor.hpp`
- `csrc/src/tensor/tensor.cpp`
- `csrc/include/lamp3/tensor/native/shape_ops.hpp`
- `csrc/src/tensor/native/shape_ops.cpp`
- `csrc/include/lamp3/tensor/native/matrix_ops.hpp`
- `csrc/src/tensor/native/matrix_ops.cpp`
- `csrc/bindings/tensor.hpp`
- `csrc/bindings/variable.hpp`
- `csrc/bindings/functions/view.hpp`
- `csrc/bindings/functions/matrix.hpp`

### Supporting files

- `csrc/src/tensor/native/memory_ops.cpp`
- `csrc/include/lamp3/tensor/native/memory_ops.hpp`
- `csrc/src/autograd/functions/view_ops.cpp`
- `csrc/include/lamp3/autograd/functions/view_ops.hpp`
- `csrc/src/autograd/functions/matrix_ops.cpp`
- `csrc/include/lamp3/autograd/functions/matrix_ops.hpp`
- `csrc/src/tensor/native/unary_ops.cpp`
- `csrc/src/tensor/native/expand_ops.cpp`
- `csrc/src/inductor/nvrtc/fused_graph.cpp`
- `csrc/src/inductor/nvrtc/codegen.cpp`
- `docs/using_tensor.dox`

### TensorImpl API

The layout API needs, at minimum:

```cpp
bool is_contiguous() const noexcept;

TensorImpl transpose(size_t dim0, size_t dim1);
TensorImpl permute(const std::vector<size_t>& dims);

TensorImpl reshape(std::vector<size_t> new_shape);
TensorImpl squeeze(size_t dim);
TensorImpl expand_dims(size_t dim);

TensorImpl contiguous() const;
```

Equivalent public `Tensor` operations are needed for user-facing behavior.
The public `Tensor::contiguous()` wrapper returns the original tensor handle
when it is already contiguous. Otherwise it wraps the newly materialized
`TensorImpl`.

### Public API and bindings

The tensor API needs:

```cpp
bool Tensor::is_contiguous() const noexcept;
Tensor Tensor::contiguous() const;
Tensor Tensor::transpose(size_t dim0, size_t dim1) const;
Tensor Tensor::permute(const std::vector<size_t>& dims) const;
```

The native operation layer needs matching free functions. The existing
no-dimension transpose free function remains as the final-two-dimensions
convenience form.

Python bindings expose:

- Tensor strides.
- Tensor contiguity.
- `contiguous()`.
- `transpose(dim0, dim1)`.
- `permute(dims)`.
- Equivalent autograd-aware transpose and permute operations on `Variable`.

The existing `.T` property retains the no-dimension transpose behavior.

### Canonical strides

Constructors for newly allocated tensors continue to create canonical
row-major strides:

```text
expected = 1
for dimensions from last to first:
    stride[dim] = expected
    expected *= shape[dim]
```

The current `update_strides()` behavior is valid only when establishing a
known-contiguous layout. It must not be called after a general view operation.
Renaming it to express that restriction is optional; changing its call sites
is required.

### Storage invariant

V1 retains the current whole-storage invariant:

- `numel` is the product of the logical shape.
- A realized tensor's storage capacity is exactly `numel` elements.
- The tensor's storage offset is zero.
- A dense permutation reaches every storage element exactly once.

Transpose, permute, squeeze, and expand-dimensions do not change `numel` or
storage capacity. Supporting slices later will require replacing this
invariant with reachable-offset validation.

### Contiguity check

The contiguity check walks dimensions from last to first:

```text
if numel == 0:
    return true

expected = 1

for dim from rank - 1 to 0:
    if shape[dim] == 1:
        continue
    if strides[dim] != expected:
        return false
    expected *= shape[dim]

return true
```

An empty tensor is contiguous because no logical element violates the layout.

### Transpose

`transpose(dim0, dim1)`:

1. Validates both dimensions.
2. Copies the `TensorImpl`.
3. Continues sharing the same `Storage`.
4. Swaps `shape_[dim0]` and `shape_[dim1]`.
5. Swaps `strides_[dim0]` and `strides_[dim1]`.
6. Does not recalculate any other stride.

The current CPU and CUDA physical transpose kernels are no longer the
implementation of the public transpose operation. Contiguous materialization
of a transposed view is handled by the generalized strided copy path.

The existing no-dimension transpose API needs compatibility semantics. The
assumed V1 contract is:

- `transpose(a, dim0, dim1)` performs an arbitrary two-axis swap.
- The existing `transpose(a)` convenience operation swaps the final two axes.
- Rank must be at least two for the no-dimension form.

### Permute

`permute(dims)`:

1. Requires one entry for every input dimension.
2. Requires each dimension to be in range.
3. Requires every dimension to appear exactly once.
4. Reorders both shape and stride metadata using the same permutation.
5. Shares storage with the input.

The autograd inverse is the inverse permutation.

### Squeeze

`squeeze(dim)`:

1. Validates the dimension.
2. Requires `shape[dim] == 1`.
3. Erases the shape entry.
4. Erases the matching stride entry.
5. Does not recalculate the remaining strides.

### Expand dimensions

`expand_dims(dim)` inserts a size-one shape entry. The V1 stride convention
is:

```text
if dim < old_rank:
    inserted_stride = old_strides[dim] * old_shape[dim]
else:
    inserted_stride = 1
```

The inserted stride does not change addressing while the new dimension
remains size one. `is_contiguous()` therefore ignores it.

This operation is not broadcast expansion. It does not create a zero stride.

### Reshape

`reshape(new_shape)` still validates that the logical element count is
unchanged.

Its behavior depends on layout:

```text
contiguous input:
    share storage
    apply new shape
    create canonical strides for the new shape

non-contiguous input:
    materialize input in logical order
    apply the contiguous reshape behavior to the materialized tensor
```

V1 does not attempt to detect non-contiguous layouts that could technically be
reshaped without a copy.

### Contiguous materialization

For a non-contiguous input:

1. Allocate storage for `numel` elements on the same device and with the same
   dtype.
2. Create an output `TensorImpl` with the same shape and canonical strides.
3. Iterate over the output's logical linear indices.
4. Use the unary offset transform to find the corresponding input storage
   offset.
5. Copy the input element to `output[logical_index]`.

This is a dtype-preserving strided copy kernel. The existing raw `copy_stub`
cannot implement it because that stub receives pointers and a flat element
count but no layout metadata.

For a contiguous `Tensor`, `contiguous()` returns the same tensor handle or an
equivalent storage-sharing handle without copying data.

A deferred contiguous tensor must not be duplicated into two independent
`TensorImpl` objects that merely share the same pending `LazyFunction`.
Returning the same public `Tensor` handle avoids that ownership problem.
Internal code that directly calls `TensorImpl::contiguous()` on a deferred
contiguous impl must realize that impl before returning a copied impl.

### View creation and lazy tensors

The current lazy representation attaches a pending operation to one
`TensorImpl`. Copying that `TensorImpl` copies the pending-operation pointer,
but realizing one copy does not update the other copy's storage.

Therefore, before V1 creates a new storage-sharing `TensorImpl` view from a
deferred input, it realizes the input. The view then shares the realized
`Storage`.

This applies to transpose, permute, squeeze, expand-dimensions, and contiguous
reshape views. It is a view-lifetime constraint in addition to the
non-contiguous-consumer fusion barrier.

### Logical materialization consumers

The following paths currently copy flat physical storage and must instead
preserve logical order:

- `Tensor::to_vector`.
- Python `tolist`.
- Tensor printing.
- `Tensor::to(device)`.

Under the V1 mutation policy:

- Read-only export and transfer paths may materialize a non-contiguous source.
- `copy` rejects a non-contiguous source or destination rather than adding
  strided reads or writes.
- `add_inplace` rejects a non-contiguous destination or source.

### Autograd view behavior

Transpose backward must store the swapped dimensions and apply the same swap
to the incoming gradient.

Permute backward must store or derive the inverse permutation.

Squeeze and expand-dimensions backward already use their inverse shape
operation. Their tensor implementations must preserve strides correctly.

Reshape backward remains a logical reshape back to the input shape. If the
forward input was non-contiguous, the forward materialization is logically an
identity and does not require a separate gradient formula.

Gradient buffers created by `zeros_like` remain fresh and contiguous even
when the variable's data is non-contiguous.

If `add_inplace` does not accept a non-contiguous source, any backward formula
that produces a view must call `contiguous()` before `Variable::incr_grad`.
Transpose backward is the immediate case.

### Fusion gating

Unary and binary native entry points currently record operations whenever
capture is enabled. Their capture condition must also require that all leaf
inputs used by the generated kernel are contiguous.

A non-contiguous consumer executes eagerly and returns a contiguous output.
No stride metadata is added to NVRTC code generation in V1.

The fused-graph builder or backend should also assert that every realized leaf
is contiguous. Native capture gating is the normal control path; the backend
check prevents a future caller from accidentally generating `input[i]` loads
for a strided leaf.

### Subset 1 acceptance tests

Run layout tests on CPU and CUDA where storage access is involved.

Metadata:

- Constructor creates canonical strides.
- Transpose swaps exactly two shape and stride entries.
- Transpose supports every valid pair of axes on a higher-rank tensor.
- Invalid transpose dimensions are rejected.
- Double transpose restores shape and strides.
- Permute reorders shape and strides.
- Inverse permute restores shape and strides.
- Invalid, missing, and repeated permutation dimensions are rejected.
- Transpose and permute share storage with their source.
- Squeeze erases the matching shape and stride.
- Expand-dimensions inserts a size-one dimension without changing values.
- Squeeze after expand-dimensions restores the prior layout.
- Size-one dimensions do not make an otherwise contiguous tensor
  non-contiguous.
- A nontrivial dense permutation is reported as non-contiguous.

Materialization:

- `contiguous()` is a no-op for an already contiguous tensor.
- `contiguous()` of a transpose produces canonical strides.
- Materialized values are in logical order.
- Materialization does not modify the source storage.
- Reshape of a contiguous tensor shares storage.
- Reshape of a non-contiguous tensor allocates new storage.
- Reshape of a non-contiguous tensor preserves logical order.

External reads:

- `index()` continues to honor strides.
- `tolist` returns logical order for a transpose and higher-rank permutation.
- Printing uses logical order.
- Device transfer preserves logical order and returns a contiguous tensor.
- Unsupported non-contiguous `copy` and in-place combinations fail clearly.

Lazy behavior:

- A view of a deferred tensor realizes the pending graph before sharing
  storage.
- A non-contiguous input is not consumed by a flat generated kernel.
- A strided eager consumer returns a contiguous result.
- Operations after that result can form a new fusion graph.

Autograd:

- Transpose backward uses the original swapped axes.
- Permute backward uses the inverse permutation.
- Squeeze and expand-dimensions preserve correct gradients after a
  permutation.
- Reshape after a non-contiguous view produces correct gradients.

## Subset 2: Logical Offset Transformation

### Goal

Provide one coordinate transformation model that supports:

- Unary reads from a strided input.
- Binary broadcasting and strided inputs.
- Reduction base-address calculation.
- Matmul batch broadcasting and batch-base calculation.

The utility maps logical iteration indices to physical input offsets. V1 does
not require it to map writes into non-contiguous destinations.

### Primary files

- `csrc/include/lamp3/tensor/cpu/offset_util.hpp`
- `csrc/src/tensor/cpu/offset_util.cpp`
- `csrc/include/lamp3/tensor/cuda/offset_util.cuh`
- `csrc/src/tensor/cuda/offset_util.cu`
- `csrc/src/tensor/cpu/meta_handler.cpp`
- `csrc/include/lamp3/tensor/cpu/meta_handler.hpp`
- `csrc/include/lamp3/tensor/infer_meta.hpp`
- `csrc/src/tensor/infer_meta.cpp`

### Core abstraction

The utility should be defined in terms of:

1. A logical iteration shape.
2. Canonical logical strides derived from that iteration shape.
3. One effective physical-stride list per input operand.

Conceptually:

```cpp
struct OperandLayout {
  std::vector<size_t> shape;
  std::vector<stride_t> strides;
};

template <size_t NArgs>
class OffsetUtil {
 public:
  OffsetUtil(std::vector<size_t> iteration_shape,
             std::array<OperandLayout, NArgs> operands);

  offsets_t<NArgs> get(size_t logical_index) const;
};
```

This is a conceptual API, not a required class spelling.

The important difference from the current utility is that the iteration shape
is explicit. It is not inferred indirectly from a physical output layout.
Normal outputs are contiguous, and matmul needs a synthetic batch-only
iteration space that is not itself a complete tensor.

### MetaHandler integration

The tensor meta handlers continue to allocate fresh contiguous outputs.
Offset ownership changes as follows:

- Unary constructs `OffsetUtil<1>` when taking the non-contiguous route.
- Binary always constructs `OffsetUtil<2>` because V1 removes the separate
  flat binary route.
- Reduction constructs `OffsetUtil<1>` when taking the non-contiguous route.
- Contiguous materialization constructs `OffsetUtil<1>`.
- Matmul owns a batch-only `OffsetUtil<2>` independently of the elementwise
  meta handlers.

The offset accessor can no longer assert that binary shape expansion occurred.
Its invariant becomes that an offset utility was constructed for the selected
route.

### Effective input strides

Input ranks align to the logical iteration rank from the right.

For each aligned dimension:

- A missing input dimension receives effective stride zero.
- A size-one input dimension broadcast over a larger iteration dimension
  receives effective stride zero.
- Otherwise the effective stride is the input's real physical stride.

The operator's metadata inference remains responsible for determining whether
shapes are semantically compatible. The offset utility performs mapping; it
does not define the broadcasting or reduction contract.

For reduction, the logical iteration shape is the keep-dimension output
shape. The reduced coordinate is therefore always zero. The input's real
reduced-axis stride is used separately by the reduction loop.

For matmul, the logical iteration shape contains only the output batch
dimensions. Operand layouts contain only each operand's leading batch shape
and leading batch strides.

### Logical index decomposition

For every logical iteration index:

```text
remaining = logical_index

for dim from first to last:
    coordinate = remaining / logical_stride[dim]
    remaining %= logical_stride[dim]

    for each input:
        input_offset += coordinate * effective_input_stride[input][dim]
```

The output write offset remains `logical_index` because V1 computational
outputs are contiguous.

### Required arities

- `OffsetUtil<1>` for unary operations, contiguous materialization, and
  reduction base offsets.
- `OffsetUtil<2>` for binary operations and matmul batch bases.
- Preserve `OffsetUtil<3>` if existing users or generated instantiations still
  require it.

Both CPU and CUDA dispatch registrations and explicit template
instantiations must exist for every supported arity.

### CPU representation

The CPU implementation may retain host vectors for logical and effective
strides.

Its `get()` operation must:

- Be safe to call concurrently from OpenMP loops.
- Perform no mutation.
- Return offsets using `stride_t`.
- Use the same mapping rules as CUDA.

### CUDA representation

The CUDA implementation needs a device-usable representation of:

- Rank.
- Canonical logical strides.
- Effective strides for every input.

Whichever representation is used must satisfy:

- All metadata remains alive until the kernel finishes.
- Copying the device-side utility does not leave invalid host pointers or
  ownership objects inside device memory.
- Rank is bounded by `LMP_MAX_DIMS`.
- `get()` is callable from unary, binary, reduction, materialization, and
  matmul kernels.
- The same utility can be constructed over full tensor dimensions or only
  leading batch dimensions.

V1 does not require offset calculation to be vectorized.

### Matmul usage

The offset utility is not invoked for every scalar multiplication.

For an output batch index:

```text
batch_offsets = batch_offset_util.get(output_batch)
a_base = batch_offsets.a
b_base = batch_offsets.b
```

The matmul kernel then uses explicit matrix strides from those bases.

For rank-two matmul, both batch shapes are empty and both bases are zero. This
case can bypass the batch utility. A batch-shape product must treat an empty
leading shape as one matrix, not as a zero-element tensor.

### Validation boundaries

The utility may assert structural conditions:

- Shape and stride ranks match for every operand.
- Operand rank does not exceed iteration rank after the operator has applied
  its alignment rules.
- Rank does not exceed `LMP_MAX_DIMS`.
- A non-empty iteration space has no zero logical-stride divisors.

An empty iteration space launches no kernel and never calls `get()`.

Semantic validation stays with the operation:

- Binary broadcast compatibility.
- Reduction axis validity.
- Matmul inner-dimension compatibility.
- Matmul batch broadcast compatibility.

### Offset utility acceptance tests

CPU and CUDA must produce identical offsets for:

- A contiguous tensor.
- A rank-two transpose.
- A higher-rank permutation.
- A unary iteration over a strided input.
- Two matching non-contiguous binary inputs.
- One contiguous and one non-contiguous binary input.
- Row, column, scalar, and missing-leading-dimension broadcasting.
- Broadcasting combined with a permutation.
- A keep-dimension reduction projection.
- Matmul with no batch dimensions.
- Matmul with matching batch dimensions.
- Matmul with a shared rank-two operand.
- Matmul with multiple broadcasted batch dimensions.
- Matmul whose batch dimensions have non-canonical physical strides.
- Maximum supported rank.

Offset tests should use deliberately distinct coordinate values so that a
wrong dimension order cannot accidentally produce the expected result.

## Subset 3: Layout-Aware Operators and Kernels

### Goal

Make all computational consumers produce correct contiguous results from
contiguous, broadcasted, and dense non-contiguous inputs.

This subset contains four operator families:

1. Unary.
2. Binary.
3. Reduction.
4. Matmul.

### Shared kernel contract

Every kernel in this subset:

- Treats its output iteration index as a logical row-major index.
- Writes to a fresh contiguous output.
- Reads each input through either:
  - Direct flat addressing on an explicitly retained contiguous fast path; or
  - The generalized offset utility and explicit operation-specific strides.
- Never mutates an input.
- Never materializes a broadcasted or transposed input merely to read it.

### Unary operations

#### Files

- `csrc/src/tensor/cpu/unary.cpp`
- `csrc/include/lamp3/tensor/cpu/unary.hpp`
- `csrc/src/tensor/cuda/unary.cu`
- `csrc/include/lamp3/tensor/cuda/unary.cuh`
- `csrc/src/tensor/cpu/kernels.cpp`
- `csrc/src/tensor/cuda/kernels.cu`
- `csrc/src/tensor/cpu/meta_handler.cpp`

#### Behavior

Unary output shape and dtype inference are unchanged.

For a non-contiguous input:

```cpp
for (size_t i = global_index; i < output.numel(); i += global_stride) {
  auto offsets = offset_util.get(i);
  output[i] = fn(input[offsets.input]);
}
```

This applies to:

- Negation.
- Exponential.
- Logarithm.
- Square root.
- Absolute value.
- Sine.
- Cosine.
- Tangent.
- Clamp.

The current direct contiguous unary kernel may remain as a separate route.
The route condition must be based on `input.is_contiguous()`.

Unary lazy recording is allowed only when the input is contiguous.

#### Unary tests

- Every unary functor on a rank-two transpose.
- Every unary functor after a higher-rank permutation.
- Clamp on a non-contiguous tensor.
- Output is contiguous.
- Input shape, strides, storage, and values remain unchanged.
- Contiguous regression tests.
- CUDA aligned and unaligned storage cases where applicable.

### Binary operations

#### Files

- `csrc/src/tensor/cpu/binary.cpp`
- `csrc/include/lamp3/tensor/cpu/binary.hpp`
- `csrc/src/tensor/cpu/expand.cpp`
- `csrc/include/lamp3/tensor/cpu/expand.hpp`
- `csrc/src/tensor/cuda/binary.cu`
- `csrc/include/lamp3/tensor/cuda/binary.cuh`
- `csrc/src/tensor/cuda/expand.cu`
- `csrc/include/lamp3/tensor/cuda/expand.cuh`
- `csrc/src/tensor/cpu/kernels.cpp`
- `csrc/src/tensor/cuda/kernels.cu`
- `csrc/src/tensor/cpu/meta_handler.cpp`
- `csrc/src/tensor/infer_meta.cpp`
- `csrc/src/tensor/CMakeLists.txt`

#### Unified path

The current split is:

```text
same shapes      -> binary kernel with flat reads
different shapes -> expand kernel with OffsetUtil<2>
```

V1 replaces that split with one offset-aware binary path:

```cpp
for (size_t i = global_index; i < output.numel(); i += global_stride) {
  auto offsets = offset_util.get(i);
  output[i] = fn(a[offsets.a], b[offsets.b]);
}
```

The generalized path handles all combinations:

- Same-shape contiguous inputs.
- Same-shape non-contiguous inputs.
- One contiguous and one non-contiguous input.
- Broadcasting.
- Broadcasting of a non-contiguous input.

The old `expand_` routing flag no longer describes the execution decision.
The old expand kernel can be renamed and generalized or merged into the
binary kernel. Dead source files, headers, includes, and CMake entries should
be removed after the unified path is established.

This scope intentionally removes the separate flat/vectorized binary route,
including its vec4 specialization. Performance changes from integer
coordinate decomposition are measured but do not change the V1 correctness
contract.

Internal binary-shaped backward kernels must use the same path:

- Absolute-value backward.
- Clamp backward.
- Any other helper that currently calls `binary_dispatch_handler` directly.

Binary lazy recording is allowed only when all inputs that would become fused
leaves are contiguous. Ordinary broadcasting remains a non-fusible eager path
under the current lazy contract.

#### Binary tests

Run arithmetic, comparison, and backward-helper functors over:

- Two matching transposes.
- A transpose and a contiguous tensor with the same logical shape.
- Differently permuted tensors with the same logical shape.
- A broadcasted contiguous operand and a non-contiguous operand.
- A broadcasted non-contiguous operand.
- Multiple leading broadcast dimensions.
- Mixed dtypes.
- Output contiguity.
- Source metadata and storage preservation.
- Existing contiguous and broadcasting regressions.

### Reduction operations

#### Files

- `csrc/src/tensor/cpu/reduct.cpp`
- `csrc/include/lamp3/tensor/cpu/reduct.hpp`
- `csrc/src/tensor/cuda/reduct.cu`
- `csrc/include/lamp3/tensor/cuda/reduct.cuh`
- `csrc/src/tensor/cpu/meta_handler.cpp`
- `csrc/src/tensor/infer_meta.cpp`
- `csrc/src/tensor/cpu/kernels.cpp`
- `csrc/src/tensor/cuda/kernels.cu`

#### Output shape

Reduction remains keep-dimension:

```text
output.shape = input.shape
output.shape[axis] = 1
```

The output is freshly allocated and contiguous.

#### Strided reduction algorithm

For each contiguous output index:

```cpp
for (size_t i = global_index; i < output.numel(); i += global_stride) {
  auto offsets = offset_util.get(i);
  stride_t input_base = offsets.input;

  auto value = OpFn::kIdentity;
  for (size_t j = 0; j < input.shape()[axis]; ++j) {
    value = fn(
        value,
        input[input_base + j * input.strides()[axis]]);
  }

  output[i] = value;
}
```

The unary offset utility maps the output coordinates to the corresponding
input coordinates with the reduced coordinate fixed at zero. The inner loop
walks the reduced dimension using its real input stride.

No transpose-back operation is performed, and the reduction result does not
retain the input's non-contiguous layout.

#### Routing

The existing contiguous reduction formula may remain as a separate route.
The strided path is selected whenever the input is non-contiguous.

Whether both paths remain after benchmarking is not part of the correctness
contract. Unlike binary, removal of the contiguous reduction path has not yet
been chosen.

#### Validation

- Axis must be less than rank before shape inference indexes it.
- Input shape and stride ranks must match.
- Zero-length reduced dimensions follow the existing functor identity
  behavior.

#### Reduction tests

For sum, minimum, maximum, and product:

- Every axis of a transposed rank-two tensor.
- Every axis of several rank-three permutations.
- Size-one reduced dimensions.
- A permutation containing size-one dimensions.
- Keep-dimension output shape.
- Canonical output strides.
- Input preservation.
- Contiguous regression tests.
- Autograd reduction formulas consuming or producing transposed views.

At least one test must use a permutation where the current
`outer/inner/index` formula produces a different output order. For example:

```text
shape:   [3, 4, 2]
strides: [4, 1, 12]
reduce axis 1
```

### Matrix multiplication

#### Files

- `csrc/src/tensor/native/matrix_ops.cpp`
- `csrc/include/lamp3/tensor/native/matrix_ops.hpp`
- `csrc/src/tensor/cpu/kernels.cpp`
- `csrc/src/tensor/cpu/matrix.cpp`
- `csrc/include/lamp3/tensor/cpu/matrix.hpp`
- `csrc/src/tensor/cuda/kernels.cu`
- `csrc/src/tensor/cuda/matrix.cu`
- `csrc/include/lamp3/tensor/cuda/matrix.cuh`
- `csrc/src/autograd/functions/matrix_ops.cpp`
- `csrc/include/lamp3/autograd/functions/matrix_ops.hpp`
- `csrc/src/autograd/utils/grad_utils.cpp`

#### Shape contract

V1 matmul:

- Requires both operands to have rank at least two.
- Treats the final two dimensions as matrix dimensions.
- Requires `a.shape[-1] == b.shape[-2]`.
- Broadcasts only the leading batch dimensions.
- Does not implement one-dimensional vector promotion.

Examples:

```text
[M, K] @ [K, N]               -> [M, N]
[B, M, K] @ [B, K, N]         -> [B, M, N]
[B, M, K] @ [K, N]            -> [B, M, N]
[4, 1, M, K] @ [1, 7, K, N]   -> [4, 7, M, N]
```

The output shape is:

```text
broadcast(a.batch_shape, b.batch_shape) + [M, N]
```

The output is contiguous.

Matmul needs dedicated metadata inference. Applying the existing elementwise
`AlignUtil` to complete operand shapes is incorrect because the final matrix
dimensions follow contraction rules rather than elementwise broadcast rules.
Only the leading shapes participate in batch broadcast inference.

The inferred metadata contains at least:

- Output batch shape.
- Batch count.
- `M`, `N`, and `K`.
- Complete output shape.
- Output dtype.

#### Routing

The current rank-two contiguous matmul behavior remains the regression path.

The stride-aware/batched route is needed when:

- Either input is non-contiguous.
- Either input has batch dimensions.
- Batch broadcasting is required.

Rank-two non-contiguous matmul uses the stride-aware route with batch count
one and base offsets zero.

#### Batch transform

Construct an `OffsetUtil<2>` over only the leading dimensions:

```text
iteration shape = output batch shape
A operand shape = A leading shape
A strides       = A leading strides
B operand shape = B leading shape
B strides       = B leading strides
```

For each output batch, it returns `a_base` and `b_base`. A broadcast dimension
has effective stride zero, so multiple output batches can reuse one physical
input matrix without copying it.

#### Matrix addressing

Inside a selected batch:

```cpp
a_value =
    A[a_base
      + row * a.strides()[a_rank - 2]
      + t   * a.strides()[a_rank - 1]];

b_value =
    B[b_base
      + t   * b.strides()[b_rank - 2]
      + col * b.strides()[b_rank - 1]];

C[batch * M * N + row * N + col] = ...;
```

The batch offset utility is not called inside the scalar `t` loop.

This addressing supports arbitrary permutations, including permutations that
move a formerly leading dimension into one of the final two matrix
dimensions.

#### CPU

The CPU implementation adds an outer loop over output batches.

Each batch:

1. Computes `a_base` and `b_base`.
2. Executes the matrix loops using the final-two-dimension input strides.
3. Writes to the contiguous output batch.

OpenMP ownership must avoid accidentally creating nested parallel reductions
for every output element. Changing the broader CPU matmul optimization
strategy is not otherwise in scope.

#### CUDA

CUDA assigns output batches through `grid.z` and a grid-stride batch loop:

```cpp
for (size_t batch = blockIdx.z; batch < batch_count;
     batch += gridDim.z) {
  // Compute batch bases and one or more output tiles.
}
```

The launch caps `grid.z` at the device-supported limit.

Both the simple and optimized kernels need:

- Batch-base calculation.
- Explicit `A` row and contraction strides.
- Explicit `B` contraction and column strides.
- Contiguous batched `C` offsets.

For the optimized kernel:

- Per-thread accumulation arrays are reset for each batch iteration.
- Shared-memory tiles are synchronized per batch.
- The complete `batch_count * M * N` output is cleared if the kernel still
  depends on a pre-clear.
- Vectorized global `B` loads are used only when the physical column stride is
  one and the selected batch address satisfies alignment.
- Scalar shared-memory loads remain the fallback for transposed or unaligned
  `B`.
- Contiguous output vector stores remain valid within each output matrix.

#### Matmul autograd

The logical gradient formulas remain:

```text
dA_pre = matmul(dC, transpose_last_two(B))
dB_pre = matmul(transpose_last_two(A), dC)
```

If batch broadcasting occurred:

```text
dA = sum_broadcast_axis(dA_pre, A.shape)
dB = sum_broadcast_axis(dB_pre, B.shape)
```

The transpose operations are metadata views. The stride-aware matmul route
must consume them without materializing temporary transposes.

The final reduced gradients are contiguous. Any non-contiguous view passed
directly to gradient accumulation must be materialized because V1
`add_inplace` does not accept non-contiguous sources.

#### Matmul tests

Forward, CPU and CUDA:

- Existing rank-two contiguous cases.
- Rank-two transposed `A`.
- Rank-two transposed `B`.
- Both rank-two operands transposed.
- Small batched matmul through the simple CUDA kernel.
- Large batched matmul through the optimized CUDA kernel.
- Unique values in every batch to catch missing base offsets.
- Matching multidimensional batch shapes.
- A shared rank-two right operand.
- A shared rank-two left operand.
- Multiple broadcasted leading dimensions.
- Permuted batch dimensions.
- Permutations mixing batch and matrix dimensions.
- Mixed dtypes.
- Incompatible contraction dimensions.
- Incompatible batch dimensions.
- Output shape and canonical strides.

Autograd:

- Rank-two forward and backward regression.
- Batched forward and backward.
- Transposed operands in backward.
- Shared broadcasted operand receives a batch-reduced gradient.
- Multiple broadcasted axes are all reduced.
- Repeated gradient accumulation remains correct.

CUDA-specific:

- Batch count greater than the selected `grid.z`.
- Optimized kernel with `B` column stride one.
- Optimized kernel with transposed `B` and column stride not one.
- Batch offsets with different alignment properties.

## Integration Pieces and Dependency Order

The conceptual subsets above translate into the following implementation
pieces:

1. Refactor the offset utility around an explicit logical iteration shape
   while preserving existing binary broadcast behavior.
2. Add CPU and CUDA unary offset utility instantiations.
3. Add `is_contiguous` and stride-preserving TensorImpl metadata operations.
4. Add public transpose-dimensions and permute APIs, autograd functions, and
   bindings.
5. Add the stride-aware contiguous materialization kernel.
6. Route reshape, logical exports, printing, and device transfer through
   contiguous materialization where required.
7. Enforce V1 restrictions on copy and in-place accumulation.
8. Add fusion barriers and capture gating.
9. Add the non-contiguous unary path.
10. Replace binary/expand routing with the unified offset-aware binary path.
11. Add strided reduction base mapping.
12. Add strided rank-two matmul.
13. Add batch shape inference, batch offset mapping, and CPU batched matmul.
14. Add CUDA batched matmul to the simple and optimized kernels.
15. Add broadcast-aware batched matmul autograd.
16. Update documentation and both C++ and Python-facing tests.

Dependencies:

```text
offset transformation
    ├── contiguous materialization
    ├── unary
    ├── binary
    ├── reduction
    └── matmul batch mapping

TensorImpl layout metadata
    ├── views
    ├── fusion barriers
    └── kernel route decisions

contiguous materialization + layout-aware kernels
    └── correct autograd and external reads
```

Each piece should leave existing contiguous behavior passing before the next
piece is added.

## Completion Criteria

V1 is complete when:

- Arbitrary transpose and permutation are metadata-only storage-sharing views.
- Every public logical read observes view order rather than physical storage
  order.
- Reshaping a non-contiguous tensor produces a correct contiguous result.
- Unary, binary, reduction, and matmul operations accept dense
  non-contiguous inputs on CPU and CUDA.
- Binary broadcasting and batch matmul broadcasting do not allocate expanded
  inputs.
- Every computational result is contiguous.
- Unsupported non-contiguous writes fail rather than silently corrupting
  logical order.
- Non-contiguous tensors never enter flat generated fusion kernels.
- Autograd produces correct gradients through views, reductions, and batched
  broadcasted matmul.
- Existing contiguous tests remain green.

## Assumptions Requiring Confirmation

This draft makes three assumptions that should be confirmed before
implementation:

1. “Copy and in-place do not support contiguous tensors” was intended to mean
   that they do not support **non-contiguous** tensors.
2. Batched matmul broadcasts leading dimensions rather than requiring exact
   batch-shape equality.
3. The legacy no-dimension `transpose(a)` API swaps the final two dimensions,
   while the new overload accepts any two explicit dimensions.

The remaining unresolved behavior is `fill` on a non-contiguous dense view.
