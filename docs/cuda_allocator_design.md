# Host-side CUDA allocator design

> The simplified, implementation-ready plan in
> `cuda_allocator_code_change_plan.md` supersedes earlier proposals here for
> allocator statistics, invariant validation, and detailed debug diagnostics.

## Status

Proposed initial design for the CUDA allocator worktree, based on
`995ae30cc1ef2a94ee8f1436717ef2183097216b`.

The allocator runaway is reproduced by repeated, unsynchronized autograd
backward calls. The earlier forward-only `pow` loop was not representative: it
allocated one uniform output per iteration and plateaued, while `pow` backward
creates a sequence of short-lived scratch tensors with a mix of lifetimes and
sizes.

The mock-only component split, fast test plan, and decisions still needed before
CUDA integration are in `cuda_allocator_preimplementation.md`.

## Decision

Implement a process-wide, per-device host allocator that:

1. acquires large device-memory segments on a cold path;
2. suballocates blocks from those segments entirely on the host;
3. returns a block to its host free index as soon as its `DataPtr` dies;
4. reuses that address immediately for later work on the same CUDA stream;
5. splits and eagerly coalesces blocks to control fragmentation; and
6. releases whole segments only on an explicit or OOM recovery path after
   establishing device completion.

The allocator API should take a byte count and return an address. Deallocation
should take the exact returned address plus the allocation's recorded device
and stream context; callers must not supply the byte count.

Do not reserve 1 GiB at process startup. Create the first segment lazily. A
configurable 256 MiB ordinary segment is a reasonable initial value, but it is
a tuning choice to validate against real workloads.

## Reproduction

### Environment

- reproducing build: `0496f53b3c40d3dd2952d9c664d4d0bc07a77ed6`
- allocator worktree base: `995ae30cc1ef2a94ee8f1436717ef2183097216b`
- GPU: NVIDIA RTX PRO 4000 Blackwell, 24,467 MiB
- driver: 580.159.04
- CUDA toolkit: 12.9
- build type: Release with CUDA enabled

The reproducing build differs in CUDA error-checking macro behavior, not in the
allocation sites or backward kernels relevant to this design.

### Workload

The self-contained reproduction builds one graph over two 512 x 512 FP32 CUDA
variables, synchronizes, then calls `out.backward()` 50,000 times without an
inner synchronization. It runs `add` as the fast control and `pow` as the slow
case. After the loop it synchronizes and reads the default CUDA pool's current
used and reserved byte counters.

Fresh result:

```text
release_threshold = 0 bytes

add backward, 50000 unsynced iterations
  start       used= 6.0 MiB   reserved=    32.0 MiB
  post-sync   used= 6.0 MiB   reserved=    32.0 MiB

pow backward, 50000 unsynced iterations
  start       used= 6.0 MiB   reserved=    32.0 MiB
  post-sync   used= 9.5 MiB   reserved= 23744.0 MiB
```

The three reuse controls were also queried and were enabled:

```text
follow_events=1
opportunistic=1
internal_dependencies=1
```

The control and failure have the same shapes and loop structure. The stable
post-sync used footprint rules out a tensor leak proportional to iteration
count. The reservation grows by roughly 23.7 GiB only for the slower backward
sequence and remains reserved after the final synchronization despite a zero
release threshold.

### What the result establishes

The host enqueues backward work much faster than the device executes the slow
`pow` sequence. Scratch tensors are logically dead on the host after each
backward step, but the default CUDA pool does not recycle and release their
storage sufficiently for this allocation pattern. Pending stream-ordered frees
and allocator fragmentation are consistent with the observation.

CUDA's exact internal block-selection behavior is not part of Rushlite's
contract, so the design must not depend on a stronger causal claim. The facts
needed for this design are:

- the Rushlite logical live set is stable;
- all current CUDA work is submitted to the null/default stream;
- default-pool reserved memory grows to device capacity;
- the growth depends strongly on kernel speed; and
- a host allocator can observe logical death before device completion and can
  safely order reuse on the same stream.

The host allocator eliminates the dependency on the default pool's per-tensor
reuse decisions. CUDA sees only a small number of long-lived segments;
Rushlite decides which sub-block address a later kernel receives.

## Stream-safety argument

For a block at address `A`, the current default-stream sequence is:

```text
old kernel using A
host destroys the old DataPtr
allocator marks A free in host metadata
allocator returns A for a later tensor
new kernel using A
```

The host does not call `cudaFreeAsync(A)` when the sub-block dies. The containing
segment remains allocated. Because both kernels are submitted to the same
stream, the new kernel cannot access `A` until the old kernel has finished
using it. No hot-path event or synchronization is required.

This is the central invariant:

> A logically freed block may be reused immediately only when every prior use
> and every later use are ordered on the same CUDA stream.

A host mutex does not establish device ordering. The initial implementation
must accept only the null/default stream and diagnose any other stream. General
multi-stream support requires pending-free queues and CUDA events, and is out
of scope for version one.

## Public internal interface

Declare the interface in `csrc/include/lamp3/tensor/cuda/memory.cuh` and define
it in `csrc/src/tensor/cuda/memory.cu`.

```cpp
namespace lmp::tensor::detail::cuda {

struct AllocatorStats {
  size_t active_bytes;
  size_t requested_bytes;
  size_t reserved_bytes;
  size_t free_bytes;
  size_t largest_free_block;
  size_t segment_count;
  size_t allocation_count;
  size_t reuse_count;
  size_t segment_growth_count;
};

class CudaAllocator final {
 public:
  static CudaAllocator& instance();

  void* allocate(size_t bytes, cudaStream_t stream = nullptr);
  void deallocate(void* address, int device,
                  cudaStream_t stream = nullptr) noexcept;

  AllocatorStats stats(int device) const;
  void empty_cache(int device);

  CudaAllocator(const CudaAllocator&) = delete;
  CudaAllocator& operator=(const CudaAllocator&) = delete;

 private:
  CudaAllocator();
};

}  // namespace lmp::tensor::detail::cuda
```

The allocation's `DataPtr` deleter captures the owning device and stream. It
passes those back with the address, so deallocation does not depend on the
calling thread's current CUDA device and does not need the allocation size.

Recommended zero-size behavior:

- `allocate(0)` returns `nullptr`;
- `deallocate(nullptr, ...)` is a no-op.

## Singleton lifetime and device ownership

Use function-local initialization for thread-safe first access, but make the
allocator intentionally immortal:

```cpp
CudaAllocator& CudaAllocator::instance() {
  static auto* allocator = new CudaAllocator();
  return *allocator;
}
```

A normally destructed Meyers singleton is unsafe here. Static `DataPtr`
instances can outlive it, and CUDA runtime teardown order across translation
units is uncontrolled. Process exit reclaims CUDA memory; explicit release is
handled by `empty_cache`.

The singleton is a manager containing independent state per CUDA device. Each
device state has its own mutex, segments, free index, allocation lookup, and
statistics. Operations on different GPUs must not share a hot-path mutex.

If allocation or release temporarily changes the current CUDA device, it must
restore the caller's original device before returning.

## Segment acquisition and sizing

### CUDA primitive

Use `cudaMalloc` to acquire segments in version one and `cudaFree` to release
them on cold paths. This deliberately bypasses the stream-ordered default pool
whose reservation behavior is being replaced. Segment growth may synchronize,
but it should be rare after warmup.

Using `cudaMallocAsync` for whole segments can be reconsidered later. It is not
needed to validate the host suballocator and would make pool statistics harder
to interpret.

### Initial sizing policy

Recommended starting policy:

- no allocation at process or CUDA initialization;
- lazily allocate a 256 MiB ordinary segment on the first request;
- align segment sizes to 2 MiB;
- for an ordinary miss, allocate
  `max(ordinary_segment_size, align_up(request, 2 MiB))`;
- requests larger than half the ordinary segment size receive a dedicated
  segment; and
- do not geometrically increase the ordinary segment size.

A fixed ordinary size bounds accidental growth and makes behavior predictable.
The 256 MiB value is intentionally much smaller than the proposed 1 GiB and
much larger than the reproduction's stable live set. It should be configurable
for experiments before becoming API.

A failed segment allocation enters the OOM recovery path rather than silently
reducing the segment size in a loop.

## Block representation

Metadata is host memory only and must have stable addresses.

```cpp
struct Segment;

struct Block {
  void* address;
  size_t size;
  size_t requested_size;
  bool allocated;

  Segment* segment;
  Block* address_prev;
  Block* address_next;

  // Valid only while free; permits constant-time removal of a known neighbor.
  FreeIndexIterator free_position;
};

struct Segment {
  void* base;
  size_t size;
  int device;
  bool dedicated;
  Block* first;
};
```

Each device state owns:

- all `Segment` objects;
- all stable `Block` nodes;
- an exact-address map for allocated blocks;
- an ordered free-block index keyed by block size; and
- counters and a mutex.

The physical block links form an address-ordered, gap-free partition of each
segment. An allocated block is in the address lookup and not in the free index.
A free block is in the free index exactly once.

## Alignment and free index

Use a 256-byte allocation quantum initially. Round requests up to that quantum,
while retaining the original requested byte count for statistics. Segment bases
from CUDA satisfy the required alignment.

Start with an ordered best-fit index such as
`std::multimap<size_t, Block*>`:

- find the smallest fitting block with `lower_bound` in `O(log n)`;
- store each free block's iterator for constant-time removal during coalescing;
- preserve exact aligned sizes rather than rounding every tensor to a power of
  two; and
- split only when the remainder is at least one quantum.

Power-of-two buckets and a nonempty bitmap can replace this index if profiling
shows metadata lookup is material. They are not the default because size-class
rounding can approach 2x internal fragmentation, directly opposing the memory
objective.

## Allocation algorithm

Under the device mutex:

1. reject a non-default stream in version one;
2. normalize the request to the allocation quantum;
3. select the smallest fitting free block;
4. if none exists, acquire a segment while retaining exclusive ownership of
   the device state, add it as one free block, and retry;
5. remove the selected block from the free index;
6. split a usable suffix and insert that suffix into the free index;
7. mark the selected block allocated;
8. store its original requested size;
9. insert its exact start address into the allocation lookup; and
10. update active, requested, reuse, and growth counters.

Only the exact block start is returned. No CUDA operation occurs on a cache hit.

Version one deliberately holds the device-state mutex across the rare,
potentially blocking `cudaMalloc`. This prevents duplicate growth and keeps the
cold path simple; split growth locking is a later optimization if measured.

## Deallocation and coalescing

Under the owning device mutex:

1. treat null as a no-op;
2. reject a non-default or mismatched stream;
3. find the exact address in the allocation lookup;
4. diagnose unknown addresses, interior pointers, and double frees;
5. erase the allocated lookup entry and mark the block free;
6. remove a free left neighbor from the free index and merge it;
7. remove a free right neighbor from the free index and merge it;
8. erase absorbed metadata nodes and repair physical links;
9. insert the one coalesced block into the free index; and
10. update active, requested, and free-byte counters.

No `cudaFree` or `cudaFreeAsync` occurs here. That distinction is what prevents
device run-ahead from forcing new physical reservations.

A deallocator is `noexcept`. In debug builds, invalid frees should terminate
with a precise diagnostic. Release builds must log the failure without throwing
through `DataPtr` destruction.

## Segment retention, release, and OOM

A segment is completely free when its physical list contains one free block
covering the full segment. Logical freedom does not prove that queued kernels
have finished using the segment.

Hot-path deallocation therefore keeps complete segments cached. Safe release
is allowed only on:

- explicit `empty_cache(device)`;
- segment-allocation failure; or
- a future deliberate memory-pressure API.

Version-one release procedure:

1. lock the device state so addresses cannot be reused during release;
2. synchronize the device;
3. remove every completely free segment from the free index and metadata;
4. call `cudaFree` for those segment bases;
5. update statistics; and
6. unlock.

Synchronizing before locking is racy: another host thread could reuse a segment
and enqueue work in the gap before the release path frees it.

On OOM, release safe completely free segments, retry the requested segment once,
and then fail with diagnostics containing requested bytes, active bytes,
reserved bytes, total free bytes, largest free block, and segment count.
Partially free segments cannot be released without relocating live allocations,
which is out of scope.

## Integration scope

Most implementation work remains localized to:

- `csrc/include/lamp3/tensor/cuda/memory.cuh`;
- `csrc/src/tensor/cuda/memory.cu`.

All CUDA allocation sites must use the allocator or they can recreate the
runaway outside its accounting:

- `empty_cuda`;
- `resize_cuda`;
- conversion temporaries in `vecCopyHostToDevice` and `copy_cuda`; and
- `ListDevicePtr` in `csrc/include/lamp3/tensor/cuda/list_ptr.cuh`.

`ListDevicePtr` is the only expected allocation call-site change outside the
memory header/source pair.

The current resize interface has a separate correctness problem:
`resize_fn` and `resize_cuda` take `DataPtr` by value, so assigning a replacement
does not update `StorageImpl::data_ptr_`. Fix or exclude resize before using it
to validate the allocator.

## Observability

The custom allocator must expose its own statistics because segments acquired
with `cudaMalloc` do not appear in default-pool reserved counters.

At minimum report per device:

- requested live bytes;
- aligned active bytes;
- segment reserved bytes;
- total free bytes;
- largest free block;
- completely free segment bytes;
- allocation and logical-free counts;
- block reuse count;
- segment growth count; and
- OOM recovery count.

Debug builds should validate after every mutation that blocks partition each
segment without gaps or overlap, every free block is indexed once, every
allocated start is mapped once, and aggregate counters equal the metadata.

## Validation plan

### Unit tests

- zero-byte and null-free behavior;
- exact-fit allocation;
- split with and without a usable remainder;
- 256-byte alignment;
- best-fit selection;
- deallocation with no free neighbor;
- left, right, and two-sided coalescing;
- address lookup updates after split and coalesce;
- complete-segment coalescing;
- unknown, interior, and double-free diagnostics;
- independent per-device state;
- current-device restoration; and
- concurrent host metadata operations.

### CUDA integration tests

- immediate pointer reuse across two same-stream kernels without a hot-path
  synchronization;
- rejection of a non-default stream;
- cold-path safe segment release;
- OOM release-and-retry; and
- Compute Sanitizer coverage for use-after-free and out-of-bounds access.

### Regression test

Keep the add-versus-pow backward reproduction as the acceptance test. Run it in
Release mode before and after the allocator change.

Expected after the change:

- add remains flat;
- `pow` completes 50,000 unsynchronized backward calls;
- custom allocator reserved bytes plateau after warmup;
- requested/active bytes return to the stable live footprint after sync; and
- device memory does not approach capacity.

The acceptance test should sample during the loop as well as before and after,
and should report both custom allocator statistics and CUDA device free memory.
Default-pool counters alone are no longer sufficient after segment allocation
moves to `cudaMalloc`.

## Open questions

1. Is 256 MiB the right ordinary segment size, or should the first experiment
   compare 64 MiB, 256 MiB, and 1 GiB?
2. Is null/default-stream-only an acceptable explicit contract for version one?
3. When is multi-stream or per-thread-default-stream support planned?
4. Should `empty_cache` remain internal or become public C++/Python API?
5. Should allocator OOM throw, terminate, or return a recoverable error?
6. Should dedicated large segments be cached or preferred for release?
7. Is a `std::multimap` best-fit index fast enough under benchmark load?
8. How should allocator calls behave during CUDA graph capture?
9. May first allocation create a CUDA context, or must initialization be
   explicit?
10. Should the reproduction become a benchmark, a long-running regression test,
    or both?

## Recommended implementation order

1. Add the allocator class, per-device state, statistics, and invariant checks.
2. Implement lazy segment acquisition, best-fit allocation, split, free, and
   coalescing.
3. Route `empty_cuda` through it and add focused unit tests.
4. Route all temporary and `ListDevicePtr` allocations through it.
5. Run Compute Sanitizer and ordinary CUDA tests.
6. Run the add-versus-pow reproduction and tune segment size only after
   correctness and plateau behavior are established.
7. Add cold-path `empty_cache` and OOM recovery.
8. Address multi-stream support only as a separate design extension.
