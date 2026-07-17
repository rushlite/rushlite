# CUDA allocator code-change plan

## Goal

Replace per-tensor `cudaMallocAsync`/`cudaFreeAsync` with a host-side
suballocator whose reservation reaches a bounded high-water mark during the
unsynchronized `pow` backward loop.

This plan keeps the simplified `BlockArena`:

- no `valid()`;
- no `stats()`;
- no debug tombstones or detailed invalid-free classification;
- one global best-fit free index per device, not one index per segment; and
- whole-segment extraction retained for explicit release and OOM recovery.

## Memory layout

Each CUDA device owns one arena:

```text
DeviceState
  mutex
  BlockArena
    global free index: size -> Block*
    allocated map: address -> Block*
    segments
      Segment A: block <-> block <-> block
      Segment B: block <-> block
```

A segment does not own buckets or a separate allocator. It only owns an
address-ordered block chain so neighboring free blocks can be coalesced and the
arena can recognize when the whole segment is free.

Allocation searches the global free index across every segment. If no single
free block is large enough, the CUDA layer acquires another segment, adds its
initial whole free block to the same index, and retries.

Total free bytes spread across several nonadjacent blocks do not satisfy a
larger request. In that fragmented case, another segment is also required.

## Production files

### New private host-core header

`csrc/src/tensor/cuda/allocator_core.hpp`

Move the simplified `BlockArena` here with this interface:

```cpp
class BlockArena {
 public:
  explicit BlockArena(size_t alignment);

  void add_segment(Address address, size_t size);
  std::optional<Allocation> allocate(size_t size);
  bool deallocate(Address address);
  std::vector<ReleasedSegment> extract_fully_free_segments();
};
```

This file contains no CUDA headers, mutex, singleton, segment-size policy,
statistics, or debug validator. It is the same algorithm used by the mock and
the CUDA implementation.

### CUDA coordinator and existing memory operations

`csrc/src/tensor/cuda/memory.cu`

Add private implementation types:

```text
CudaAllocator
  process-lifetime singleton
  lazily created DeviceState per CUDA device

DeviceState
  mutex
  BlockArena
```

`CudaAllocator::allocate(bytes)`:

1. Return null for zero bytes.
2. Read the current CUDA device.
3. Lock that device's state.
4. Ask its arena for a block.
5. On a miss, call `cudaMalloc` for one segment.
6. Add the segment to the arena and retry.
7. Return the address together with the owning device.

`CudaAllocator::deallocate(address, device)`:

1. Return immediately for null.
2. Lock the recorded device's state.
3. Return the address to its arena.
4. Do not call CUDA and do not synchronize.

The `DataPtr` deleter captures the device returned during allocation. It must
not use the device current on the thread when destruction happens.

`CudaAllocator::release_free_segments(device)`:

1. Lock the device state.
2. Set the owning CUDA device, preserving the caller's previous device.
3. Synchronize the device.
4. Extract completely free segments.
5. Call `cudaFree` on their base addresses.
6. Restore the caller's previous device.

The lock must cover synchronization and extraction so another host thread
cannot reuse a segment between those operations.

Use `cudaMalloc`/`cudaFree` for whole segments, not the async pool. Normal block
allocation and deallocation make no CUDA runtime calls.

### Internal declarations

`csrc/include/lamp3/tensor/cuda/memory.cuh`

Keep the allocator class private to `memory.cu`. The existing `empty_cuda`
function is sufficient as the ownership boundary for tensor code.

Only add a cache-release declaration if it is required by a CUDA test. Do not
expose allocator statistics or block metadata.

### List ownership

`csrc/include/lamp3/tensor/cuda/list_ptr.cuh`

Replace its direct `cudaMallocAsync`/`cudaFreeAsync` ownership with `DataPtr`
returned by `empty_cuda`:

```text
DataPtr storage_
get() -> static_cast<T*>(storage_.data())
```

This avoids exposing the allocator itself in a template header.

### Build contract

`csrc/src/tensor/CMakeLists.txt`

Explicitly compile CUDA code with legacy default-stream semantics. Immediate
host-side reuse is safe only because every old and new use is ordered on that
same legacy default stream.

Do not add multi-stream or event handling in this change.

## Segment policy

Start with one fixed ordinary segment size:

```text
segment_bytes = max(64 MiB, aligned_request_bytes)
```

The initial recommendation is 64 MiB because the reproduced stable live set is
roughly 10 MiB and a smaller retained high-water mark is preferable while
validating behavior.

There is no dedicated-segment flag or separate large-allocation path. A request
larger than 64 MiB simply receives a request-sized segment. Every segment uses
the same global arena logic.

All block requests are aligned to 256 bytes. No separate 2 MiB segment
alignment policy is needed for the first implementation.

The segment-size constant can be tuned after the acceptance test; it should not
become public configuration in this change.

## OOM behavior

The allocation path already holds the device-state mutex. OOM recovery calls a
private `release_free_segments_locked` helper and must not re-enter a public
function that attempts to lock the same mutex again.

Segment acquisition must inspect the `cudaMalloc` return value directly. The
allocator worktree's release-mode `LMP_CUDA_CHECK` does not provide a usable
OOM branch.

On `cudaErrorMemoryAllocation`:

1. safely release every completely free segment;
2. retry the requested segment allocation once;
3. if it still fails, throw a runtime error containing the requested segment
   size and CUDA error string.

Other CUDA errors fail immediately without an OOM retry.

`extract_fully_free_segments()` is used only on this cold recovery path and,
optionally, by an internal cache-release test hook.

## Call-site rollout

### Change 1: production arena and `empty_cuda`

- Promote the mock arena into the private host-core header.
- Add the singleton/per-device coordinator in `memory.cu`.
- Route `empty_cuda` through it.
- Keep all other allocation sites unchanged for the first build.

This is the smallest change capable of testing the `pow` backward failure.

### Change 2: temporary buffers in `memory.cu`

Replace raw temporary allocation/free pairs in:

- `vecCopyHostToDevice`;
- CUDA-to-CPU `copy_cuda`; and
- CUDA-to-CUDA `copy_cuda`.

Use local `DataPtr` objects from `empty_cuda`. Their ordinary scope supplies
logical deallocation even when a CUDA check throws.

### Change 3: `ListDevicePtr`

Store a `DataPtr` from `empty_cuda` instead of constructing a separate
`shared_ptr` around an async CUDA allocation.

### Explicitly excluded: resize

`resize_cuda` and the dispatch signature currently take `DataPtr` by value.
Replacing the local `DataPtr` does not update `StorageImpl`, independent of the
allocator.

Do not mix that correctness fix into the first allocator implementation.
Either:

- leave resize on its current path with a documented TODO; or
- fix the resize API in a separate preceding change and then route it through
  `empty_cuda`.

## Mock versus CUDA implementation

| Concern | Mock | CUDA implementation |
|---|---|---|
| Block selection | `BlockArena` global best-fit index | Same `BlockArena` |
| Segment contents | Address-ordered block chain | Same |
| Segment acquisition | `MockSegmentProvider::acquire` | `cudaMalloc` |
| Segment release | `MockSegmentProvider::release` | synchronize, then `cudaFree` |
| Address | Deterministic fake integer | CUDA pointer represented as integer in the arena |
| Capacity/OOM | Fixed mock capacity | CUDA return code |
| Devices | One model instance | One locked state per CUDA device |
| Thread safety | None | Per-device mutex |
| Stream safety | Not modeled | Legacy default stream only |
| Object lifetime | Stack object | Intentionally immortal singleton |
| Tensor ownership | Direct test calls | `DataPtr` captures owning device |

The mock provider and `HostAllocatorModel` do not ship. Only `BlockArena` is
promoted into production.

## Test sequence

### Fast host-only loop

Point the existing standalone tests at the production
`allocator_core.hpp`. Continue compiling them directly with the host compiler.

Required cases:

- alignment, splitting, and exact reuse;
- global best-fit selection across segments;
- left/right/two-sided coalescing;
- invalid address rejection;
- whole-segment extraction;
- OOM release-and-retry through the mock provider;
- 50,000 scratch cycles with one segment acquisition; and
- randomized non-overlap testing under ASan/UBSan.

### Focused CUDA tests

After routing only `empty_cuda`:

1. Allocate, logically free, and reallocate the same-sized tensor; verify
   address reuse.
2. Queue a kernel using the first allocation, logically free it, reuse the
   address, queue a second kernel, then synchronize and verify both operations.
3. Exercise safe whole-segment release after queued work.
4. Verify an allocation on one device is returned to that device even if
   destruction occurs with another device current.

### End-to-end acceptance

Run the existing add-versus-pow reproduction in Release mode.

Expected result:

- `add` remains stable;
- 50,000 unsynchronized `pow` backward calls complete;
- post-run device memory consumption is bounded to a small number of segments,
  rather than approximately 23.7 GiB; and
- default async-pool reservation no longer grows with each logical tensor
  allocation.

Use `cudaMemGetInfo` before and after the run rather than adding a production
statistics API.

## Non-goals

- allocator statistics;
- metadata validation passes;
- debug tombstones;
- per-segment buckets;
- power-of-two size classes;
- multi-stream reuse and CUDA events;
- per-thread default streams;
- CUDA graph capture;
- public cache controls;
- configurable segment size; and
- resize API repair.
