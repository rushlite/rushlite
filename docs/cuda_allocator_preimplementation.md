# CUDA allocator pre-implementation plan

> The final simplified scope is recorded in
> `cuda_allocator_code_change_plan.md`; `valid()`, `stats()`, tombstones, and
> per-segment free indexes are not part of the implementation.

## Scope boundary

This phase stops before CUDA or Rushlite integration.

The only executable artifact is the standalone, CUDA-free model in
`prototypes/cuda_allocator/`. This phase does not change:

- `memory.cuh` or `memory.cu`;
- `DataPtr`;
- CUDA allocation call sites;
- CMake targets; or
- the Python module.

The model is an executable specification for the host metadata behavior. It is
not linked into Rushlite and is not intended to ship.

## Proposed production decomposition

### 1. Public internal facade

The eventual `memory.cuh` change should expose only:

- allocation by byte count and stream;
- deallocation by exact returned address, owning device, and stream;
- allocator statistics; and
- an internal cache-release operation.

Block and segment metadata should not be visible to tensor code.

### 2. Pure host block arena

The core algorithm should have no CUDA dependencies. Its responsibilities are:

- maintain a gap-free physical block chain for every segment;
- find the smallest fitting free block;
- split a block when the remainder is usable;
- find allocations by exact returned address;
- classify unknown and interior frees, with debug tombstones for reliable
  double-free classification after coalescing;
- eagerly coalesce adjacent free blocks;
- identify and extract completely free segments;
- derive byte/block statistics; and
- validate all metadata invariants.

It should accept segment bases as integer address values and never dereference
them. It should not own a mutex: its caller serializes access.

The recommended implementation is one private CUDA-free header used by
`memory.cu` and by a small standalone unit-test target. That adds one production
file but ensures fast tests exercise the real algorithm. Keeping everything
inside `memory.cu` would preserve strict localization, but the fast tests would
then cover only a duplicate model.

### 3. Per-device CUDA coordinator

The eventual CUDA layer should own one state object per device:

```text
DeviceState
  mutex
  BlockArena
  configuration
  event counters
```

It will be responsible for:

- validating the version-one stream contract;
- choosing ordinary versus dedicated segment sizes;
- switching to the owning CUDA device and restoring the caller's device;
- acquiring and releasing whole CUDA segments;
- synchronizing before safe segment release;
- one-shot release-and-retry on OOM;
- translating failures into the project's error mechanism; and
- combining arena totals with event counters.

Version one should serialize cold `cudaMalloc`, synchronization, and
`cudaFree` operations with the same per-device lock used for metadata. That is
simple and prevents duplicate growth and release races. The cold calls are rare;
more elaborate growth/maintenance gates can wait for profiling.

Synchronizing first and locking afterward is incorrect: another host thread
could reuse a segment and enqueue work in the gap before it is released.

### 4. Process lifetime and ownership adapters

The eventual intentionally immortal singleton should create device states
lazily. A manager lock protects state creation only, so different devices do
not share a hot-path lock.

`empty_cuda` should capture the owning device and stream in its `DataPtr`
deleter. Temporary CUDA allocations should use a small internal RAII owner so
an early error cannot strand an allocator block. `ListDevicePtr` should use the
same ownership path.

### 5. Observability

The arena should derive:

- requested live bytes;
- aligned active bytes;
- reserved and free bytes;
- largest free block;
- completely free segment bytes; and
- block and segment counts.

The coordinator should count:

- allocation and logical-free operations;
- cache hits/reuses;
- segment growth;
- cache releases; and
- OOM recoveries.

No CUDA query or virtual dispatch should occur on a cache hit or logical free.

## CUDA-free model

The current prototype contains:

- `BlockArena`, which models best-fit allocation, split, coalescing, lookup,
  free-segment extraction, statistics, and invariants;
- `MockSegmentProvider`, which supplies deterministic fake addresses, a fixed
  capacity, release, and allocation failure; and
- `HostAllocatorModel`, which models segment policy, retention, explicit cache
  release, and one-shot OOM recovery.

It deliberately omits stream ordering, synchronization, device restoration,
the singleton, per-device threading, CUDA errors, and graph capture. Those
belong in later CUDA integration tests rather than an inaccurate host
simulation.

Once a production `BlockArena` exists, these tests should be moved onto it and
the duplicate prototype model removed.

## Test plan

### Layer 1: standalone model

Already covered:

- zero-byte allocation and null free;
- 256-byte alignment, splitting, and same-address reuse;
- exact best-fit selection;
- two-sided eager coalescing;
- unknown, interior, and double-free classification;
- releasing fully free segments without releasing a live segment;
- recoverable OOM by releasing cached segments and retrying once;
- hard OOM while preserving live allocations;
- 50,000 scratch-allocation churn iterations that plateau at one segment; and
- 20,000 deterministic randomized operations with invariant validation after
  every mutation.

Run the model with an ordinary compiler plus warning-clean, AddressSanitizer,
and UndefinedBehaviorSanitizer builds.

Additional useful model cases before promotion:

- exact-fit allocation with no split (all segment and block sizes share the
  allocation quantum, so a nonzero sub-quantum remainder is impossible);
- overlapping and misaligned segment rejection;
- allocation-size and address overflow;
- multiple equal-size best-fit candidates;
- release ordering for ordinary versus dedicated segments; and
- randomized runs across several seeds and segment policies.

### Layer 2: production host-core tests

When implementation begins, port Layer 1 onto the real CUDA-free `BlockArena`.
Build it as a tiny standalone test target that links neither `tensor_core` nor
CUDA. This preserves the fast edit/test loop.

If allocation-size tracing is added temporarily, replay the real `pow`
backward scratch trace through this layer and verify reservation plateaus.

### Layer 3: CUDA coordinator tests

These are explicitly outside the current mock phase:

- same-address reuse across two ordered default-stream kernels;
- clear rejection of a non-default stream;
- current-device restoration on success and failure;
- independent state for multiple devices;
- concurrent host allocation/free on one device;
- safe cache release with queued kernels;
- actual CUDA OOM release-and-retry;
- cleanup at each temporary-allocation call site; and
- Compute Sanitizer use-after-free and bounds checks.

### Layer 4: end-to-end regression

Keep the verified add-versus-pow reproduction.

The short CUDA smoke form should assert:

- custom reserved bytes stop increasing after warmup;
- segment count is bounded over the second half of the run;
- active/requested bytes return to the stable logical live set; and
- the run stays well below device capacity.

The 50,000-iteration form should remain a long regression/benchmark rather
than a per-commit unit test. It should also check that host allocator overhead
does not materially regress the fast `add` control.

## Recommended decisions before CUDA implementation

1. Allow one private CUDA-free allocator-core header so tests exercise
   production metadata code.
2. Make version one null/default-stream-only.
3. Compile explicitly with legacy default-stream semantics. Null stream reuse
   across host threads is unsafe if per-thread default-stream semantics are
   enabled.
4. Keep one per-device mutex across both hot metadata mutation and rare cold
   CUDA operations initially.
5. Keep cache release internal initially.
6. Start with exact best-fit and 256-byte alignment.
7. Make ordinary segment size experimentally configurable without promising it
   as stable API.
8. Treat CUDA graph capture as unsupported in version one. A captured graph can
   retain a pointer after the `DataPtr` whose destruction normally authorizes
   reuse.
9. Capture the owning device at allocation time; never infer it in a deleter
   from the current thread.
10. Prefer completely free dedicated segments during pressure recovery, while
    permitting release of every completely free segment if needed.

## Questions that block CUDA integration

1. What is the project-level OOM contract: throw, terminate through the
   assertion machinery, or return a recoverable error?
2. Can Rushlite guarantee legacy default-stream compilation for downstream
   builds, or must reuse also be scoped to the allocating host thread?
3. Should invalid and double frees terminate in all builds, or terminate only
   in debug and log in release? Reliable double-free classification after
   coalescing needs freed-address tombstones; the recommendation is to keep them
   only in debug builds.
4. Is the private allocator-core header acceptable, or is strict
   `memory.cuh`/`memory.cu` localization required?
5. Should the existing resize-by-value correctness bug be fixed in the
   allocator change or explicitly excluded?

## Questions that can wait for measurement

1. Should the ordinary segment default be 64 MiB, 256 MiB, or another value?
2. Is `std::multimap` best-fit fast enough under the real benchmark?
3. Should completely free dedicated segments be safely released eagerly or
   cached until pressure/explicit cache release?
4. Should allocator statistics and cache release become public Python APIs?
5. When are multi-stream, per-thread default stream, and CUDA graph capture
   required?
