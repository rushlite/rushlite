"""Minimal reproducible example for the rushlite CUDA benchmark OOM.

Runs a single backward op in an unsynchronized loop (exactly what the
benchmark timer does) and reports the cudaMallocAsync pool's USED vs
RESERVED bytes.

Two ops are run for contrast:
  - add  (fast kernel)  -> pool stays flat            = control
  - pow  (slow kernel)  -> RESERVED explodes, USED flat = the bug

USED  = bytes actually handed to live tensors (the true footprint).
RESERVED = physical device bytes the pool holds from the driver.

If USED is flat while RESERVED balloons, there is no leak: the pool is
reserving memory it can neither reuse nor release. See explanation below.

Run:  python repro_oom_minimal.py            # both ops
      python repro_oom_minimal.py pow 100000 # one op, more iters
"""

import ctypes
import sys

import rushlite

_rt = ctypes.CDLL("libcudart.so")
_RELEASE_THRESHOLD, _RESERVED_CURRENT, _USED_CURRENT = 0x4, 0x5, 0x7


def _pool():
    p = ctypes.c_void_p()
    assert _rt.cudaDeviceGetDefaultMemPool(ctypes.byref(p), 0) == 0
    return p


def _attr(pool, attr) -> int:
    v = ctypes.c_uint64(0)
    assert _rt.cudaMemPoolGetAttribute(pool, attr, ctypes.byref(v)) == 0
    return v.value


def _mib(x) -> float:
    return x / (1024 * 1024)


def _mem_info() -> tuple[int, int]:
    free = ctypes.c_size_t()
    total = ctypes.c_size_t()
    assert _rt.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)) == 0
    return free.value, total.value


def run(op: str, n: int, pool) -> None:
    shape = [512, 512]
    a = rushlite.rand(shape, requires_grad=True, device=rushlite.device.cuda,
                      dtype=rushlite.dtype.float32)
    b = rushlite.rand(shape, requires_grad=True, device=rushlite.device.cuda,
                      dtype=rushlite.dtype.float32)
    out = getattr(rushlite, op)(a, b)  # build the graph once
    rushlite.cuda.sync()

    def line(tag: str) -> None:
        free, total = _mem_info()
        print(f"  {tag:11s} used={_mib(_attr(pool, _USED_CURRENT)):8.1f} MiB   "
              f"reserved={_mib(_attr(pool, _RESERVED_CURRENT)):9.1f} MiB   "
              f"device-global-used={_mib(total - free):9.1f} MiB", flush=True)

    print(f"\n=== {op} backward, {n} unsynced iterations ===", flush=True)
    line("start")
    for i in range(n):
        out.backward()            # no sync inside the loop, like the benchmark
    rushlite.cuda.sync()
    line("post-sync")


def main() -> None:
    pool = _pool()
    print(f"release_threshold = {_attr(pool, _RELEASE_THRESHOLD)} bytes "
          "(0 = pool should release free memory at every sync)")
    if len(sys.argv) > 1:
        run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 50000, pool)
    else:
        run("add", 50000, pool)   # control: fast kernel
        run("pow", 50000, pool)   # bug: slow kernel


if __name__ == "__main__":
    main()
