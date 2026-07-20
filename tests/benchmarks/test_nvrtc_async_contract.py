"""Contract: the fused NVRTC launch path must not synchronize internally.

Eager CUDA kernel launches are asynchronous; the benchmark timer owns all
synchronization via rushlite.cuda.sync(). If the fused path regained an
internal cuCtxSynchronize after cuLaunchKernel, every fused/eager latency
comparison would silently become unfair. This source-level check documents
and enforces that contract.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
NVRTC_BACKEND = REPO_ROOT / "csrc/src/inductor/nvrtc/nvrtc_backend.cpp"


def test_fused_launch_has_no_internal_synchronize():
    source = NVRTC_BACKEND.read_text()
    assert "cuLaunchKernel" in source  # sanity: still the launch site
    assert "cuCtxSynchronize" not in source, (
        "fused kernel launches must stay asynchronous; timing code "
        "synchronizes explicitly via rushlite.cuda.sync()"
    )
