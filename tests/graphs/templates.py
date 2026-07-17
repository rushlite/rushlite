"""Hand-authored chain skeletons with swappable, typed op slots.

A template is shape-agnostic wiring: it threads roles together and never
builds tensors or assumes equal shapes. Shapes live in ``input_shapes`` so a
broadcasting variant is a shape swap, not a body change. The runner samples a
concrete op per slot and feeds leaves in.

Slot categories: BIN (safe binary, pure fusible), BIN_BOUNDED (guarded
div/pow), UNARY, REDUCT. MATMUL is a fixed barrier.
"""

from dataclasses import dataclass
from typing import Callable

S = (16, 16)  # square block, fully fusible elementwise
M = (16, 8)  # post-matmul / right-operand block


@dataclass
class Template:
    name: str
    input_shapes: list  # one shape per leaf, in xs order
    slots: list  # category per slot, in build order
    build: Callable  # build(ops, xs) -> head tensor


def _long_build(ops, xs):
    # Pure safe-binary, one shape throughout: a single maximal fused region.
    t = ops.bin[0](xs[0], xs[1])
    t = ops.bin[1](t, xs[2])
    t = ops.bin[2](t, xs[3])
    t = ops.bin[3](t, xs[4])
    t = ops.bin[4](t, xs[5])
    return t


long_fuse_chain = Template(
    name="long_fuse_chain",
    input_shapes=[S, S, S, S, S, S],
    slots=["BIN", "BIN", "BIN", "BIN", "BIN"],
    build=_long_build,
)


def _barrier_build(ops, xs):
    # Fusible runs (each >=2 safe binary ops) separated by barriers, so the IR
    # has to partition correctly around things fusion can't absorb. The bounded
    # binary sits before run 2 (not right before the reduction): its clamp
    # creates value plateaus, and a max/min over an exact tie is subgradient-
    # ambiguous -- the fresh leaves in run 2 break the tie first.
    t = ops.bin[0](xs[0], xs[1])  # fusible run 1: 2 binary on (16,16)
    t = ops.bin[1](t, xs[2])
    t = ops.unary[0](t)  # unary stays in the pre-matmul fusion group
    t = ops.matmul(t, xs[3])  # barrier: matmul (16,16)@(16,8) -> (16,8)
    t = ops.bin_bounded[0](t, xs[4])  # guarded div/pow
    t = ops.bin[2](t, xs[5])  # fusible run 2: 2 binary on (16,8)
    t = ops.bin[3](t, xs[6])
    t = ops.reduct[0](t, 0)  # barrier: reduction -> (1,8)
    return t


barrier_chain = Template(
    name="barrier_chain",
    input_shapes=[S, S, S, M, M, M, M],
    slots=["BIN", "BIN", "UNARY", "BIN_BOUNDED", "BIN", "BIN", "REDUCT"],
    build=_barrier_build,
)


TEMPLATES = [long_fuse_chain, barrier_chain]
