"""Chain-level equivalence tests for the IR graph + kernel fusion module.

Each (template, seed) samples a slot fill once, then draws randomized inputs a
handful of times. A chain exercises many ops at once, so a few draws per fill is
enough -- far fewer iterations than the single-op stress suite needs.
"""

import numpy as np
import pytest
import rushlite

from templates import TEMPLATES
from ops import sample_fills
from runner import run_once

DRAWS = 4
SEEDS = list(range(16))


def _cuda_available():
    try:
        rushlite.Variable(
            [[0.0]], device=rushlite.device.cuda, dtype=rushlite.dtype.float64
        )
        return True
    except Exception:
        return False


def _devices():
    devices = {"cpu": rushlite.device.cpu}
    if _cuda_available():
        devices["cuda"] = rushlite.device.cuda
    return devices


@pytest.mark.parametrize("device", _devices().values(), ids=_devices().keys())
@pytest.mark.parametrize("template", TEMPLATES, ids=[t.name for t in TEMPLATES])
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("fusion", [True, False])
def test_graph(template, device, seed, fusion):
    rng = np.random.default_rng(seed)
    fills = sample_fills(template, rng)
    for _ in range(DRAWS):
        rushlite.capture(enable=fusion)(run_once)(template, fills, device, rng)
