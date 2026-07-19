"""Op catalog, config helpers, and shape-key generation.

Mirrors benchmark_utils.cross_product_configs and config_list, stripped of torch.
All canonical shape keys are generated here so both frameworks emit identical strings.
"""

import itertools
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Config helpers (mirrors benchmark_utils.py)
# ---------------------------------------------------------------------------


def cross_product_configs(**axes) -> list[tuple[dict, ...]]:
    """Return the cartesian product of all axis values.

    Example:
        cross_product_configs(device=["cpu","cuda"], dtype=["float32"])
        -> (({'device':'cpu'},{'dtype':'float32'}),
            ({'device':'cuda'},{'dtype':'float32'}))
    """
    per_axis = []
    for key, values in axes.items():
        per_axis.append([{key: v} for v in values])
    return list(itertools.product(*per_axis))


def config_list(
    attr_names: list[str],
    attrs: list[list[Any]],
    cross_product_configs: list[tuple[dict, ...]] | None = None,
    tags: list[str] | None = None,
) -> list[list[dict]]:
    """Generate configs from a list of input shapes, optionally cross-produced.

    Args:
        attr_names: names for each positional attr (e.g. ["R", "V"]).
        attrs: list of value tuples, one per config row.
        cross_product_configs: pre-computed cross product (from cross_product_configs()).
        tags: tag strings joined with "_" and appended as {"tags": ...}.

    Returns:
        List of config rows, each a list of single-key dicts ending with {"tags":...}.
    """
    if tags is None:
        tags = []
    tag_dict = {"tags": "_".join(tags)}

    generated: list[list[dict]] = []
    for row in attrs:
        base = [{attr_names[i]: v} for i, v in enumerate(row)]
        base.append(tag_dict)
        if cross_product_configs:
            for combo in cross_product_configs:
                generated.append(base + list(combo))
        else:
            generated.append(base)
    return generated


# ---------------------------------------------------------------------------
# Canonical shape-key generation
# ---------------------------------------------------------------------------


def shape_key(*, in_one=None, in_two=None, R=None, V=None, dim=None, **_extra) -> str:
    """Return a canonical string key for a shape config.

    Binary same-size / unary:  "512x512"
    Binary broadcast:          "64x1x64|1x64x1"
    Reduction:                 "64x512@dim1"   (shape is R x V unless dim==0 gives V x R)
    """
    if R is not None:
        # reduction: shape is (R, V) when dim==0, (V, R) when dim==1
        if dim == 0:
            rows, cols = R, V
        else:
            rows, cols = V, R
        return f"{rows}x{cols}@dim{dim}"

    if in_two is not None:
        # binary
        s1 = "x".join(str(d) for d in in_one)
        s2 = "x".join(str(d) for d in in_two)
        if in_one == in_two:
            return s1
        return f"{s1}|{s2}"

    # unary
    return "x".join(str(d) for d in in_one)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BenchCase:
    """A single fully-resolved benchmark case."""

    op: str
    category: str  # "binary" | "unary" | "reduction"
    device: str
    dtype: str
    shape: str  # canonical key from shape_key()
    tag: str
    params: dict = field(default_factory=dict)

    # resolved from the raw config row - kept so runner can reconstruct inputs
    raw: dict = field(default_factory=dict)


@dataclass
class OpEntry:
    """One operator with its full config list."""

    name: str
    category: str  # "binary" | "unary" | "reduction"
    configs: list[list[dict]]  # each row is a list of single-key dicts
    input_recipe: str  # "randn" | "rand"
    params: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Op catalog
# ---------------------------------------------------------------------------

_DEVICES = ["cpu", "cuda"]
_DTYPE = "float32"

# Cross-product base used by binary and unary
_cross_base = cross_product_configs(device=_DEVICES)

# --- Binary ---

_binary_same_size = config_list(
    attr_names=["in_one", "in_two"],
    attrs=[
        [[128, 128], [128, 128]],
        [[256, 256], [256, 256]],
        [[512, 512], [512, 512]],
        [[1024, 1024], [1024, 1024]],
    ],
    cross_product_configs=_cross_base,
    tags=["long"],
)

_binary_broadcast = config_list(
    attr_names=["in_one", "in_two"],
    attrs=[
        [[64, 1, 64], [1, 64, 1]],
    ],
    cross_product_configs=_cross_base,
    tags=["long"],
)

_binary_short = config_list(
    attr_names=["in_one", "in_two"],
    attrs=[
        [[512, 512], [512, 512]],
    ],
    cross_product_configs=_cross_base,
    tags=["short"],
)

_binary_configs = _binary_same_size + _binary_broadcast + _binary_short

_BINARY_OPS = ["add", "sub", "mul", "div", "pow"]

# --- Unary ---

_unary_long = config_list(
    attr_names=["in_one"],
    attrs=[
        [[128, 128]],
        [[256, 256]],
        [[512, 512]],
        [[1024, 1024]],
    ],
    cross_product_configs=_cross_base,
    tags=["long"],
)

_unary_short = config_list(
    attr_names=["in_one"],
    attrs=[[[512, 512]]],
    cross_product_configs=_cross_base,
    tags=["short"],
)

_unary_configs = _unary_long + _unary_short

_UNARY_OPS = ["abs", "clamp", "cos", "exp", "log", "neg", "sin", "sqrt", "tan"]

# --- Reduction ---

# Keep this workflow reduction-only while iterating on the CUDA reduction
# kernels. Each physical matrix shape is benchmarked along both axes.
_reduction_shapes = [
    [512, 4096],
    [4096, 512],
    [1024, 2048],
    [512, 64],
    [256, 128],
]

_reduction_configs: list[list[dict]] = []
for _rows, _columns in _reduction_shapes:
    for _dim in (0, 1):
        # The reduction harness represents dim 0 as shape [R, V] and dim 1 as
        # shape [V, R], so swap the metadata to preserve the physical shape.
        _R, _V = (
            (_rows, _columns) if _dim == 0 else (_columns, _rows)
        )
        _reduction_configs.append(
            [
                {"R": _R},
                {"V": _V},
                {"dim": _dim},
                {"device": "cuda"},
                {"tags": "long"},
            ]
        )

_REDUCTION_OPS = ["sum", "min", "max", "prod"]


def _build_catalog() -> list[OpEntry]:
    catalog: list[OpEntry] = []

    # Binary and unary entries are intentionally omitted while this benchmark
    # workflow is focused on the CUDA reduction kernels.
    for op in _REDUCTION_OPS:
        catalog.append(
            OpEntry(
                name=op,
                category="reduction",
                configs=_reduction_configs,
                input_recipe="rand",
            )
        )

    return catalog


CATALOG: list[OpEntry] = _build_catalog()


def _row_to_dict(row: list[dict]) -> dict:
    """Flatten a config row (list of single-key dicts) into one dict."""
    out: dict = {}
    for d in row:
        out.update(d)
    return out


def resolve_cases(entry: OpEntry) -> list[BenchCase]:
    """Expand all config rows of an OpEntry into BenchCase objects."""
    cases: list[BenchCase] = []
    for row in entry.configs:
        flat = _row_to_dict(row)
        tag = flat.get("tags", "")
        device = flat.get("device", "cpu")
        key = shape_key(**flat)
        cases.append(
            BenchCase(
                op=entry.name,
                category=entry.category,
                device=device,
                dtype=_DTYPE,
                shape=key,
                tag=tag,
                params=entry.params,
                raw=flat,
            )
        )
    return cases
