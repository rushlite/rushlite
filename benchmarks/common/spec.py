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

_reduction_long_cross = cross_product_configs(
    R=[64, 256],
    V=[32, 512],
    dim=[0, 1],
    device=_DEVICES,
)

_reduction_long: list[list[dict]] = []
for _combo in _reduction_long_cross:
    _row = list(_combo) + [{"tags": "long"}]
    _reduction_long.append(_row)

# one shape for short: R=64, V=32, dim=0
_reduction_short_cross = cross_product_configs(device=_DEVICES)
_reduction_short: list[list[dict]] = []
for _combo in _reduction_short_cross:
    _row = [{"R": 64}, {"V": 32}, {"dim": 0}] + list(_combo) + [{"tags": "short"}]
    _reduction_short.append(_row)

_reduction_configs = _reduction_long + _reduction_short

_REDUCTION_OPS = ["sum", "min", "max", "prod"]


def _build_catalog() -> list[OpEntry]:
    catalog: list[OpEntry] = []

    for op in _BINARY_OPS:
        params = {"min": 0.25, "max": 0.75} if op == "clamp" else {}
        catalog.append(
            OpEntry(
                name=op,
                category="binary",
                configs=_binary_configs,
                input_recipe="randn",
                params=params,
            )
        )

    for op in _UNARY_OPS:
        params = {"min": 0.25, "max": 0.75} if op == "clamp" else {}
        catalog.append(
            OpEntry(
                name=op,
                category="unary",
                configs=_unary_configs,
                input_recipe="rand",
                params=params,
            )
        )

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
