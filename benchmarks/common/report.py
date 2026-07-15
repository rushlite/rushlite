"""CSV report writer.

Schema: framework, op, category, direction, device, dtype, shape, tag, iters, runs, time_us

One file per backend: benchmarks/results/<machine>/<backend>.csv
Rows append incrementally (read-existing, rewrite), matching pytorch's _output_csv approach.
Rows are comparable across backends by exact string match on (op,category,direction,device,dtype,shape,tag).
"""

import csv
import os
import platform

HEADERS = [
    "framework",
    "op",
    "category",
    "direction",
    "device",
    "dtype",
    "shape",
    "tag",
    "iters",
    "runs",
    "time_us",
]


def _results_dir(output_dir: str | None) -> str:
    if output_dir:
        return output_dir
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(here, "results", platform.node())


def csv_path(backend_name: str, output_dir: str | None = None) -> str:
    """Return the output CSV path for a given backend."""
    return os.path.join(_results_dir(output_dir), f"{backend_name}.csv")


def append_row(
    path: str,
    *,
    framework: str,
    op: str,
    category: str,
    direction: str,
    device: str,
    dtype: str,
    shape: str,
    tag: str,
    iters: int,
    runs: int,
    time_us: float,
) -> None:
    """Append one measurement row to path, creating the file (and dirs) if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path):
        with open(path, newline="") as fh:
            lines = list(csv.reader(fh)) or [HEADERS]
        if not lines:
            lines = [HEADERS]
        # fix header if file was written with wrong/empty header
        if len(lines[0]) < len(HEADERS):
            lines[0] = HEADERS
    else:
        lines = [HEADERS]

    row = [
        framework,
        op,
        category,
        direction,
        device,
        dtype,
        shape,
        tag,
        str(iters),
        str(runs),
        f"{time_us:.6f}",
    ]
    lines.append(row)

    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        for line in lines:
            # pad short rows to header width (matches pytorch _output_csv)
            padded = list(line) + ["0"] * (len(HEADERS) - len(line))
            writer.writerow(padded)
