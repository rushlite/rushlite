"""CSV writing and eager/fused pairing for the fusion benchmarks.

Raw per-mode rows are the primary output. The summary pairs exact cases and
reports absolute savings, per-case speedup (eager_us / fused_us), the
geometric mean over the selected set, and any regressions. It deliberately
does not headline the maximum speedup.
"""

from __future__ import annotations

import csv
import math
from dataclasses import astuple, dataclass, fields
from pathlib import Path
from typing import Sequence

from .runner import ResultRow

CSV_HEADERS = [f.name for f in fields(ResultRow)]


@dataclass(frozen=True)
class PairedResult:
    workload: str
    device: str
    dtype: str
    input_size: int
    hidden_size: int
    batch_size: int
    tag: str
    eager_us: float
    fused_us: float

    @property
    def saved_us(self) -> float:
        return self.eager_us - self.fused_us

    @property
    def speedup(self) -> float:
        return self.eager_us / self.fused_us


def write_results(path: Path, rows: Sequence[ResultRow]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(CSV_HEADERS)
        for row in rows:
            writer.writerow(astuple(row))


def _case_key(row: ResultRow) -> tuple:
    return (
        row.workload,
        row.device,
        row.dtype,
        row.input_size,
        row.hidden_size,
        row.batch_size,
        row.sequence_steps,
        row.tag,
        row.seed,
    )


def pair_results(rows: Sequence[ResultRow]) -> list[PairedResult]:
    """Pair rows whose case fields match exactly across the two modes.

    Cases that were run in only one mode are skipped; a duplicated
    (case, mode) combination is an error.
    """
    by_case: dict[tuple, dict[str, ResultRow]] = {}
    for row in rows:
        modes = by_case.setdefault(_case_key(row), {})
        if row.mode in modes:
            raise ValueError(f"duplicate result for {row.mode} {_case_key(row)}")
        modes[row.mode] = row

    paired: list[PairedResult] = []
    for key in sorted(by_case):
        modes = by_case[key]
        if "eager" not in modes or "fused" not in modes:
            continue
        eager, fused = modes["eager"], modes["fused"]
        paired.append(
            PairedResult(
                workload=eager.workload,
                device=eager.device,
                dtype=eager.dtype,
                input_size=eager.input_size,
                hidden_size=eager.hidden_size,
                batch_size=eager.batch_size,
                tag=eager.tag,
                eager_us=eager.median_us,
                fused_us=fused.median_us,
            )
        )
    return paired


def geometric_mean_speedup(paired: Sequence[PairedResult]) -> float:
    if not paired:
        raise ValueError("no paired results")
    return math.exp(sum(math.log(p.speedup) for p in paired) / len(paired))


def print_summary(rows: Sequence[ResultRow]) -> None:
    paired = pair_results(rows)
    if not paired:
        print("no eager/fused pairs to summarize")
        return

    header = (
        f"{'workload':8s}  {'tag':5s}  {'I':>5s}  {'H':>5s}  {'B':>2s}"
        f"  {'eager_us':>12s}  {'fused_us':>12s}  {'saved_us':>12s}"
        f"  {'speedup':>8s}"
    )
    print(header)
    print("-" * len(header))
    for p in paired:
        print(
            f"{p.workload:8s}  {p.tag:5s}  {p.input_size:5d}"
            f"  {p.hidden_size:5d}  {p.batch_size:2d}"
            f"  {p.eager_us:12.3f}  {p.fused_us:12.3f}  {p.saved_us:12.3f}"
            f"  {p.speedup:8.3f}"
        )

    print(f"\ngeometric mean speedup: {geometric_mean_speedup(paired):.3f}")

    regressions = [p for p in paired if p.speedup < 1.0]
    if regressions:
        print(f"regressions ({len(regressions)}):")
        for p in regressions:
            print(
                f"  {p.workload} H={p.hidden_size} B={p.batch_size}"
                f" tag={p.tag}: {p.speedup:.3f}x"
            )
    else:
        print("regressions: none")
