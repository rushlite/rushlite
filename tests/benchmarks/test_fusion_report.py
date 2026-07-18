"""Pairing, speedup, and CSV tests for the fusion report module."""

import csv
import math

import pytest

from benchmarks.fusion.report import (
    CSV_HEADERS,
    geometric_mean_speedup,
    pair_results,
    write_results,
)
from benchmarks.fusion.runner import ResultRow


def make_row(mode, *, workload="gru", hidden=256, median_us=100.0):
    return ResultRow(
        workload=workload,
        mode=mode,
        device="cuda",
        dtype="float32",
        input_size=hidden,
        hidden_size=hidden,
        batch_size=1,
        sequence_steps=1,
        tag="short",
        seed=0,
        iterations_per_sample=50,
        samples=10,
        median_us=median_us,
        min_us=median_us * 0.9,
    )


def test_pairing_matches_exact_cases():
    rows = [
        make_row("eager", workload="gru", median_us=200.0),
        make_row("fused", workload="gru", median_us=100.0),
        make_row("eager", workload="lstm", median_us=300.0),
        make_row("fused", workload="lstm", median_us=400.0),
    ]
    paired = pair_results(rows)
    assert len(paired) == 2

    by_workload = {p.workload: p for p in paired}
    gru = by_workload["gru"]
    assert gru.eager_us == 200.0
    assert gru.fused_us == 100.0
    assert gru.saved_us == 100.0
    assert gru.speedup == 2.0

    lstm = by_workload["lstm"]
    assert lstm.speedup == 0.75  # regression is representable
    assert lstm.saved_us == -100.0


def test_unpaired_rows_are_skipped():
    rows = [make_row("eager", workload="gru")]
    assert pair_results(rows) == []


def test_duplicate_mode_rejected():
    rows = [make_row("eager"), make_row("eager")]
    with pytest.raises(ValueError, match="duplicate"):
        pair_results(rows)


def test_geometric_mean():
    rows = [
        make_row("eager", workload="gru", median_us=400.0),
        make_row("fused", workload="gru", median_us=100.0),  # 4x
        make_row("eager", workload="lstm", median_us=100.0),
        make_row("fused", workload="lstm", median_us=100.0),  # 1x
    ]
    assert geometric_mean_speedup(pair_results(rows)) == pytest.approx(math.sqrt(4.0))


def test_write_results_roundtrip(tmp_path):
    rows = [make_row("eager"), make_row("fused", median_us=50.0)]
    path = tmp_path / "out" / "results.csv"
    write_results(path, rows)

    with path.open(newline="") as fh:
        lines = list(csv.reader(fh))
    assert lines[0] == CSV_HEADERS
    assert len(lines) == 3
    assert lines[1][CSV_HEADERS.index("mode")] == "eager"
    assert float(lines[2][CSV_HEADERS.index("median_us")]) == 50.0
