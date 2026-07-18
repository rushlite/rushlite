"""Behavioral tests for measure_samples: warmup, one-time calibration,
fixed iterations per sample, and sync boundaries."""

import pytest

from benchmarks.common.timer import Measurement, measure_samples


class Recorder:
    """Records the interleaving of run/sync events."""

    def __init__(self):
        self.events: list[str] = []

    def run_once(self):
        self.events.append("run")

    def sync(self):
        self.events.append("sync")


def test_fixed_iterations_event_sequence():
    rec = Recorder()
    result = measure_samples(rec.run_once, rec.sync, warmup=2, samples=3, iterations=4)

    expected = ["sync", "run", "run", "sync"]  # warmup between syncs
    for _ in range(3):  # each sample: sync, 4 runs, sync
        expected += ["sync"] + ["run"] * 4 + ["sync"]
    assert rec.events == expected

    assert result.iterations_per_sample == 4
    assert len(result.samples_us) == 3


def test_calibration_runs_once_and_count_stays_fixed():
    rec = Recorder()
    result = measure_samples(rec.run_once, rec.sync, warmup=1, samples=2)

    # warmup (1 run) + calibration (1 run) + samples (2 * iterations)
    runs = rec.events.count("run")
    assert runs == 1 + 1 + 2 * result.iterations_per_sample
    assert result.iterations_per_sample >= 1

    # every sample ran the same iteration count: samples are the only
    # remaining run blocks and each is bounded by syncs
    sample_events = rec.events[rec.events.index("sync", 3) :]
    blocks = [
        len(block)
        for block in "".join("r" if e == "run" else "|" for e in sample_events).split(
            "|"
        )
        if block
    ]
    assert blocks[-2:] == [result.iterations_per_sample] * 2


def test_zero_warmup_allowed():
    rec = Recorder()
    measure_samples(rec.run_once, rec.sync, warmup=0, samples=1, iterations=1)
    assert rec.events == ["sync", "sync", "sync", "run", "sync"]


def test_invalid_arguments_rejected():
    rec = Recorder()
    with pytest.raises(ValueError):
        measure_samples(rec.run_once, rec.sync, warmup=-1, samples=1)
    with pytest.raises(ValueError):
        measure_samples(rec.run_once, rec.sync, warmup=0, samples=0)
    with pytest.raises(ValueError):
        measure_samples(rec.run_once, rec.sync, warmup=0, samples=1, iterations=0)


def test_measurement_statistics():
    m = Measurement(iterations_per_sample=5, samples_us=(3.0, 1.0, 2.0))
    assert m.median_us == 2.0
    assert m.min_us == 1.0
