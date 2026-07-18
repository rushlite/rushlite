"""Catalog tests: predefined cases and enforced axes."""

import pytest

from benchmarks.fusion.spec import FusionCase, RunConfig, resolve_cases


def test_short_tag_cases():
    cases = resolve_cases(RunConfig(tags=("short",)))
    assert cases == [
        FusionCase("gru", 256, 256, 1, "cuda", "float32", "short"),
        FusionCase("lstm", 256, 256, 1, "cuda", "float32", "short"),
    ]


def test_long_tag_sizes():
    cases = resolve_cases(RunConfig(tags=("long",), workloads=("gru",)))
    assert [c.hidden_size for c in cases] == [64, 128, 256, 512, 1024]
    assert all(c.input_size == c.hidden_size for c in cases)
    assert all(c.batch_size == 1 for c in cases)


def test_all_cases_are_cuda_float32():
    cases = resolve_cases(RunConfig(tags=("short", "long")))
    assert cases
    assert all(c.device == "cuda" and c.dtype == "float32" for c in cases)


def test_batch_size_other_than_one_rejected():
    with pytest.raises(ValueError, match="batch size 2"):
        resolve_cases(RunConfig(batch_sizes=(2,)))
    with pytest.raises(ValueError, match="batch size 8"):
        resolve_cases(RunConfig(batch_sizes=(1, 8)))


def test_unknown_axes_rejected():
    with pytest.raises(ValueError, match="workload"):
        resolve_cases(RunConfig(workloads=("rnn",)))
    with pytest.raises(ValueError, match="tag"):
        resolve_cases(RunConfig(tags=("medium",)))
    with pytest.raises(ValueError, match="mode"):
        resolve_cases(RunConfig(modes=("graph",)))
