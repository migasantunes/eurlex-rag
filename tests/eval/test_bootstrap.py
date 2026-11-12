"""Bootstrap intervals: properties that must hold whatever the data."""

from __future__ import annotations

import pytest

from eurlex_rag.eval.bootstrap import bootstrap_ci, paired_bootstrap_diff, percentile


def test_constant_values_give_zero_width() -> None:
    assert bootstrap_ci([0.7] * 20) == pytest.approx((0.7, 0.7))


def test_interval_contains_the_mean() -> None:
    values = [0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1]
    low, high = bootstrap_ci(values)
    observed = sum(values) / len(values)
    assert low < observed < high
    assert 0 <= low and high <= 1


def test_same_seed_same_result() -> None:
    values = [0.1, 0.9, 0.4, 0.3, 0.8, 0.0]
    assert bootstrap_ci(values, seed=3) == bootstrap_ci(values, seed=3)
    assert bootstrap_ci(values, seed=3) != bootstrap_ci(values, seed=4)


def test_more_questions_narrow_the_interval() -> None:
    small = [0, 1] * 10
    large = [0, 1] * 200
    low_s, high_s = bootstrap_ci(small)
    low_l, high_l = bootstrap_ci(large)
    assert (high_l - low_l) < (high_s - low_s)


def test_paired_identical_lists() -> None:
    a = [0.2, 0.5, 1.0, 0.0]
    diff, low, high, share = paired_bootstrap_diff(a, list(a))
    assert (diff, low, high) == (0.0, 0.0, 0.0)
    assert share == 0.0  # b is never strictly better


def test_paired_clear_separation() -> None:
    a = [0.0, 0.1, 0.0, 0.2] * 10
    b = [x + 0.5 for x in a]
    diff, low, high, share = paired_bootstrap_diff(a, b)
    assert diff == pytest.approx(0.5)
    assert low == pytest.approx(0.5) and high == pytest.approx(0.5)
    assert share == pytest.approx(1.0)


def test_paired_noisy_but_better() -> None:
    a = [0, 0, 1, 0, 1, 0, 0, 1, 0, 0] * 3
    b = [1, 0, 1, 1, 1, 0, 1, 1, 0, 1] * 3
    diff, low, high, share = paired_bootstrap_diff(a, b)
    assert low <= diff <= high
    assert share > 0.95


def test_bad_inputs() -> None:
    with pytest.raises(ValueError):
        bootstrap_ci([])
    with pytest.raises(ValueError):
        bootstrap_ci([1.0], confidence=1.5)
    with pytest.raises(ValueError):
        paired_bootstrap_diff([1.0], [1.0, 2.0])


def test_percentile_interpolates() -> None:
    assert percentile([0.0, 10.0], 0.5) == 5.0
    assert percentile([1.0, 2.0, 3.0], 0.0) == 1.0
    assert percentile([1.0, 2.0, 3.0], 1.0) == 3.0
    assert percentile([4.0], 0.95) == 4.0
