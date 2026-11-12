"""Retrieval metrics, checked against numbers worked out by hand."""

from __future__ import annotations

import math

import pytest

from eurlex_rag.eval.metrics import mean, ndcg_at_k, recall_at_k, reciprocal_rank


def test_recall_single_gold() -> None:
    assert recall_at_k(["a", "b", "c"], {"b"}, 1) == 0.0
    assert recall_at_k(["a", "b", "c"], {"b"}, 2) == 1.0


def test_recall_multi_gold_and_cutoff() -> None:
    # Top 2 holds a only, out of {a, b, c}.
    assert recall_at_k(["a", "x", "b"], {"a", "b", "c"}, 2) == pytest.approx(1 / 3)
    assert recall_at_k(["a", "x", "b"], {"a", "b", "c"}, 3) == pytest.approx(2 / 3)


def test_recall_gold_not_retrieved() -> None:
    assert recall_at_k(["x", "y"], {"a"}, 10) == 0.0
    assert recall_at_k([], {"a"}, 10) == 0.0


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(["a", "b"], {"a"}) == 1.0
    assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)
    # Only the first gold hit counts.
    assert reciprocal_rank(["x", "a", "b"], {"a", "b"}) == 0.5
    # Cut off at k.
    assert reciprocal_rank(["x", "y", "a"], {"a"}, k=2) == 0.0
    assert reciprocal_rank(["x"], {"a"}) == 0.0


def test_ndcg_perfect_and_missing() -> None:
    assert ndcg_at_k(["a", "b", "x"], {"a", "b"}) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "y"], {"a"}) == 0.0


def test_ndcg_multi_gold_by_hand() -> None:
    # Gold at ranks 2 and 3: DCG = 1/log2(3) + 1/log2(4); IDCG = 1 + 1/log2(3).
    expected = (1 / math.log2(3) + 0.5) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(["x", "a", "b"], {"a", "b"}) == pytest.approx(expected)
    assert expected == pytest.approx(0.6934, abs=1e-4)


def test_ndcg_single_gold_lower_rank() -> None:
    # One gold at rank 2: (1/log2(3)) / 1.
    assert ndcg_at_k(["x", "a"], {"a"}) == pytest.approx(1 / math.log2(3))


def test_ndcg_ideal_is_capped_at_k() -> None:
    # Three gold, k=2, both top slots correct: ideal also has only 2 slots -> 1.
    assert ndcg_at_k(["a", "b", "c"], {"a", "b", "c"}, k=2) == pytest.approx(1.0)
    # Third gold beyond the cut-off is ignored.
    assert ndcg_at_k(["x", "y", "a"], {"a"}, k=2) == 0.0


@pytest.mark.parametrize("metric", [recall_at_k, reciprocal_rank, ndcg_at_k])
def test_empty_gold_raises(metric) -> None:
    with pytest.raises(ValueError, match="Empty gold"):
        metric(["a"], set(), 5)


def test_bad_k_raises() -> None:
    with pytest.raises(ValueError):
        recall_at_k(["a"], {"a"}, 0)


def test_mean() -> None:
    assert mean([1.0, 0.0, 0.5]) == 0.5
    assert mean(x for x in (2.0, 4.0)) == 3.0
    with pytest.raises(ValueError):
        mean([])
