"""Retrieval metrics on one ranked list of article IDs.

Each function takes the ranking a retriever returned (best first, already
collapsed to distinct article IDs) and the set of gold article IDs, and returns
a number between 0 and 1. Higher is better.

A question with no gold articles (tier C, unanswerable) has no right answer to
find, so these functions refuse it with ``ValueError``. The runner leaves such
questions out of the retrieval numbers and reports how many it left out.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def _check(gold: Iterable[str], k: int) -> set[str]:
    gold_set = set(gold)
    if not gold_set:
        raise ValueError("Empty gold set: unanswerable questions have no retrieval score")
    if k <= 0:
        raise ValueError("k must be positive")
    return gold_set


def recall_at_k(ranking: Sequence[str], gold: Iterable[str], k: int) -> float:
    """Share of the gold articles found in the top ``k``.

    Intuition: did the right articles reach the LLM at all?

    Example: ranking ``[a, x, b]``, gold ``{a, b, c}``, k=2 -> only ``a`` is in
    the top 2, so recall = 1/3.
    """
    gold_set = _check(gold, k)
    found = gold_set & set(ranking[:k])
    return len(found) / len(gold_set)


def reciprocal_rank(ranking: Sequence[str], gold: Iterable[str], k: int = 10) -> float:
    """One over the rank of the first gold article; 0 if none is in the top ``k``.

    Intuition: how far down does the reader have to look before the first
    correct article? Rank 1 scores 1, rank 2 scores 0.5, rank 4 scores 0.25.
    Its mean over questions is MRR.

    Example: ranking ``[x, y, a]``, gold ``{a}`` -> first hit at rank 3 -> 1/3.
    With k=2 the hit is cut off -> 0.
    """
    gold_set = _check(gold, k)
    for rank, article_id in enumerate(ranking[:k], start=1):
        if article_id in gold_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranking: Sequence[str], gold: Iterable[str], k: int = 10) -> float:
    """Normalised discounted cumulative gain, with yes/no relevance.

    Intuition: every gold article found earns a point, but a point at rank r is
    divided by log2(r + 1), so finding it lower counts for less. The total is
    divided by the best total possible (all gold articles at the top), so a
    perfect ranking scores 1 whatever the number of gold articles.

        DCG  = sum over ranks r <= k where ranking[r] is gold of 1 / log2(r + 1)
        IDCG = sum over r = 1 .. min(len(gold), k) of 1 / log2(r + 1)

    Example: ranking ``[x, a, b]``, gold ``{a, b}``, k=10 ->
    DCG = 1/log2(3) + 1/log2(4) = 0.631 + 0.5 = 1.131;
    IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.631 = 1.631; nDCG = 0.693.
    """
    gold_set = _check(gold, k)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, article_id in enumerate(ranking[:k], start=1)
        if article_id in gold_set
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold_set), k) + 1))
    return dcg / ideal


def mean(values: Iterable[float]) -> float:
    """Arithmetic mean. Raises ``ValueError`` on an empty input rather than
    returning a misleading 0."""
    values = list(values)
    if not values:
        raise ValueError("mean of an empty list")
    return sum(values) / len(values)
