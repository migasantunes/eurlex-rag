"""Confidence intervals by bootstrap resampling.

Why: a Recall@5 of 0.62 on 60 questions is one draw. Another 60 questions of the
same kind would give a different number. The bootstrap estimates how different
without any distributional assumption: pretend the questions we have are the
population, draw a new set of the same size *with replacement* many times, and
look at how much the mean moves. The middle 95% of those means is the 95%
interval.

With about 60 test questions the intervals are wide (roughly +-0.1 for a
recall near 0.5). That is the honest state of things: two rungs whose intervals
overlap heavily are not shown to differ, and the write-up says so.

To compare two rungs, ``paired_bootstrap_diff`` resamples the *same* questions
for both, so a question that is hard for every rung does not add noise to the
difference. This is much tighter than comparing two separate intervals.

The resampling unit is the **question**. When a question is asked in three
languages, those three queries are not independent evidence; the caller first
averages them into one value per question and passes that.

Everything is seeded through ``random.Random(seed)``, so a re-run gives the same
interval.
"""

from __future__ import annotations

import random
from collections.abc import Sequence


def percentile(sorted_values: Sequence[float], fraction: float) -> float:
    """Value at ``fraction`` (0..1) of a sorted list, interpolating linearly."""
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = fraction * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _check(confidence: float, n_resamples: int) -> None:
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")


def bootstrap_ci(
    values: Sequence[float],
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval ``(low, high)`` for the mean of ``values``.

    Each resample draws ``len(values)`` values with replacement and takes the
    mean; the interval is the ``(1 - confidence) / 2`` and ``(1 + confidence) / 2``
    percentiles of those means.
    """
    _check(confidence, n_resamples)
    values = list(values)
    if not values:
        raise ValueError("bootstrap_ci needs at least one value")
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(n_resamples))
    alpha = (1 - confidence) / 2
    return percentile(means, alpha), percentile(means, 1 - alpha)


def paired_bootstrap_diff(
    a: Sequence[float],
    b: Sequence[float],
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float, float]:
    """Compare two systems scored on the same questions.

    ``a[i]`` and ``b[i]`` must be the scores of the same question. Each resample
    draws question positions with replacement and takes the mean of
    ``b - a`` over them.

    Returns ``(mean_diff, low, high, share_b_better)``: the observed mean of
    ``b - a``, its percentile interval, and the share of resamples in which b
    beat a. A share near 0.5 means "no detectable difference"; near 1 means b
    is better on almost any resample of the questions.
    """
    _check(confidence, n_resamples)
    if len(a) != len(b):
        raise ValueError(f"paired lists differ in length ({len(a)} vs {len(b)})")
    if not a:
        raise ValueError("paired_bootstrap_diff needs at least one pair")
    diffs = [y - x for x, y in zip(a, b, strict=True)]
    n = len(diffs)
    rng = random.Random(seed)
    resampled = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(n_resamples))
    alpha = (1 - confidence) / 2
    share_b_better = sum(1 for d in resampled if d > 0) / n_resamples
    return (
        sum(diffs) / n,
        percentile(resampled, alpha),
        percentile(resampled, 1 - alpha),
        share_b_better,
    )
