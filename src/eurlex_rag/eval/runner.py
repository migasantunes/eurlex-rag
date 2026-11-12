"""Running one retriever over the benchmark and summarising the scores.

The runner is deliberately ignorant of how a retriever works. It hands each
question text to ``retriever.search`` exactly as a user would type it, collapses
the chunk hits to a ranking of articles, and scores that ranking against the
gold IDs. Every rung goes through this same function, which is what makes their
numbers comparable.

What comes out is a plain ``dict`` (JSON-ready) with three levels:

* ``per_query`` - one row per (question, language): the ranking and its scores.
  Kept so any number can be traced back to the queries behind it.
* ``by_lang``   - the mean of each metric per query language, with a 95%
  bootstrap interval. The gap between English and the other languages is the
  cross-lingual result (docs/EVAL.md §4).
* ``pooled``    - one value per question (its mean over languages), then the
  mean and interval over questions. Averaging first means a question asked in
  three languages counts once, not three times, in the uncertainty.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from eurlex_rag.eval.bootstrap import bootstrap_ci, percentile
from eurlex_rag.eval.gold import Question, query_pairs
from eurlex_rag.eval.metrics import mean, ndcg_at_k, recall_at_k, reciprocal_rank
from eurlex_rag.retrieve.base import Retriever, article_ranking

# The metric names used in result files, reports and the CLI. One spelling.
METRICS = ("recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10")


def score_ranking(ranking: Sequence[str], gold: Sequence[str]) -> dict[str, float]:
    """All metrics for one ranked list of article IDs."""
    return {
        "recall@1": recall_at_k(ranking, gold, 1),
        "recall@5": recall_at_k(ranking, gold, 5),
        "recall@10": recall_at_k(ranking, gold, 10),
        "mrr@10": reciprocal_rank(ranking, gold, 10),
        "ndcg@10": ndcg_at_k(ranking, gold, 10),
    }


def summarise(values: Sequence[float], seed: int = 0) -> dict[str, float]:
    """Mean and 95% bootstrap interval of a list of scores."""
    low, high = bootstrap_ci(values, seed=seed)
    return {"mean": mean(values), "low": low, "high": high}


def run_retrieval_eval(
    retriever: Retriever,
    questions: Sequence[Question],
    langs: Sequence[str],
    k: int = 10,
) -> dict:
    """Score an already indexed ``retriever`` on ``questions`` in ``langs``.

    ``k`` is how many chunks are asked for per query. The metrics are always
    cut at 1, 5 and 10 articles, so ``k`` below 10 means @10 only sees ``k``
    chunks. With paragraph chunking several chunks can belong to the same
    article, so the article ranking can be shorter than ``k``.

    Tier C questions (no gold articles) have no retrieval score and are left
    out; how many were left out is returned as ``excluded_unanswerable``.
    """
    answerable = [q for q in questions if q.answerable]
    excluded = len(questions) - len(answerable)

    per_query: list[dict] = []
    latencies: list[float] = []
    for question, lang in query_pairs(answerable, langs):
        start = time.perf_counter()
        hits = retriever.search(question.text[lang], k)
        latency_ms = (time.perf_counter() - start) * 1000
        ranking = article_ranking(hits)
        latencies.append(latency_ms)
        per_query.append(
            {
                "qid": question.qid,
                "lang": lang,
                "tier": question.tier,
                "gold": list(question.gold),
                "ranking": ranking,
                **score_ranking(ranking, question.gold),
                "latency_ms": latency_ms,
            }
        )

    by_lang: dict[str, dict] = {}
    for lang in langs:
        rows = [row for row in per_query if row["lang"] == lang]
        if not rows:
            continue
        by_lang[lang] = {"n": len(rows)}
        for metric in METRICS:
            by_lang[lang][metric] = summarise([row[metric] for row in rows])

    # One value per question: the mean of its languages. This is the unit the
    # bootstrap resamples, and the list a paired comparison between rungs uses.
    per_question: list[dict] = []
    for question in answerable:
        rows = [row for row in per_query if row["qid"] == question.qid]
        if not rows:
            continue
        entry: dict = {"qid": question.qid, "n_langs": len(rows)}
        for metric in METRICS:
            entry[metric] = mean(row[metric] for row in rows)
        per_question.append(entry)

    pooled: dict = {"n": len(per_question)}
    if per_question:
        for metric in METRICS:
            pooled[metric] = summarise([entry[metric] for entry in per_question])

    ordered = sorted(latencies)
    latency = (
        {"p50": percentile(ordered, 0.50), "p95": percentile(ordered, 0.95)}
        if ordered
        else {"p50": None, "p95": None}
    )

    return {
        "k": k,
        "langs": list(langs),
        "n_questions": len(questions),
        "n_queries": len(per_query),
        "excluded_unanswerable": excluded,
        "by_lang": by_lang,
        "pooled": pooled,
        "latency_ms": latency,
        "per_question": per_question,
        "per_query": per_query,
    }
