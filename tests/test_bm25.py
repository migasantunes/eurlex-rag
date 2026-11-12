"""BM25: the arithmetic checked by hand, then its behaviour on the tiny corpus."""

from __future__ import annotations

import math

import pytest

from eurlex_rag.retrieve.bm25 import BM25Retriever, tokenize
from eurlex_rag.types import Chunk


def test_tokenize() -> None:
    assert tokenize("High-risk AI systems, Article 6(2)!") == [
        "high",
        "risk",
        "ai",
        "systems",
        "article",
        "6",
        "2",
    ]
    assert tokenize("Violação de DADOS") == ["violação", "de", "dados"]
    assert tokenize("") == []


def test_score_matches_the_formula_by_hand() -> None:
    chunks = [
        Chunk("AIA:art:1", "AIA:art:1", "cat cat dog"),
        Chunk("AIA:art:2", "AIA:art:2", "dog bird"),
        Chunk("AIA:art:3", "AIA:art:3", "fish"),
    ]
    k1, b = 1.5, 0.75
    retriever = BM25Retriever(k1=k1, b=b)
    retriever.index(chunks)

    (hit,) = retriever.search("cat", k=5)
    assert hit.chunk_id == "AIA:art:1"

    n, df, tf, length, avg = 3, 1, 2, 3, (3 + 2 + 1) / 3
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
    expected = idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / avg))
    assert hit.score == pytest.approx(expected)


def test_rare_words_outweigh_common_ones() -> None:
    chunks = [
        Chunk("AIA:art:1", "AIA:art:1", "system system system"),
        Chunk("AIA:art:2", "AIA:art:2", "system biometric"),
        Chunk("AIA:art:3", "AIA:art:3", "system other"),
    ]
    retriever = BM25Retriever()
    retriever.index(chunks)
    assert retriever.search("system biometric", k=1)[0].chunk_id == "AIA:art:2"


def test_ranks_scores_and_k(chunks: list[Chunk]) -> None:
    retriever = BM25Retriever()
    retriever.index(chunks)
    hits = retriever.search("safety component of a product", k=3)
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    assert all(a.score >= b.score for a, b in zip(hits, hits[1:], strict=False))
    assert len(hits) <= 3
    assert retriever.search("safety", k=0) == []


def test_english_query_finds_the_right_article(chunks: list[Chunk]) -> None:
    retriever = BM25Retriever()
    retriever.index(chunks)
    assert retriever.search("notify the authority of a data breach", k=1)[0].article_id == (
        "GDPR:art:33"
    )
    assert retriever.search("delete personal data", k=1)[0].article_id == "GDPR:art:17"


def test_portuguese_query_misses_english_text(chunks: list[Chunk]) -> None:
    """The reason the ladder exists: keyword search does not cross languages."""
    retriever = BM25Retriever()
    retriever.index(chunks)
    hits = retriever.search("apagar os meus dados pessoais", k=5)
    assert "GDPR:art:17" not in [h.article_id for h in hits]


def test_no_match_and_empty_index() -> None:
    retriever = BM25Retriever()
    assert retriever.search("anything") == []
    retriever.index([Chunk("AIA:art:1", "AIA:art:1", "alpha beta")])
    assert retriever.search("gamma") == []


def test_results_are_repeatable_on_ties() -> None:
    chunks = [Chunk(f"AIA:art:{i}", f"AIA:art:{i}", "same words") for i in (3, 1, 2)]
    retriever = BM25Retriever()
    retriever.index(chunks)
    assert [h.chunk_id for h in retriever.search("same")] == ["AIA:art:1", "AIA:art:2", "AIA:art:3"]


def test_reindexing_replaces_the_old_index() -> None:
    retriever = BM25Retriever()
    retriever.index([Chunk("AIA:art:1", "AIA:art:1", "alpha")])
    retriever.index([Chunk("AIA:art:2", "AIA:art:2", "beta")])
    assert retriever.search("alpha") == []
    assert retriever.search("beta")[0].chunk_id == "AIA:art:2"


def test_bad_parameters() -> None:
    with pytest.raises(ValueError):
        BM25Retriever(b=1.5)


def test_describe() -> None:
    assert BM25Retriever(k1=1.2, b=0.5).describe() == {
        "name": "01_bm25",
        "method": "bm25",
        "k1": 1.2,
        "b": 0.5,
    }
