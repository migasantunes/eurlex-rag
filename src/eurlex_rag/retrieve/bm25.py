"""Rung 1: BM25 keyword search, written out by hand.

BM25 scores a chunk for a query by adding up, for every query word that appears
in the chunk, a term that is

* larger when the word is **rare** across the corpus (inverse document
  frequency, IDF): "the" tells us nothing, "biometric" tells us a lot;
* larger when the word appears **more often** in the chunk, but with
  diminishing returns (controlled by ``k1``): the fifth occurrence adds much
  less than the first;
* scaled down for **long** chunks (controlled by ``b``): a long article
  contains more words by chance, so matches in it count for less.

    score(q, d) = sum over words w in q of
        idf(w) * tf(w, d) * (k1 + 1) / (tf(w, d) + k1 * (1 - b + b * len(d) / avg_len))

    idf(w) = ln(1 + (N - df(w) + 0.5) / (df(w) + 0.5))

where ``tf`` is how often the word occurs in the chunk, ``df`` is how many
chunks contain it, and ``N`` is the number of chunks.

It is written here instead of imported so that every step can be read. It has
no notion of meaning: a Portuguese question shares almost no words with an
English article, so this rung is expected to do badly on cross-lingual queries.
Measuring how badly is the point of the baseline.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

from eurlex_rag.retrieve.base import Retriever
from eurlex_rag.types import Chunk, Hit

# Runs of letters or digits in any script. Punctuation and spaces split tokens.
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lower-case words. No stemming and no stop-word list: IDF already makes
    very common words count for almost nothing."""
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever(Retriever):
    """Okapi BM25 over an in-memory inverted index."""

    def __init__(self, k1: float = 1.5, b: float = 0.75, name: str = "01_bm25") -> None:
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError("BM25 needs k1 >= 0 and 0 <= b <= 1")
        self.name = name
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._lengths: list[int] = []
        self._avg_length = 0.0
        # word -> list of (chunk position, occurrences in that chunk)
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._idf: dict[str, float] = {}

    def index(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        self._lengths = []
        self._postings = {}
        for position, chunk in enumerate(self._chunks):
            counts = Counter(tokenize(chunk.text))
            self._lengths.append(sum(counts.values()))
            for word, count in counts.items():
                self._postings.setdefault(word, []).append((position, count))

        total = len(self._chunks)
        self._avg_length = sum(self._lengths) / total if total else 0.0
        self._idf = {
            word: math.log(1 + (total - len(postings) + 0.5) / (len(postings) + 0.5))
            for word, postings in self._postings.items()
        }

    def search(self, query: str, k: int = 10) -> list[Hit]:
        if k <= 0 or not self._chunks:
            return []
        scores: dict[int, float] = {}
        # A word repeated in the query counts once; BM25's query-side weighting
        # matters for long queries, and questions are short.
        for word in set(tokenize(query)):
            postings = self._postings.get(word)
            if not postings:
                continue
            idf = self._idf[word]
            for position, tf in postings:
                length_norm = 1 - self.b + self.b * self._lengths[position] / self._avg_length
                term = idf * tf * (self.k1 + 1) / (tf + self.k1 * length_norm)
                scores[position] = scores.get(position, 0.0) + term

        # Highest score first; ties broken by chunk ID so results are repeatable.
        ordered = sorted(scores.items(), key=lambda item: (-item[1], self._chunks[item[0]].id))
        hits: list[Hit] = []
        for rank, (position, score) in enumerate(ordered[:k], start=1):
            chunk = self._chunks[position]
            hits.append(Hit(chunk_id=chunk.id, article_id=chunk.article_id, score=score, rank=rank))
        return hits

    def describe(self) -> dict:
        return {"name": self.name, "method": "bm25", "k1": self.k1, "b": self.b}
