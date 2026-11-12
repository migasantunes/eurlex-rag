"""The contract every retrieval rung implements.

A *retriever* is one way of finding the chunks relevant to a question: keyword
search, vector search, a fusion of the two, with or without a reranker. The
evaluation harness and the HTTP service only ever talk to this interface. That
is what makes the rungs comparable: same chunks, same questions, same metrics,
different way of searching.

Two operations are enough:

* ``index``  - take the chunks once and prepare whatever the method needs
  (term statistics, vectors).
* ``search`` - return the best ``k`` chunks for one query, best first.

The query arrives as the user typed it, in whatever language. A rung that
translates or rewrites the query does so inside its own ``search``; the harness
does not know or care.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from eurlex_rag.types import Chunk, Hit


class Retriever(ABC):
    """Base class for all retrieval rungs."""

    #: Rung name, e.g. ``"01_bm25"``. Recorded with every result.
    name: str = "unnamed"

    @abstractmethod
    def index(self, chunks: Sequence[Chunk]) -> None:
        """Prepare the retriever to search ``chunks``. Replaces any earlier index."""

    @abstractmethod
    def search(self, query: str, k: int = 10) -> list[Hit]:
        """The ``k`` best chunks for ``query``, best first, ranks starting at 1."""

    def describe(self) -> dict:
        """Settings worth recording next to a result (parameters, model IDs)."""
        return {"name": self.name}


def article_ranking(hits: Sequence[Hit]) -> list[str]:
    """Collapse chunk hits to a ranked list of distinct article IDs.

    Several chunks of the same article can be retrieved (its paragraphs). For
    scoring, an article counts once, at the rank of its best chunk.
    """
    seen: set[str] = set()
    ranking: list[str] = []
    for hit in sorted(hits, key=lambda h: h.rank):
        if hit.article_id not in seen:
            seen.add(hit.article_id)
            ranking.append(hit.article_id)
    return ranking
