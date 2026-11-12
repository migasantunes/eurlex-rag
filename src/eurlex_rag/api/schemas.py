"""Request and response bodies for the HTTP API.

Pydantic checks every request before our code runs: an empty query, a ``k``
out of range or a malformed language code is rejected with a 422 and never
reaches the retriever. The response models document the exact JSON shape in
the automatic OpenAPI page at ``/docs``.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

# Upper bound on ``k``. Fifty articles is far more than an answer can use; the
# limit keeps one request from asking for the whole corpus.
MAX_K = 50
# Questions are short. The limit stops someone posting a whole book as a query.
MAX_QUERY_CHARS = 2000

# Leading and trailing spaces are stripped first, so "   " counts as empty.
Query = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUERY_CHARS)
]
# A two-letter ISO 639-1 code such as "pt". Upper case is accepted and lowered
# (the pattern is checked before lowering, so it allows both cases).
LangCode = Annotated[
    str, StringConstraints(strip_whitespace=True, to_lower=True, pattern=r"^[A-Za-z]{2}$")
]


class HealthResponse(BaseModel):
    """Body of ``GET /healthz`` when the service is ready."""

    status: str = Field(description='Always "ok" when the service answers 200.')
    rung: str = Field(description="The retrieval rung being served, e.g. 01_bm25.")
    chunks: int = Field(description="How many English chunks are indexed.")
    languages: list[str] = Field(description="Languages stored in the parallel table.")


class RetrieveRequest(BaseModel):
    """Body of ``POST /retrieve``."""

    query: Query = Field(description="The question, in any language. Must not be empty.")
    k: int = Field(default=5, ge=1, le=MAX_K, description="How many articles to return.")
    lang: LangCode | None = Field(
        default=None,
        description=(
            "Language to show the article text in (default en). The search itself "
            "always runs on the English index; this only changes what is displayed."
        ),
    )


class Citation(BaseModel):
    """One article as shown to the user. See ``generate.citations.build_citation``."""

    id: str = Field(description="Article-level unit ID, e.g. AIA:art:6.")
    title: str
    text: str = Field(description="The official text, never machine-translated.")
    text_language: str = Field(description="Language of `text`: the requested one, or en.")
    url: str = Field(description="EUR-Lex link to the regulation in `text_language`.")


class RetrieveResult(BaseModel):
    """One retrieved article."""

    rank: int = Field(description="1 is best. Counted over articles, not chunks.")
    score: float = Field(description="Retriever score of the article's best chunk.")
    article_id: str
    chunk_id: str = Field(description="The best-scoring chunk of this article.")
    citation: Citation | None = Field(
        description="Display text. Null only if the article is missing from the table."
    )


class RetrieveResponse(BaseModel):
    """Body returned by ``POST /retrieve``."""

    rung: str
    query: str
    results: list[RetrieveResult]
    latency_ms: float = Field(description="Time spent searching and building citations.")
