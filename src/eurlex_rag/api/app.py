"""FastAPI application that serves one retrieval rung over HTTP.

Start it with::

    uvicorn eurlex_rag.api.app:create_app --factory --host 0.0.0.0 --port 8000

There is deliberately no module-level ``app = create_app()``: building the app
loads the corpus and indexes it, and that should happen when the server starts,
not whenever this module is imported (for example by the tests).

Configuration comes from arguments or, when they are not given, from
environment variables, so one container image can serve any rung:

``EURLEX_UNITS``     path to the units file (default ``data/processed/units.jsonl``)
``EURLEX_RUNG``      retrieval rung (default ``registry.BASELINE_RUNG``)
``EURLEX_CHUNKING``  ``article`` or ``paragraph`` (default ``article``)

Endpoints:

``GET  /healthz``    200 when ready; 503 with the reason when not.
``POST /retrieve``   the best articles for a question, with their text in the
                     requested language.
``POST /ask``        501 until answer generation arrives in milestone M5.

Everything the endpoints need lives on ``app.state``; there is no other global.
"""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request

from eurlex_rag.api.schemas import (
    Citation,
    HealthResponse,
    RetrieveRequest,
    RetrieveResponse,
    RetrieveResult,
)
from eurlex_rag.corpus import INDEX_LANG
from eurlex_rag.generate.citations import build_citation
from eurlex_rag.index.chunking import STRATEGIES, build_chunks
from eurlex_rag.registry import BASELINE_RUNG, build_retriever
from eurlex_rag.retrieve.base import Retriever
from eurlex_rag.store import DEFAULT_UNITS_PATH, ParallelTable, load_units
from eurlex_rag.types import Hit

ASK_NOT_READY = "Answer generation is added in milestone M5. Use /retrieve until then."


def search_articles(retriever: Retriever, query: str, k: int, total_chunks: int) -> list[Hit]:
    """The ``k`` best *articles* for ``query``: one hit per article, best first.

    With paragraph chunking, several paragraphs of the same article can be among
    the top chunks, so asking the retriever for ``k`` chunks may give fewer than
    ``k`` different articles. We therefore ask for more chunks than we need and,
    if that is still not enough, ask again for twice as many, until we have
    ``k`` articles or the retriever has nothing more to give.

    Each article keeps its best chunk (the first one seen, since hits arrive
    best first), and ranks are renumbered 1, 2, 3 ... over articles.
    """
    want = min(k * 3, total_chunks)
    while True:
        hits = retriever.search(query, k=want)
        best: dict[str, Hit] = {}
        for hit in sorted(hits, key=lambda h: h.rank):
            best.setdefault(hit.article_id, hit)
        exhausted = len(hits) < want or want >= total_chunks
        if len(best) >= k or exhausted:
            break
        want = min(want * 2, total_chunks)

    return [
        Hit(chunk_id=hit.chunk_id, article_id=hit.article_id, score=hit.score, rank=rank)
        for rank, hit in enumerate(list(best.values())[:k], start=1)
    ]


def create_app(units_path: str | Path | None = None, rung: str | None = None) -> FastAPI:
    """Build the FastAPI app.

    Args:
        units_path: The units file to load. If None, ``EURLEX_UNITS`` or the default.
        rung: The rung to serve. If None, ``EURLEX_RUNG`` or the baseline.

    A missing units file does not stop the app from starting: ``/healthz`` then
    answers 503 and says what is missing, which is easier to diagnose than a
    container that exits. An unknown rung or chunking strategy is a mistake in
    the configuration, so it fails startup with a clear error instead.
    """
    path = Path(units_path or os.environ.get("EURLEX_UNITS") or DEFAULT_UNITS_PATH)
    rung_name = rung or os.environ.get("EURLEX_RUNG") or BASELINE_RUNG
    chunking = os.environ.get("EURLEX_CHUNKING") or "article"

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if chunking not in STRATEGIES:
            raise ValueError(f"EURLEX_CHUNKING must be one of {STRATEGIES}, got {chunking!r}")
        retriever = build_retriever(rung_name)  # raises on an unknown rung

        try:
            units = load_units(path)
        except FileNotFoundError:
            app.state.not_ready = (
                f"No units file at {path}. Build it with `eurlex ingest` and mount it "
                "(or set EURLEX_UNITS)."
            )
            yield
            return

        # Only English is indexed; every language goes into the parallel table.
        english = [unit for unit in units if unit.lang == INDEX_LANG]
        chunks = build_chunks(english, strategy=chunking)
        if not chunks:
            app.state.not_ready = f"The units file at {path} has no English units to index."
            yield
            return

        retriever.index(chunks)
        table = ParallelTable(units)
        app.state.retriever = retriever
        app.state.table = table
        app.state.chunk_count = len(chunks)
        app.state.languages = sorted(table.languages())
        app.state.not_ready = None
        yield

    app = FastAPI(
        title="eurlex-rag",
        description="Cross-lingual retrieval over EU regulation.",
        lifespan=lifespan,
    )
    app.state.rung = rung_name
    # A reason until startup has finished loading; None once ready.
    app.state.not_ready = "The service is still starting."

    def require_ready(request: Request) -> None:
        """Fail with 503, and the reason, if the corpus is not loaded."""
        reason = request.app.state.not_ready
        if reason is not None:
            raise HTTPException(status_code=503, detail=reason)

    @app.get("/healthz", response_model=HealthResponse)
    def healthz(request: Request) -> HealthResponse:
        """200 with what is loaded once ready; 503 with the reason before."""
        require_ready(request)
        state = request.app.state
        return HealthResponse(
            status="ok", rung=state.rung, chunks=state.chunk_count, languages=state.languages
        )

    # A plain ``def`` (not ``async def``): FastAPI runs it on a worker thread, so
    # a slow search does not block the event loop. Searching only reads the
    # index, so concurrent requests are safe.
    @app.post("/retrieve", response_model=RetrieveResponse)
    def retrieve(body: RetrieveRequest, request: Request) -> RetrieveResponse:
        """The best articles for a question, shown in the requested language."""
        require_ready(request)
        state = request.app.state
        display_lang = body.lang or INDEX_LANG

        start = time.perf_counter()
        # The query goes to the English index as typed, whatever its language.
        hits = search_articles(state.retriever, body.query, body.k, state.chunk_count)
        results = []
        for hit in hits:
            citation = build_citation(hit.article_id, display_lang, state.table)
            results.append(
                RetrieveResult(
                    rank=hit.rank,
                    score=hit.score,
                    article_id=hit.article_id,
                    chunk_id=hit.chunk_id,
                    citation=Citation(**citation) if citation else None,
                )
            )
        latency_ms = (time.perf_counter() - start) * 1000.0

        return RetrieveResponse(
            rung=state.rung, query=body.query, results=results, latency_ms=latency_ms
        )

    @app.post("/ask", status_code=501)
    def ask() -> None:
        """Not built yet. Answers with 501 rather than pretending to answer."""
        raise HTTPException(status_code=501, detail=ASK_NOT_READY)

    return app
