"""Rung names mapped to the code that builds each retriever.

Each factory imports its engine inside the function, so listing rungs or
importing ``eurlex_rag`` never pulls in an embedding model or a database driver.

Adding a rung means: implement ``Retriever``, add a factory here, add its spec
under ``docs/rungs/``. The harness, the CLI and the service pick it up by name.
"""

from __future__ import annotations

from collections.abc import Callable

from eurlex_rag.retrieve.base import Retriever

BASELINE_RUNG = "01_bm25"


def _bm25() -> Retriever:
    from eurlex_rag.retrieve.bm25 import BM25Retriever

    return BM25Retriever(name="01_bm25")


_FACTORIES: dict[str, Callable[[], Retriever]] = {
    "01_bm25": _bm25,
}

# Rungs that are specified but not built yet, with the milestone that adds them.
PLANNED_RUNGS: dict[str, str] = {
    "00_no_retrieval": "M5",
    "01_bm25_translated": "M4",
    "02_dense": "M3",
    "03_hybrid": "M3",
    "04_rerank": "M4",
    "05_query_transform": "M4",
    "06_agentic": "M6",
}


def available_rungs() -> list[str]:
    """Names of all rungs that can be run today, sorted."""
    return sorted(_FACTORIES)


def build_retriever(rung: str) -> Retriever:
    """Construct the retriever for ``rung`` (not yet indexed)."""
    if rung in _FACTORIES:
        return _FACTORIES[rung]()
    if rung in PLANNED_RUNGS:
        raise ValueError(f"Rung {rung!r} is planned for {PLANNED_RUNGS[rung]} and not built yet.")
    raise ValueError(f"Unknown rung {rung!r}. Available rungs: {', '.join(available_rungs())}")
