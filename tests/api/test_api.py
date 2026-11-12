"""Tests for the HTTP service, on the synthetic corpus written to a temporary file.

Skipped cleanly when the ``serve`` extra (FastAPI) is not installed, so the core
test suite still runs on the standard library alone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # TestClient is built on httpx

from fastapi.testclient import TestClient  # noqa: E402

from eurlex_rag.api.app import ASK_NOT_READY, create_app  # noqa: E402
from eurlex_rag.corpus import eurlex_url  # noqa: E402
from eurlex_rag.index.chunking import build_chunks  # noqa: E402
from eurlex_rag.registry import BASELINE_RUNG  # noqa: E402
from eurlex_rag.retrieve.bm25 import BM25Retriever  # noqa: E402
from eurlex_rag.store import write_units  # noqa: E402
from eurlex_rag.types import Unit  # noqa: E402

GDPR_CELEX = "32016R0679"


@pytest.fixture
def units_file(tmp_path: Path, units: list[Unit]) -> Path:
    path = tmp_path / "units.jsonl"
    write_units(units, path)
    return path


@pytest.fixture
def client(units_file: Path):
    # Using TestClient as a context manager runs the app's startup and shutdown.
    with TestClient(create_app(units_path=units_file)) as c:
        yield c


# --- /healthz ------------------------------------------------------------------


def test_healthz_ok(client: TestClient, en_units: list[Unit]) -> None:
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "rung": BASELINE_RUNG,
        "chunks": len(build_chunks(en_units, strategy="article")),
        "languages": ["en", "fr", "pt"],
    }


def test_healthz_503_when_units_file_missing(tmp_path: Path) -> None:
    missing = tmp_path / "nowhere" / "units.jsonl"
    with TestClient(create_app(units_path=missing)) as c:
        resp = c.get("/healthz")
        assert resp.status_code == 503
        assert str(missing) in resp.json()["detail"]
        assert "eurlex ingest" in resp.json()["detail"]
        # The other endpoints refuse too, instead of crashing.
        assert c.post("/retrieve", json={"query": "data breach"}).status_code == 503


def test_healthz_503_before_startup(units_file: Path) -> None:
    # Without the context manager the startup hook never runs: nothing is loaded.
    c = TestClient(create_app(units_path=units_file))
    assert c.get("/healthz").status_code == 503


def test_config_from_environment(
    monkeypatch: pytest.MonkeyPatch, units_file: Path, en_units: list[Unit]
) -> None:
    monkeypatch.setenv("EURLEX_UNITS", str(units_file))
    monkeypatch.setenv("EURLEX_RUNG", "01_bm25")
    monkeypatch.setenv("EURLEX_CHUNKING", "paragraph")
    with TestClient(create_app()) as c:
        body = c.get("/healthz").json()
    assert body["rung"] == "01_bm25"
    assert body["chunks"] == len(build_chunks(en_units, strategy="paragraph"))


def test_unknown_rung_fails_startup(units_file: Path) -> None:
    with (
        pytest.raises(ValueError, match="Unknown rung"),
        TestClient(create_app(units_path=units_file, rung="99_magic")),
    ):
        pass


def test_unknown_chunking_fails_startup(monkeypatch: pytest.MonkeyPatch, units_file: Path) -> None:
    monkeypatch.setenv("EURLEX_CHUNKING", "sentence")
    with (
        pytest.raises(ValueError, match="EURLEX_CHUNKING"),
        TestClient(create_app(units_path=units_file)),
    ):
        pass


# --- /retrieve -----------------------------------------------------------------


def test_retrieve_english(client: TestClient) -> None:
    resp = client.post("/retrieve", json={"query": "notify the authority of a data breach"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["rung"] == BASELINE_RUNG
    assert body["query"] == "notify the authority of a data breach"
    assert body["latency_ms"] >= 0

    top = body["results"][0]
    assert top["rank"] == 1
    assert top["article_id"] == "GDPR:art:33"
    assert top["chunk_id"] == "GDPR:art:33"
    assert top["citation"] == {
        "id": "GDPR:art:33",
        "title": "Notification of a breach",
        "text": "The controller must notify the authority of a data breach within 72 hours.",
        "text_language": "en",
        "url": eurlex_url(GDPR_CELEX, "en"),
    }
    assert [r["rank"] for r in body["results"]] == list(range(1, len(body["results"]) + 1))


def test_retrieve_shows_portuguese_text(client: TestClient) -> None:
    # The query is English (BM25 needs shared words); lang only changes the display.
    resp = client.post("/retrieve", json={"query": "data breach 72 hours", "lang": "pt"})
    assert resp.status_code == 200
    top = resp.json()["results"][0]
    assert top["article_id"] == "GDPR:art:33"
    assert top["citation"]["text_language"] == "pt"
    assert top["citation"]["title"] == "Notificação de uma violação"
    assert top["citation"]["text"].startswith("O responsável deve notificar")
    assert top["citation"]["url"] == eurlex_url(GDPR_CELEX, "pt")


def test_retrieve_language_code_is_case_insensitive(client: TestClient) -> None:
    resp = client.post("/retrieve", json={"query": "data breach", "lang": "PT"})
    assert resp.status_code == 200
    assert resp.json()["results"][0]["citation"]["text_language"] == "pt"


def test_retrieve_unstored_language_falls_back_to_english(client: TestClient) -> None:
    resp = client.post("/retrieve", json={"query": "data breach 72 hours", "lang": "de"})
    assert resp.status_code == 200
    citation = resp.json()["results"][0]["citation"]
    assert citation["text_language"] == "en"
    assert citation["text"].startswith("The controller must notify")
    assert citation["url"] == eurlex_url(GDPR_CELEX, "en")


def test_retrieve_respects_k(client: TestClient) -> None:
    resp = client.post("/retrieve", json={"query": "systems", "k": 1})
    assert resp.status_code == 200
    assert len(resp.json()["results"]) == 1


def test_retrieve_no_match_returns_empty_list(client: TestClient) -> None:
    resp = client.post("/retrieve", json={"query": "zzzz qqqq"})
    assert resp.status_code == 200
    assert resp.json()["results"] == []


@pytest.mark.parametrize(
    "payload",
    [
        {"query": ""},
        {"query": "   "},
        {"query": "data breach", "k": 0},
        {"query": "data breach", "k": 51},
        {"query": "data breach", "lang": "portuguese"},
        {"k": 5},
    ],
    ids=["empty-query", "blank-query", "k-zero", "k-too-large", "bad-lang", "missing-query"],
)
def test_retrieve_validation_errors(client: TestClient, payload: dict[str, object]) -> None:
    assert client.post("/retrieve", json=payload).status_code == 422


def test_retrieve_deduplicates_paragraphs_to_articles(
    monkeypatch: pytest.MonkeyPatch, units_file: Path, en_units: list[Unit]
) -> None:
    query = "high-risk systems safety component biometric"

    # Check the premise: with paragraph chunks, the raw retriever returns two
    # paragraphs of AIA:art:6 for this query.
    raw = BM25Retriever()
    raw.index(build_chunks(en_units, strategy="paragraph"))
    raw_ids = [hit.chunk_id for hit in raw.search(query, k=10)]
    assert {"AIA:art:6:1", "AIA:art:6:2"} <= set(raw_ids)

    monkeypatch.setenv("EURLEX_CHUNKING", "paragraph")
    with TestClient(create_app(units_path=units_file)) as c:
        resp = c.post("/retrieve", json={"query": query, "k": 10})
    assert resp.status_code == 200
    results = resp.json()["results"]
    article_ids = [r["article_id"] for r in results]

    assert len(article_ids) == len(set(article_ids))  # each article once
    assert article_ids.count("AIA:art:6") == 1
    art6 = next(r for r in results if r["article_id"] == "AIA:art:6")
    assert art6["chunk_id"] in {"AIA:art:6:1", "AIA:art:6:2"}  # its best paragraph
    assert art6["citation"]["id"] == "AIA:art:6"  # the citation is the whole article
    assert [r["rank"] for r in results] == list(range(1, len(results) + 1))


def test_retrieve_paragraph_chunking_still_fills_k(
    monkeypatch: pytest.MonkeyPatch, units_file: Path
) -> None:
    # Several top chunks belong to one article, yet k distinct articles come back.
    monkeypatch.setenv("EURLEX_CHUNKING", "paragraph")
    with TestClient(create_app(units_path=units_file)) as c:
        resp = c.post("/retrieve", json={"query": "high-risk safety component biometric", "k": 3})
    article_ids = [r["article_id"] for r in resp.json()["results"]]
    assert len(article_ids) == 3
    assert len(set(article_ids)) == 3


# --- /ask ----------------------------------------------------------------------


def test_ask_is_not_implemented_yet(client: TestClient) -> None:
    resp = client.post("/ask", json={"query": "anything"})
    assert resp.status_code == 501
    assert resp.json() == {"detail": ASK_NOT_READY}
    assert ASK_NOT_READY == (
        "Answer generation is added in milestone M5. Use /retrieve until then."
    )
