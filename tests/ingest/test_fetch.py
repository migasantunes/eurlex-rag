"""Downloading and caching, with a fake HTTP transport: nothing touches the network."""

from __future__ import annotations

import hashlib
import json

import httpx
import pytest

from eurlex_rag.ingest.fetch import (
    LOCAL_FILE,
    USER_AGENT,
    FetchError,
    cellar_url,
    eurlex_html_url,
    fetch_document,
)

CELEX = "32024R1689"
PAGE = b"<html><body><p class='oj-ti-art'>Article 1</p></body></html>"


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_cellar_route_saves_file_and_manifest(tmp_path):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=PAGE, headers={"content-type": "application/xhtml+xml"})

    path = fetch_document(CELEX, "pt", tmp_path, client=_client(handler), delay=0)

    assert path == tmp_path / f"{CELEX}.pt.html"
    assert path.read_bytes() == PAGE
    request = seen[0]
    assert str(request.url) == cellar_url(CELEX)
    assert request.headers["accept-language"] == "por"
    assert request.headers["accept"] == "application/xhtml+xml"
    assert request.headers["user-agent"] == USER_AGENT

    record = json.loads((tmp_path / "manifest.json").read_text())[path.name]
    assert record["url"] == cellar_url(CELEX)
    assert record["sha256"] == hashlib.sha256(PAGE).hexdigest()
    assert record["bytes"] == len(PAGE)
    assert record["retrieved"].endswith("+00:00")


def test_follows_redirects(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "publications.europa.eu" and "celex" in request.url.path:
            return httpx.Response(303, headers={"location": "http://publications.europa.eu/doc"})
        return httpx.Response(200, content=PAGE, headers={"content-type": "text/html"})

    path = fetch_document(CELEX, "en", tmp_path, client=_client(handler), delay=0)
    assert path.read_bytes() == PAGE


def test_existing_file_is_not_downloaded(tmp_path):
    (tmp_path / f"{CELEX}.fr.html").write_bytes(PAGE)

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not download a file that is already there")

    fetch_document(CELEX, "fr", tmp_path, client=_client(handler), delay=0)
    record = json.loads((tmp_path / "manifest.json").read_text())[f"{CELEX}.fr.html"]
    assert record["url"] == LOCAL_FILE
    assert record["sha256"] == hashlib.sha256(PAGE).hexdigest()


def test_falls_back_to_eurlex_page(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "publications.europa.eu":
            return httpx.Response(300, text="multiple choices")
        return httpx.Response(200, content=PAGE, headers={"content-type": "text/html"})

    path = fetch_document(CELEX, "de", tmp_path, client=_client(handler), delay=0)
    record = json.loads((tmp_path / "manifest.json").read_text())[path.name]
    assert record["url"] == eurlex_html_url(CELEX, "de")


def test_non_html_answer_is_a_failure(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"%PDF", headers={"content-type": "application/pdf"})

    with pytest.raises(FetchError, match="not HTML"):
        fetch_document(CELEX, "en", tmp_path, client=_client(handler), delay=0)


def test_error_message_says_what_to_do(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host", request=request)

    with pytest.raises(FetchError) as caught:
        fetch_document(CELEX, "it", tmp_path, client=_client(handler), delay=0)
    message = str(caught.value)
    assert cellar_url(CELEX) in message and "ConnectError" in message
    assert f"Open {eurlex_html_url(CELEX, 'it')} in a browser" in message
    assert str(tmp_path / f"{CELEX}.it.html") in message
    assert not (tmp_path / f"{CELEX}.it.html").exists()
