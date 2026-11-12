"""Downloading each regulation in each language, once, and keeping the files.

Where the files come from
-------------------------
The Publications Office runs "Cellar", the repository behind EUR-Lex. It
answers a plain URL per document, ``/resource/celex/{CELEX}``, and picks the
file to send from two request headers (this is called *content negotiation*):

- ``Accept: application/xhtml+xml`` - we want the XHTML version;
- ``Accept-Language: por`` - in Portuguese (Cellar uses three-letter codes).

It answers with redirects to the actual file, so redirects are followed. That
is the supported, programmatic route. If it fails, we try the HTML page of the
EUR-Lex website for the same document and language. If both fail, the error
message says which URL to open in a browser and where to save the file: the
corpus is only 30 files, so a manual download is an acceptable last resort.

Caching
-------
Each file is saved as ``<raw_dir>/{CELEX}.{lang}.html`` and never downloaded
again while it exists. That makes re-runs free and polite, and lets you drop in
a file you saved by hand. ``manifest.json`` in the same folder records, per
file, where it came from, its SHA-256 hash, its size and when it was retrieved:
the "version pinning" of SPEC.md §2. If the hash of a file changes between two
ingests, the source changed and the results are no longer comparable.

Being polite
------------
A User-Agent that says who we are, a pause between requests, and a timeout so
a stuck server does not hang the run.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import httpx

from eurlex_rag.corpus import Regulation

USER_AGENT = "eurlex-rag/0.1 (student research project; fetches five EU regulations once)"
TIMEOUT_SECONDS = 60.0
DELAY_SECONDS = 1.0
MANIFEST_NAME = "manifest.json"

# Cellar wants ISO 639-2 three-letter language codes.
LANG3 = {"en": "eng", "pt": "por", "fr": "fra", "es": "spa", "de": "deu", "it": "ita"}

# What the manifest says for a file we found on disk instead of downloading.
LOCAL_FILE = "local file (placed by hand, not downloaded)"


class FetchError(RuntimeError):
    """A document could not be downloaded by any route."""


def cellar_url(celex: str) -> str:
    """Cellar's address for a document. The language goes in a header, not the URL."""
    return f"http://publications.europa.eu/resource/celex/{celex}"


def eurlex_html_url(celex: str, lang: str) -> str:
    """The fallback: the EUR-Lex website's HTML page for one language."""
    return f"https://eur-lex.europa.eu/legal-content/{lang.upper()}/TXT/HTML/?uri=CELEX:{celex}"


def raw_path(raw_dir: Path, celex: str, lang: str) -> Path:
    """Where the file for one regulation in one language is kept."""
    return Path(raw_dir) / f"{celex}.{lang}.html"


# --- The manifest ---------------------------------------------------------------


def load_manifest(raw_dir: Path) -> dict[str, dict]:
    """The manifest as ``{file name: record}``; empty if there is none yet."""
    path = Path(raw_dir) / MANIFEST_NAME
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(raw_dir: Path, manifest: dict[str, dict]) -> None:
    path = Path(raw_dir) / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _record(path: Path, url: str) -> dict:
    data = path.read_bytes()
    return {
        "url": url,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "retrieved": datetime.now(UTC).isoformat(timespec="seconds"),
    }


# --- Downloading ----------------------------------------------------------------


def _looks_like_html(response: httpx.Response) -> bool:
    """Cellar can answer 200 with something else (a PDF, a list of choices)."""
    content_type = response.headers.get("content-type", "")
    return "html" in content_type or "xml" in content_type


def _try(client: httpx.Client, url: str, headers: dict[str, str]) -> tuple[bytes | None, str]:
    """One download attempt. Returns ``(content, "")`` or ``(None, why it failed)``."""
    try:
        response = client.get(
            url,
            headers={"User-Agent": USER_AGENT, **headers},
            follow_redirects=True,
            timeout=TIMEOUT_SECONDS,
        )
    except httpx.HTTPError as error:
        return None, f"{url}: {type(error).__name__}: {error}"
    if response.status_code != 200:
        return None, f"{url}: HTTP {response.status_code}"
    if not _looks_like_html(response):
        content_type = response.headers.get("content-type", "?")
        return None, f"{url}: not HTML (content-type {content_type})"
    return response.content, ""


def fetch_document(
    celex: str,
    lang: str,
    raw_dir: Path,
    client: httpx.Client | None = None,
    delay: float = DELAY_SECONDS,
) -> Path:
    """Make sure the file for one regulation in one language is on disk.

    Returns its path. Downloads nothing if the file is already there. Updates
    the manifest either way, so every file in ``raw_dir`` has a hash recorded.

    Args:
        client: An ``httpx.Client`` to use. Tests pass one with a fake
            transport; normally leave it out and one is created.
        delay: Seconds to wait before each request, to go easy on the server.

    Raises:
        FetchError: if both routes fail. The message says what to do by hand.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_path(raw_dir, celex, lang)
    manifest = load_manifest(raw_dir)

    if path.exists():
        if path.name not in manifest:
            manifest[path.name] = _record(path, LOCAL_FILE)
            save_manifest(raw_dir, manifest)
        return path

    if lang not in LANG3:
        raise FetchError(f"No Cellar language code for {lang!r}. Known: {', '.join(LANG3)}")

    attempts = [
        (cellar_url(celex), {"Accept": "application/xhtml+xml", "Accept-Language": LANG3[lang]}),
        (eurlex_html_url(celex, lang), {"Accept": "text/html"}),
    ]
    own_client = client is None
    client = client or httpx.Client()
    failures: list[str] = []
    try:
        for url, headers in attempts:
            if delay:
                time.sleep(delay)
            content, failure = _try(client, url, headers)
            if content is not None:
                path.write_bytes(content)
                manifest[path.name] = _record(path, url)
                save_manifest(raw_dir, manifest)
                return path
            failures.append(failure)
    finally:
        if own_client:
            client.close()

    tried = "\n  ".join(failures)
    raise FetchError(
        f"Could not download {celex} ({lang}). Tried:\n  {tried}\n"
        f"Open {eurlex_html_url(celex, lang)} in a browser, save the page as {path}, "
        "and run the ingest again: files already on disk are not downloaded."
    )


def fetch_all(
    regs: Sequence[Regulation],
    langs: Sequence[str],
    raw_dir: Path,
    client: httpx.Client | None = None,
    delay: float = DELAY_SECONDS,
) -> tuple[list[Path], list[str]]:
    """Fetch every (regulation, language). Returns the paths and the failure messages.

    One failed file does not stop the others: you get the full list of what is
    missing in one run.
    """
    paths: list[Path] = []
    problems: list[str] = []
    for reg in regs:
        for lang in langs:
            try:
                paths.append(fetch_document(reg.celex, lang, raw_dir, client=client, delay=delay))
            except FetchError as error:
                problems.append(str(error))
    return paths, problems
