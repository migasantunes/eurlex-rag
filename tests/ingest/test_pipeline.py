"""``run_ingest`` end to end on temporary folders, using the synthetic fixtures.

The fixtures are filed under the AI Act's CELEX number so the corpus registry
can find them. They are tiny, so the published-count check is switched off
with ``expected={}``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import httpx

from eurlex_rag.ingest.pipeline import run_ingest
from eurlex_rag.store import load_units

FIXTURES = Path(__file__).parent.parent / "fixtures"
CELEX = "32024R1689"  # AIA


def _raw_dir(tmp_path: Path, langs=("en", "pt")) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    for lang in langs:
        shutil.copy(FIXTURES / f"new_style.{lang}.html", raw / f"{CELEX}.{lang}.html")
    return raw


def test_end_to_end_without_fetching(tmp_path):
    out = tmp_path / "processed" / "units.jsonl"
    report = run_ingest(
        _raw_dir(tmp_path), out, langs=("en", "pt"), fetch=False, regs=["AIA"], expected={}
    )

    assert report.problems == []
    assert report.ok
    assert report.out_path == out
    assert report.counts == {"AIA": {"en": 9, "pt": 9}}
    units = load_units(out)
    assert len(units) == 18
    assert {u.id for u in units if u.lang == "en"} == {u.id for u in units if u.lang == "pt"}
    assert any(u.id == "AIA:art:2:2" for u in units)


def test_missing_file_is_reported(tmp_path):
    out = tmp_path / "units.jsonl"
    report = run_ingest(
        _raw_dir(tmp_path, langs=("en",)),
        out,
        langs=("en", "pt"),
        fetch=False,
        regs=["AIA"],
        expected={},
    )
    assert not report.ok
    assert any("AIA/pt: missing" in p for p in report.problems)
    assert report.counts == {"AIA": {"en": 9}}
    assert out.exists()  # what was parsed is still written, for inspection


def test_published_counts_are_checked_by_default(tmp_path):
    report = run_ingest(
        _raw_dir(tmp_path, langs=("en",)),
        tmp_path / "u.jsonl",
        langs=("en",),
        fetch=False,
        regs=["AIA"],
    )
    assert "AIA: parsed 3 articles in en, expected 113" in report.problems


def test_unparseable_file_is_reported(tmp_path):
    raw = _raw_dir(tmp_path, langs=("en",))
    (raw / f"{CELEX}.pt.html").write_text("<html><body>Error 404</body></html>")
    report = run_ingest(
        raw, tmp_path / "u.jsonl", langs=("en", "pt"), fetch=False, regs=["AIA"], expected={}
    )
    assert any("No articles found" in p for p in report.problems)


def test_fetching_uses_files_on_disk_and_downloads_the_rest(tmp_path):
    raw = _raw_dir(tmp_path, langs=("en",))
    portuguese = (FIXTURES / "new_style.pt.html").read_bytes()
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request.headers["accept-language"])
        return httpx.Response(200, content=portuguese, headers={"content-type": "text/html"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    report = run_ingest(
        raw,
        tmp_path / "u.jsonl",
        langs=("en", "pt"),
        regs=["AIA"],
        expected={},
        client=client,
        delay=0,
    )
    assert report.ok, report.problems
    assert requested == ["por"]  # English was already on disk
    assert (raw / "manifest.json").exists()
