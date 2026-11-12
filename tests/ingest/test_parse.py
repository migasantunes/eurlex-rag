"""Parsing EUR-Lex HTML into units, on synthetic fixtures in both layouts."""

from __future__ import annotations

from pathlib import Path

import pytest

from eurlex_rag.ingest.parse import ParseError, normalise, parse_document

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _parse(name: str, lang: str) -> dict:
    html = (FIXTURES / name).read_text(encoding="utf-8")
    return {unit.id: unit for unit in parse_document(html, "TST", lang)}


@pytest.fixture
def new_en() -> dict:
    return _parse("new_style.en.html", "en")


@pytest.fixture
def new_pt() -> dict:
    return _parse("new_style.pt.html", "pt")


@pytest.fixture
def old_en() -> dict:
    return _parse("old_style.en.html", "en")


def test_new_style_ids(new_en):
    assert list(new_en) == [
        "TST:rec:1",
        "TST:rec:2",
        "TST:art:1",
        "TST:art:2",
        "TST:art:2:1",
        "TST:art:2:2",
        "TST:art:3",
        "TST:anx:I",
        "TST:anx:II",
    ]
    assert all(unit.lang == "en" for unit in new_en.values())


def test_titles_and_headings(new_en):
    assert new_en["TST:art:2"].title == "Watering"
    assert new_en["TST:art:2"].heading == "CHAPTER I — GENERAL PROVISIONS"
    # An article inside a section keeps both levels.
    assert new_en["TST:art:3"].heading == "CHAPTER II — LIGHT / SECTION 1 — Windows"
    assert new_en["TST:anx:I"].title == "List of plants that need light"


def test_article_text_has_no_number_or_title_line(new_en):
    text = new_en["TST:art:1"].text
    assert text == "This Regulation lays down rules on the care of houseplants."
    assert "Article" not in text and "Subject matter" not in text


def test_footnotes_and_signature_are_left_out(new_en):
    all_text = " ".join(unit.text for unit in new_en.values())
    assert "footnote" not in all_text
    assert "(1)" not in new_en["TST:art:1"].text  # the footnote marker
    assert "Done at" not in all_text


def test_paragraph_units_keep_their_list(new_en):
    first, second = new_en["TST:art:2:1"], new_en["TST:art:2:2"]
    assert first.text == "Owners shall water each plant once a week."
    assert second.text == "Watering shall be reduced:\n(a) in winter;\n(b) for cacti."
    # Paragraphs get their context from the parent article at chunking time.
    assert first.title == "" and first.heading == ""
    # The article itself holds the whole text.
    assert "(b) for cacti." in new_en["TST:art:2"].text


def test_recitals_lose_their_number(new_en):
    assert new_en["TST:rec:1"].text == "Houseplants improve the well-being of their owners."


def test_whitespace_is_normalised(new_en):
    assert new_en["TST:rec:2"].text == "Watering should be regular but moderate."
    assert normalise("  a\xa0\xa0b \n\t c ") == "a b c"


def test_english_and_portuguese_have_the_same_ids(new_en, new_pt):
    assert list(new_en) == list(new_pt)
    assert new_pt["TST:art:2"].title == "Rega"
    assert new_pt["TST:art:3"].heading == "CAPÍTULO II — LUZ / SECÇÃO 1 — Janelas"


def test_old_style_without_ids(old_en):
    assert list(old_en) == [
        "TST:rec:1",
        "TST:rec:2",
        "TST:art:1",
        "TST:art:2",
        "TST:art:3",
        "TST:art:3:1",
        "TST:art:3:2",
        "TST:art:3:3",
        "TST:anx:I",
    ]


def test_old_style_details(old_en):
    # "Article premier" has no digit: the walker counted it as 1.
    assert old_en["TST:art:1"].title == "Subject matter"
    assert old_en["TST:art:3"].heading == "CHAPTER II — Maintenance"
    assert (
        old_en["TST:art:3:2"].text
        == "The chain shall be cleaned:\n(a) after rain;\n(b) before oiling."
    )
    # Definitions "(1) ..." laid out as list rows are not taken for paragraphs.
    assert "TST:art:2:1" not in old_en
    # A single annex called just "ANNEX" is annex I, and the signature is not in article 3.
    assert old_en["TST:anx:I"].title == "Tools"
    assert "Done at" not in old_en["TST:art:3"].text
    assert old_en["TST:rec:2"].text == "Chains rust when left wet."


def test_german_style_paragraph_numbers():
    html = """
    <p class="ti-art">Artikel 1</p><p class="sti-art">Gegenstand</p>
    <p class="normal">(1) Erster Absatz.</p><p class="normal">(2) Zweiter Absatz.</p>
    """
    units = {u.id: u for u in parse_document(html, "TST", "de")}
    assert units["TST:art:1:2"].text == "Zweiter Absatz."


def test_document_without_articles_raises():
    with pytest.raises(ParseError, match="No articles found"):
        parse_document("<html><body><p>Page not found</p></body></html>", "TST", "en")
