"""The alignment and completeness checks, on the synthetic corpus from conftest."""

from __future__ import annotations

from eurlex_rag.ingest.checks import MAX_LISTED, check_units
from eurlex_rag.types import Unit


def test_aligned_corpus_passes(units):
    # The conftest corpus has gaps in its article numbers on purpose (5, 6, 17, 33),
    # so check only alignment, emptiness and duplicates here.
    problems = check_units(units, expected={})
    assert [p for p in problems if "gaps" not in p] == []


def test_missing_unit_in_one_language(units):
    broken = [u for u in units if not (u.id == "AIA:art:6:2" and u.lang == "pt")]
    problems = check_units(broken, expected={})
    assert any("AIA/pt" in p and "missing" in p and "AIA:art:6:2" in p for p in problems)


def test_extra_unit_in_one_language(units):
    extra = [*units, Unit(id="AIA:art:99", lang="fr", text="En trop.")]
    problems = check_units(extra, expected={})
    assert any("AIA/fr" in p and "not present in en" in p for p in problems)


def test_empty_text(units):
    broken = [*units[1:], Unit(id=units[0].id, lang=units[0].lang, text="   ")]
    problems = check_units(broken, expected={})
    assert any("empty text" in p and units[0].id in p for p in problems)


def test_duplicate(units):
    problems = check_units([*units, units[0]], expected={})
    assert any("more than once" in p and units[0].id in p for p in problems)


def test_article_gaps():
    units = [Unit(id=f"TST:art:{n}", lang="en", text="x") for n in (1, 2, 4)]
    assert check_units(units, expected={}) == ["TST: article numbers have gaps: TST:art:3"]


def test_expected_counts():
    units = [Unit(id="TST:art:1", lang="en", text="x"), Unit(id="TST:rec:1", lang="en", text="y")]
    problems = check_units(units, expected={"TST": {"articles": 2, "recitals": 1}})
    assert problems == ["TST: parsed 1 articles in en, expected 2"]


def test_long_lists_are_capped():
    english = [Unit(id=f"TST:art:{n}", lang="en", text="x") for n in range(1, 31)]
    problems = check_units([*english, Unit(id="TST:art:1", lang="pt", text="x")], expected={})
    assert any(f"(+{29 - MAX_LISTED} more)" in p for p in problems)
