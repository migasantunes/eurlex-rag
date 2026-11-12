"""IDs, the units file, chunking and the registry."""

from __future__ import annotations

import pytest

from eurlex_rag import corpus, registry
from eurlex_rag.ids import article_level, is_valid_id, make_id, parse_id
from eurlex_rag.index.chunking import build_chunks, sorted_ids
from eurlex_rag.retrieve.base import article_ranking
from eurlex_rag.store import ParallelTable, load_index_units, load_units, write_units
from eurlex_rag.types import Hit, Unit

# --- IDs -------------------------------------------------------------------


@pytest.mark.parametrize(
    "uid",
    ["AIA:art:6", "AIA:art:6:2", "GDPR:rec:173", "AIA:anx:III", "DATA:art:6a", "AIA:anx:III:1"],
)
def test_valid_ids_round_trip(uid: str) -> None:
    assert is_valid_id(uid)
    assert str(parse_id(uid)) == uid


@pytest.mark.parametrize("uid", ["aia:art:6", "AIA:article:6", "AIA:art", "AIA:art:", "AIA:art:x9"])
def test_malformed_ids_are_rejected(uid: str) -> None:
    assert not is_valid_id(uid)
    with pytest.raises(ValueError):
        parse_id(uid)


def test_make_id_and_article_level() -> None:
    assert make_id("AIA", "art", 6, 2) == "AIA:art:6:2"
    assert article_level("AIA:art:6:2") == "AIA:art:6"
    assert article_level("AIA:art:6") == "AIA:art:6"
    assert article_level("AIA:anx:III:1") == "AIA:anx:III"


def test_unit_rejects_a_bad_id() -> None:
    with pytest.raises(ValueError):
        Unit(id="not an id", lang="en", text="x")


def test_ids_sort_in_document_order() -> None:
    ids = ["AIA:art:10", "AIA:anx:IX", "AIA:art:2", "AIA:rec:3", "AIA:anx:IV", "AIA:art:2:1"]
    assert sorted_ids(ids) == [
        "AIA:rec:3",
        "AIA:art:2",
        "AIA:art:2:1",
        "AIA:art:10",
        "AIA:anx:IV",
        "AIA:anx:IX",
    ]


# --- corpus ----------------------------------------------------------------


def test_corpus_registry() -> None:
    assert corpus.by_short("AIA").celex == "32024R1689"
    assert corpus.by_celex("32016R0679").short == "GDPR"
    assert len({r.short for r in corpus.REGULATIONS}) == len(corpus.REGULATIONS) == 5
    assert "PT" in corpus.eurlex_url("32016R0679", "pt")
    with pytest.raises(ValueError):
        corpus.by_short("NOPE")


# --- units file and parallel table ------------------------------------------


def test_units_round_trip_through_jsonl(units: list[Unit], tmp_path) -> None:
    path = tmp_path / "nested" / "units.jsonl"
    assert write_units(units, path) == len(units)
    assert sorted(load_units(path), key=lambda u: (u.lang, u.id)) == sorted(
        units, key=lambda u: (u.lang, u.id)
    )
    assert {u.lang for u in load_index_units(path)} == {"en"}
    assert all(u.lang == "pt" for u in load_units(path, lang="pt"))


def test_missing_units_file_says_what_to_do(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="eurlex ingest"):
        load_units(tmp_path / "absent.jsonl")


def test_parallel_table_lookup_and_fallback(units: list[Unit]) -> None:
    table = ParallelTable(units)
    assert table.get("GDPR:art:17", "pt").lang == "pt"
    # German is not stored in the fixture: fall back to English.
    assert table.get("GDPR:art:17", "de").lang == "en"
    assert table.get("GDPR:art:999", "pt") is None
    assert table.languages() == {"en", "pt", "fr"}
    # The alignment the whole project relies on: same IDs in every language.
    assert table.ids("en") == table.ids("pt") == table.ids("fr")


# --- chunking ---------------------------------------------------------------


def test_article_strategy_gives_one_chunk_per_article(en_units: list[Unit]) -> None:
    chunks = build_chunks(en_units, strategy="article")
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
    assert "AIA:art:6" in ids and "AIA:art:6:1" not in ids
    assert all(c.id == c.article_id for c in chunks)


def test_paragraph_strategy_splits_and_keeps_the_title(en_units: list[Unit]) -> None:
    chunks = {c.id: c for c in build_chunks(en_units, strategy="paragraph")}
    assert "AIA:art:6:1" in chunks and "AIA:art:6:2" in chunks
    assert "AIA:art:6" not in chunks  # never indexed twice
    paragraph = chunks["AIA:art:6:2"]
    assert paragraph.article_id == "AIA:art:6"
    assert "Classification of high-risk systems" in paragraph.text
    assert "Chapter III" in paragraph.text
    # An article without paragraphs is still there, as a whole.
    assert "GDPR:art:17" in chunks


def test_article_strategy_joins_paragraphs_when_only_they_exist() -> None:
    parts = [
        Unit(id="DSA:art:9:1", lang="en", text="First."),
        Unit(id="DSA:art:9:2", lang="en", text="Second."),
    ]
    (chunk,) = build_chunks(parts, strategy="article")
    assert chunk.id == chunk.article_id == "DSA:art:9"
    assert chunk.text == "First.\nSecond."


def test_recitals_and_annexes_can_be_left_out(en_units: list[Unit]) -> None:
    kinds = {c.kind for c in build_chunks(en_units, include_recitals=False, include_annexes=False)}
    assert kinds == {"art"}


def test_non_english_units_are_never_chunked(units: list[Unit]) -> None:
    with pytest.raises(ValueError, match="parallel table"):
        build_chunks(units)


def test_unknown_strategy() -> None:
    with pytest.raises(ValueError):
        build_chunks([], strategy="sentence")


# --- retriever helpers and registry ------------------------------------------


def test_article_ranking_dedupes_keeping_best_rank() -> None:
    hits = [
        Hit("AIA:art:6:2", "AIA:art:6", 3.0, 1),
        Hit("GDPR:art:17", "GDPR:art:17", 2.0, 2),
        Hit("AIA:art:6:1", "AIA:art:6", 1.0, 3),
    ]
    assert article_ranking(hits) == ["AIA:art:6", "GDPR:art:17"]


def test_registry() -> None:
    assert registry.BASELINE_RUNG in registry.available_rungs()
    assert registry.build_retriever("01_bm25").name == "01_bm25"
    with pytest.raises(ValueError, match="planned for M3"):
        registry.build_retriever("02_dense")
    with pytest.raises(ValueError, match="Unknown rung"):
        registry.build_retriever("99_magic")
