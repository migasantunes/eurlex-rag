"""Tests for eurlex_rag.generate.citations, on the synthetic corpus."""

from __future__ import annotations

import pytest

from eurlex_rag.corpus import eurlex_url
from eurlex_rag.generate.citations import (
    CitationCheck,
    build_citation,
    citation_precision_recall,
    extract_citations,
    validate_citations,
)
from eurlex_rag.store import ParallelTable
from eurlex_rag.types import Unit

AIA_CELEX = "32024R1689"

# --- extract_citations -------------------------------------------------------


def test_extract_single_citation() -> None:
    assert extract_citations("It must be assessed [AIA:art:6] first.") == ["AIA:art:6"]


def test_extract_rolls_paragraphs_up_to_article() -> None:
    assert extract_citations("See [AIA:art:6:2].") == ["AIA:art:6"]


def test_extract_adjacent_and_comma_separated_groups() -> None:
    answer = "Both apply [AIA:art:6][GDPR:art:22], and also [DSA:art:34, DMA:art:5; AIA:anx:III]."
    assert extract_citations(answer) == [
        "AIA:art:6",
        "GDPR:art:22",
        "DSA:art:34",
        "DMA:art:5",
        "AIA:anx:III",
    ]


def test_extract_deduplicates_in_order_of_first_appearance() -> None:
    answer = "[GDPR:art:17] then [AIA:art:6:1], [AIA:art:6:2] and again [GDPR:art:17]."
    assert extract_citations(answer) == ["GDPR:art:17", "AIA:art:6"]


@pytest.mark.parametrize(
    "answer",
    [
        "No citations here.",
        "A footnote [1] and a remark [see above].",
        "A [Markdown link](https://example.org).",
        "Lower case [aia:art:6] is not an ID.",
        "Unknown kind [AIA:chapter:3].",
        "Mixed bracket [AIA:art:6, the recital] is not a citation group.",
        "Empty [] and spaces [  ].",
    ],
)
def test_extract_ignores_malformed_brackets(answer: str) -> None:
    assert extract_citations(answer) == []


def test_extract_tolerates_spaces_inside_brackets() -> None:
    assert extract_citations("Yes [ AIA:art:6 ,GDPR:art:22 ].") == ["AIA:art:6", "GDPR:art:22"]


# --- validate_citations ------------------------------------------------------


def test_validate_keeps_retrieved_citations_untouched() -> None:
    answer = "It is high-risk [AIA:art:6:2] and banned elsewhere [AIA:art:5]."
    check = validate_citations(answer, ["AIA:art:6", "AIA:art:5"])
    assert check == CitationCheck(
        valid=["AIA:art:6", "AIA:art:5"], invalid=[], cleaned_answer=answer
    )


def test_validate_accepts_paragraph_level_retrieved_ids() -> None:
    check = validate_citations("Yes [AIA:art:6].", ["AIA:art:6:1"])
    assert check.valid == ["AIA:art:6"]
    assert check.invalid == []


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        # Mid-sentence: the double space is closed up.
        (
            "It must be assessed [GDPR:art:99] and registered.",
            "It must be assessed and registered.",
        ),
        # Before punctuation: no space left in front of the full stop.
        ("It is high-risk [GDPR:art:99].", "It is high-risk."),
        # Several invalid groups in a row, before punctuation.
        ("It is high-risk [GDPR:art:99] [DSA:art:1][DMA:art:2].", "It is high-risk."),
        # At the start of the answer and of a line.
        ("[GDPR:art:99] It applies.\n[DSA:art:1] Also this.", "It applies.\nAlso this."),
        # At the end of a line.
        ("It applies [GDPR:art:99]\nNext line.", "It applies\nNext line."),
        # No space around it: nothing is added.
        ("risk[GDPR:art:99] here", "risk here"),
    ],
)
def test_validate_removes_invalid_and_tidies(answer: str, expected: str) -> None:
    check = validate_citations(answer, ["AIA:art:6"])
    assert check.cleaned_answer == expected
    assert check.valid == []
    assert check.invalid  # every removal is counted


def test_validate_mixed_bracket_keeps_only_the_valid_ids() -> None:
    answer = "Both apply [GDPR:art:99, AIA:art:6:2; DSA:art:1] here."
    check = validate_citations(answer, ["AIA:art:6"])
    assert check.cleaned_answer == "Both apply [AIA:art:6:2] here."
    assert check.valid == ["AIA:art:6"]
    assert check.invalid == ["GDPR:art:99", "DSA:art:1"]


def test_validate_adjacent_groups_one_invalid() -> None:
    answer = "Both apply [GDPR:art:99][AIA:art:6] here, and [AIA:art:6] [DSA:art:1]."
    check = validate_citations(answer, ["AIA:art:6"])
    assert check.cleaned_answer == "Both apply [AIA:art:6] here, and [AIA:art:6]."
    assert check.valid == ["AIA:art:6"]
    assert check.invalid == ["GDPR:art:99", "DSA:art:1"]


def test_validate_counts_each_invalid_article_once() -> None:
    check = validate_citations("[GDPR:art:99] x [GDPR:art:99:1] y.", [])
    assert check.invalid == ["GDPR:art:99"]
    assert check.cleaned_answer == "x y."


def test_validate_leaves_non_citation_brackets_alone() -> None:
    answer = "See note [1] and [the annex], not [GDPR:art:99]."
    check = validate_citations(answer, [])
    assert check.cleaned_answer == "See note [1] and [the annex], not."


# --- citation_precision_recall -----------------------------------------------


def test_precision_recall_partial_overlap() -> None:
    precision, recall = citation_precision_recall(
        ["AIA:art:6", "GDPR:art:17"], ["AIA:art:6", "AIA:anx:III", "GDPR:art:33"]
    )
    assert precision == pytest.approx(1 / 2)
    assert recall == pytest.approx(1 / 3)


def test_precision_recall_rolls_up_and_deduplicates() -> None:
    # Two paragraphs of the same gold article count as one correct citation.
    assert citation_precision_recall(["AIA:art:6:1", "AIA:art:6:2"], ["AIA:art:6"]) == (1.0, 1.0)


@pytest.mark.parametrize(
    ("cited", "gold", "expected"),
    [
        ([], [], (1.0, 1.0)),  # correctly cited nothing (unanswerable question)
        ([], ["AIA:art:6"], (1.0, 0.0)),  # missed it: recall carries the penalty
        (["AIA:art:6"], [], (0.0, 1.0)),  # cited something where nothing applies
    ],
    ids=["both-empty", "nothing-cited", "nothing-expected"],
)
def test_precision_recall_empty_set_conventions(
    cited: list[str], gold: list[str], expected: tuple[float, float]
) -> None:
    assert citation_precision_recall(cited, gold) == expected


# --- build_citation ------------------------------------------------------------


@pytest.fixture
def table(units: list[Unit]) -> ParallelTable:
    return ParallelTable(units)


def test_build_citation_in_the_users_language(table: ParallelTable) -> None:
    citation = build_citation("AIA:art:5", "pt", table)
    assert citation == {
        "id": "AIA:art:5",
        "title": "Práticas proibidas",
        "text": "São proibidos os sistemas que manipulam pessoas com técnicas subliminares.",
        "text_language": "pt",
        "url": eurlex_url(AIA_CELEX, "pt"),
    }


def test_build_citation_falls_back_to_english(table: ParallelTable) -> None:
    # German is not in the synthetic corpus: the English text is shown, and said so.
    citation = build_citation("GDPR:art:17", "de", table)
    assert citation is not None
    assert citation["text_language"] == "en"
    assert citation["title"] == "Right to erasure"
    assert citation["url"] == eurlex_url("32016R0679", "en")


def test_build_citation_rolls_paragraph_up(table: ParallelTable) -> None:
    citation = build_citation("AIA:art:6:2", "fr", table)
    assert citation is not None
    assert citation["id"] == "AIA:art:6"
    assert citation["title"] == "Classification des systèmes à haut risque"


def test_build_citation_joins_paragraphs_when_article_unit_missing(
    units: list[Unit],
) -> None:
    only_paragraphs = ParallelTable(u for u in units if u.id != "AIA:art:6")
    citation = build_citation("AIA:art:6", "pt", only_paragraphs)
    assert citation is not None
    assert citation["text_language"] == "pt"
    assert citation["text"].startswith("O prestador deve avaliar")
    assert "identificação biométrica" in citation["text"]


def test_build_citation_unknown_article(table: ParallelTable) -> None:
    assert build_citation("GDPR:art:99", "pt", table) is None
