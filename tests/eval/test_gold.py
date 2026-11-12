"""The gold file: loading, every validation rule, and the helpers."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from eurlex_rag.eval.gold import (
    by_split,
    by_tier,
    load_gold,
    query_pairs,
    validate_against_corpus,
    write_gold,
)


def write_jsonl(path: Path, records: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def test_valid_file_loads(gold_path: Path) -> None:
    questions = load_gold(gold_path)
    assert [q.qid for q in questions] == ["q0001", "q0002", "q0003", "q0004"]
    q3 = questions[2]
    assert q3.tier == "B" and q3.gold == ("AIA:art:6", "AIA:anx:III")
    assert q3.text["pt"].startswith("Um sistema")
    assert not questions[3].answerable and questions[0].answerable


def test_blank_lines_are_skipped(tmp_path: Path, gold_records: list[dict]) -> None:
    path = write_jsonl(tmp_path / "g.jsonl", gold_records)
    path.write_text(path.read_text() + "\n\n", encoding="utf-8")
    assert len(load_gold(path)) == 4


def _broken(tmp_path: Path, records: list[dict], index: int, **changes) -> Path:
    records = copy.deepcopy(records)
    for key, value in changes.items():
        if value is _DELETE:
            del records[index][key]
        else:
            records[index][key] = value
    return write_jsonl(tmp_path / "broken.jsonl", records)


_DELETE = object()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"tier": "D"}, "tier must be one of A/B/C"),
        ({"split": "train"}, "split must be one of dev/test"),
        ({"gold": ["AIA:article:6"]}, "malformed gold ID"),
        ({"gold": ["AIA:art:6:2"]}, "paragraph part"),
        ({"gold": []}, "tier A needs at least one gold ID"),
        ({"gold": "GDPR:art:33"}, "gold must be a list"),
        ({"gold": ["GDPR:art:33", "GDPR:art:33"]}, "duplicate IDs"),
        ({"text": {"pt": "Olá?"}}, "at least an 'en' entry"),
        ({"text": {"en": "  "}}, "text\\['en'\\] must be a non-empty string"),
        ({"qid": ""}, "qid must be a non-empty string"),
        ({"qid": _DELETE}, "missing field"),
        ({"extra": 1}, "unknown field"),
        ({"notes": 3}, "notes must be a string"),
    ],
)
def test_each_rule_has_a_clear_message(
    tmp_path: Path, gold_records: list[dict], changes: dict, message: str
) -> None:
    path = _broken(tmp_path, gold_records, 1, **changes)
    with pytest.raises(ValueError, match=message) as info:
        load_gold(path)
    assert f"{path}:2:" in str(info.value)  # the second record is on line 2


def test_tier_c_must_have_empty_gold(tmp_path: Path, gold_records: list[dict]) -> None:
    path = _broken(tmp_path, gold_records, 3, gold=["GDPR:art:17"])
    with pytest.raises(ValueError, match=r":4: q0004: tier C"):
        load_gold(path)


def test_duplicate_qid(tmp_path: Path, gold_records: list[dict]) -> None:
    path = _broken(tmp_path, gold_records, 2, qid="q0001")
    with pytest.raises(ValueError, match="duplicate qid 'q0001' \\(first on line 1\\)"):
        load_gold(path)


def test_bad_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"qid": "q1",\n', encoding="utf-8")
    with pytest.raises(ValueError, match=":1: not valid JSON"):
        load_gold(path)
    path.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a JSON object"):
        load_gold(path)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="M2"):
        load_gold(tmp_path / "nope.jsonl")


def test_validate_against_corpus(gold_path: Path) -> None:
    questions = load_gold(gold_path)
    known = {"GDPR:art:33", "GDPR:art:17", "AIA:art:6", "AIA:anx:III"}
    assert validate_against_corpus(questions, known) == []
    problems = validate_against_corpus(questions, known - {"AIA:anx:III"})
    assert problems == ["q0003: gold ID AIA:anx:III is not in the corpus"]


def test_round_trip(tmp_path: Path, gold_path: Path) -> None:
    questions = load_gold(gold_path)
    copy_path = tmp_path / "copy" / "q.jsonl"
    assert write_gold(questions, copy_path) == 4
    assert load_gold(copy_path) == questions


def test_filters_and_pairs(gold_path: Path) -> None:
    questions = load_gold(gold_path)
    assert [q.qid for q in by_split(questions, "dev")] == ["q0001", "q0002"]
    assert [q.qid for q in by_tier(questions, ["B", "C"])] == ["q0003", "q0004"]
    with pytest.raises(ValueError):
        by_split(questions, "train")

    pairs = query_pairs(questions[:1], ["en", "pt", "es"])  # no Spanish text: skipped
    assert [(q.qid, lang) for q, lang in pairs] == [("q0001", "en"), ("q0001", "pt")]
