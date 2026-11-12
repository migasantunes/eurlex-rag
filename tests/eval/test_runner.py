"""Runner, result files and the report, end to end with BM25 on the tiny corpus."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eurlex_rag.eval.gold import load_gold
from eurlex_rag.eval.report import build_table, write_report
from eurlex_rag.eval.results import load_results, sha256_file, write_result
from eurlex_rag.eval.runner import METRICS, run_retrieval_eval
from eurlex_rag.retrieve.bm25 import BM25Retriever
from eurlex_rag.types import Chunk


@pytest.fixture
def evaluation(chunks: list[Chunk], gold_path: Path) -> dict:
    retriever = BM25Retriever()
    retriever.index(chunks)
    return run_retrieval_eval(retriever, load_gold(gold_path), ["en", "pt", "fr"], k=10)


def test_english_scores_well_portuguese_poorly(evaluation: dict) -> None:
    """The baseline finding: keyword search does not cross languages."""
    en = evaluation["by_lang"]["en"]
    pt = evaluation["by_lang"]["pt"]
    assert en["recall@5"]["mean"] == 1.0
    assert en["mrr@10"]["mean"] == 1.0
    assert pt["recall@5"]["mean"] < 0.5
    assert pt["mrr@10"]["mean"] < en["mrr@10"]["mean"]


def test_unanswerable_are_excluded_and_counted(evaluation: dict) -> None:
    assert evaluation["excluded_unanswerable"] == 1
    assert evaluation["n_questions"] == 4
    assert "q0004" not in {row["qid"] for row in evaluation["per_query"]}
    assert evaluation["n_queries"] == 3 * 3
    assert evaluation["pooled"]["n"] == 3


def test_per_query_rows(evaluation: dict) -> None:
    row = next(r for r in evaluation["per_query"] if r["qid"] == "q0003" and r["lang"] == "en")
    assert row["tier"] == "B"
    assert row["gold"] == ["AIA:art:6", "AIA:anx:III"]
    assert set(row["ranking"]) >= {"AIA:art:6", "AIA:anx:III"}
    assert set(METRICS) <= set(row)
    assert row["latency_ms"] >= 0


def test_summaries_have_intervals(evaluation: dict) -> None:
    for lang in ("en", "pt", "fr"):
        summary = evaluation["by_lang"][lang]
        assert summary["n"] == 3
        for metric in METRICS:
            cell = summary[metric]
            assert cell["low"] <= cell["mean"] <= cell["high"]
    pooled = evaluation["pooled"]
    # Pooled = mean over questions of each question's mean across languages.
    per_question = evaluation["per_question"]
    expected = sum(q["recall@5"] for q in per_question) / len(per_question)
    assert pooled["recall@5"]["mean"] == pytest.approx(expected)
    assert evaluation["latency_ms"]["p50"] <= evaluation["latency_ms"]["p95"]


def test_language_missing_from_every_question(chunks: list[Chunk], gold_path: Path) -> None:
    retriever = BM25Retriever()
    retriever.index(chunks)
    evaluation = run_retrieval_eval(retriever, load_gold(gold_path), ["en", "es"])
    assert set(evaluation["by_lang"]) == {"en"}


def test_only_unanswerable(chunks: list[Chunk], gold_path: Path) -> None:
    retriever = BM25Retriever()
    retriever.index(chunks)
    questions = [q for q in load_gold(gold_path) if q.tier == "C"]
    evaluation = run_retrieval_eval(retriever, questions, ["en"])
    assert evaluation["per_query"] == [] and evaluation["excluded_unanswerable"] == 1
    assert evaluation["pooled"] == {"n": 0}
    assert evaluation["latency_ms"] == {"p50": None, "p95": None}


def test_result_file(tmp_path: Path, evaluation: dict, gold_path: Path) -> None:
    settings = {"chunking": "article", "k": 10, "gold_sha256": sha256_file(gold_path)}
    info = BM25Retriever().describe()
    path = write_result(tmp_path / "results", "01_bm25", "dev", evaluation, info, settings)
    assert path.name == "01_bm25__dev.json"

    record = json.loads(path.read_text(encoding="utf-8"))
    for field in (
        "schema_version",
        "rung",
        "split",
        "created",
        "git_commit",
        "git_dirty",
        "python",
        "platform",
        "retriever",
        "settings",
        "evaluation",
    ):
        assert field in record
    assert record["retriever"]["method"] == "bm25"
    assert record["settings"]["gold_sha256"] == sha256_file(gold_path)
    assert record["created"].endswith("+00:00")

    # A re-run overwrites the same file.
    write_result(tmp_path / "results", "01_bm25", "dev", evaluation, info, settings)
    assert len(load_results(tmp_path / "results")) == 1


def test_sha256(tmp_path: Path) -> None:
    path = tmp_path / "x.txt"
    path.write_bytes(b"abc")
    assert sha256_file(path) == ("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    assert sha256_file(tmp_path / "missing") is None


def test_load_results_missing_dir(tmp_path: Path) -> None:
    assert load_results(tmp_path / "nothing") == []


def test_table(tmp_path: Path, evaluation: dict) -> None:
    out = tmp_path / "results"
    write_result(out, "01_bm25", "test", evaluation, {}, {})
    write_result(out, "01_bm25", "dev", evaluation, {}, {})
    en_only = dict(evaluation, by_lang={"en": evaluation["by_lang"]["en"]})
    write_result(out, "00_other", "dev", en_only, {}, {})

    table = build_table(load_results(out))
    lines = table.strip().splitlines()
    assert lines[0].startswith("| Rung | Split | R@5 EN | R@5 PT | R@5 FR | MRR@10")
    assert [line.split(" | ")[0:2] for line in lines[2:]] == [
        ["| 00_other", "dev"],
        ["| 01_bm25", "dev"],
        ["| 01_bm25", "test"],
    ]
    assert "1.00 [1.00, 1.00]" in lines[3]  # English R@5 with its interval
    assert lines[2].count("–") == 2  # PT and FR not evaluated
    assert lines[3].rstrip().endswith("| 3 |")

    path = write_report(out)
    text = path.read_text(encoding="utf-8")
    assert path.name == "table.md"
    assert "Do not edit by hand" in text
    assert "test numbers are the ones to quote" in text
    assert table in text
