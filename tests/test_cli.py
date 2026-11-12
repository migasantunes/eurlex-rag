"""The ``eurlex`` command line, run in-process against temporary files."""

from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from eurlex_rag.cli import main
from eurlex_rag.store import write_units
from eurlex_rag.types import Unit


@pytest.fixture
def units_path(tmp_path: Path, units: list[Unit]) -> Path:
    path = tmp_path / "units.jsonl"
    write_units(units, path)
    return path


@pytest.fixture
def gold_path(tmp_path: Path, gold_records: list[dict]) -> Path:
    path = tmp_path / "questions.jsonl"
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in gold_records), encoding="utf-8"
    )
    return path


def test_rungs(capsys) -> None:
    assert main(["rungs"]) == 0
    out = capsys.readouterr().out
    assert "01_bm25" in out
    assert "02_dense" in out and "M3" in out


def test_search(capsys, units_path: Path) -> None:
    code = main(["search", "data breach notification", "--units", str(units_path), "--k", "2"])
    assert code == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) <= 2
    assert "GDPR:art:33" in lines[0]
    assert "data breach within 72" in lines[0]
    assert lines[0].endswith("…")  # long text is cut to ~100 characters


def test_search_displays_the_requested_language(capsys, units_path: Path) -> None:
    assert main(["search", "data breach", "--units", str(units_path), "--lang", "pt"]) == 0
    assert "Notificação de uma violação" in capsys.readouterr().out


def test_search_paragraph_chunking(capsys, units_path: Path) -> None:
    args = ["search", "biometric", "--units", str(units_path), "--chunking", "paragraph"]
    assert main(args) == 0
    assert "AIA:" in capsys.readouterr().out


def test_search_unknown_rung(capsys, units_path: Path) -> None:
    assert main(["search", "x", "--units", str(units_path), "--rung", "02_dense"]) == 1
    assert "planned for M3" in capsys.readouterr().err


def test_missing_units_file(capsys, tmp_path: Path) -> None:
    assert main(["search", "x", "--units", str(tmp_path / "none.jsonl")]) == 1
    err = capsys.readouterr().err
    assert "no units file" in err and "eurlex ingest" in err
    assert "Traceback" not in err


def test_eval_dev(capsys, tmp_path: Path, units_path: Path, gold_path: Path) -> None:
    out_dir = tmp_path / "results"
    args = ["eval", "--units", str(units_path), "--gold", str(gold_path), "--out", str(out_dir)]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "01_bm25 on dev" in out
    assert "  en  n=2" in out and "  pt  n=2" in out
    assert "frozen" not in out

    record = json.loads((out_dir / "01_bm25__dev.json").read_text(encoding="utf-8"))
    settings = record["settings"]
    assert settings["chunking"] == "article" and settings["k"] == 10
    assert len(settings["units_sha256"]) == 64 and len(settings["gold_sha256"]) == 64
    assert settings["n_questions_in_split"] == 2
    assert record["evaluation"]["by_lang"]["en"]["recall@5"]["mean"] == 1.0


def test_eval_test_split_reminds_and_counts_excluded(
    capsys, tmp_path: Path, units_path: Path, gold_path: Path
) -> None:
    out_dir = tmp_path / "results"
    args = [
        "eval",
        "--split",
        "test",
        "--units",
        str(units_path),
        "--gold",
        str(gold_path),
        "--out",
        str(out_dir),
        "--langs",
        "en",
        "--no-recitals",
    ]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "frozen configuration" in out
    assert "1 unanswerable question(s) excluded" in out
    record = json.loads((out_dir / "01_bm25__test.json").read_text(encoding="utf-8"))
    assert record["settings"]["include_recitals"] is False


def test_eval_missing_gold(capsys, tmp_path: Path, units_path: Path) -> None:
    args = ["eval", "--units", str(units_path), "--gold", str(tmp_path / "none.jsonl")]
    assert main(args) == 1
    err = capsys.readouterr().err
    assert "No gold file" in err and "M2" in err


def test_eval_invalid_gold(capsys, tmp_path: Path, units_path: Path) -> None:
    gold = tmp_path / "bad.jsonl"
    gold.write_text('{"qid": "q1", "tier": "Z", "split": "dev", "gold": [], "text": {}}\n')
    assert main(["eval", "--units", str(units_path), "--gold", str(gold)]) == 1
    assert "invalid gold file" in capsys.readouterr().err


def test_eval_gold_not_in_corpus(
    capsys, tmp_path: Path, units_path: Path, gold_records: list[dict]
) -> None:
    gold_records[0]["gold"] = ["GDPR:art:999"]
    gold = tmp_path / "q.jsonl"
    gold.write_text("".join(json.dumps(r) + "\n" for r in gold_records), encoding="utf-8")
    out_dir = tmp_path / "results"
    args = ["eval", "--units", str(units_path), "--gold", str(gold), "--out", str(out_dir)]
    assert main(args) == 1
    assert "GDPR:art:999 is not in the corpus" in capsys.readouterr().err
    assert not out_dir.exists()  # nothing written when the gold set is wrong


def test_bad_k(capsys, units_path: Path) -> None:
    assert main(["search", "x", "--units", str(units_path), "--k", "0"]) == 1
    assert "--k must be positive" in capsys.readouterr().err


def test_report(capsys, tmp_path: Path, units_path: Path, gold_path: Path) -> None:
    out_dir = tmp_path / "results"
    base = ["--units", str(units_path), "--gold", str(gold_path), "--out", str(out_dir)]
    assert main(["eval", *base]) == 0
    capsys.readouterr()
    assert main(["report", "--results", str(out_dir)]) == 0
    out = capsys.readouterr().out
    assert "| 01_bm25 | dev |" in out
    assert (out_dir / "table.md").exists()


def test_report_without_results(capsys, tmp_path: Path) -> None:
    assert main(["report", "--results", str(tmp_path / "empty")]) == 1
    assert "eurlex eval" in capsys.readouterr().err


# --- ingest: the pipeline is another module, so it is faked here -------------


@dataclass
class _FakeReport:
    counts: dict
    problems: list = field(default_factory=list)
    out_path: Path = Path("units.jsonl")
    ok: bool = True


def _install_fake_pipeline(monkeypatch, report: _FakeReport, calls: list) -> None:
    module = types.ModuleType("eurlex_rag.ingest.pipeline")

    def run_ingest(raw_dir, out_path, langs, fetch=True, regs=None):
        calls.append({"raw_dir": raw_dir, "langs": langs, "fetch": fetch, "regs": regs})
        return report

    module.run_ingest = run_ingest
    monkeypatch.setitem(sys.modules, "eurlex_rag.ingest.pipeline", module)


def test_ingest_prints_counts(capsys, monkeypatch, tmp_path: Path) -> None:
    calls: list = []
    report = _FakeReport(counts={"GDPR": {"en": 99, "pt": 99}}, out_path=tmp_path / "u.jsonl")
    _install_fake_pipeline(monkeypatch, report, calls)
    assert main(["ingest", "--no-fetch", "--regs", "GDPR", "--langs", "en", "pt"]) == 0
    out = capsys.readouterr().out
    assert "GDPR" in out and "99" in out
    assert calls == [
        {"raw_dir": Path("data/raw"), "langs": ["en", "pt"], "fetch": False, "regs": ["GDPR"]}
    ]


def test_ingest_problems_give_exit_1(capsys, monkeypatch) -> None:
    report = _FakeReport(counts={}, problems=["DSA: pt has 2 fewer articles"], ok=False)
    _install_fake_pipeline(monkeypatch, report, [])
    assert main(["ingest"]) == 1
    assert "DSA: pt has 2 fewer articles" in capsys.readouterr().out


def test_ingest_missing_dependencies(capsys, monkeypatch) -> None:
    # A None entry in sys.modules makes the import raise ImportError.
    monkeypatch.setitem(sys.modules, "eurlex_rag.ingest.pipeline", None)
    assert main(["ingest"]) == 1
    assert 'pip install -e ".[ingest]"' in capsys.readouterr().err
