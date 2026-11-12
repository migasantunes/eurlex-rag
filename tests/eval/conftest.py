"""Fixtures for the harness tests: the synthetic corpus and gold set on disk."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eurlex_rag.store import write_units
from eurlex_rag.types import Unit


def write_jsonl(path: Path, records: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
    )
    return path


@pytest.fixture
def gold_path(tmp_path: Path, gold_records: list[dict]) -> Path:
    return write_jsonl(tmp_path / "questions.jsonl", gold_records)


@pytest.fixture
def units_path(tmp_path: Path, units: list[Unit]) -> Path:
    path = tmp_path / "units.jsonl"
    write_units(units, path)
    return path
