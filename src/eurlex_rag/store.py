"""Reading and writing parsed units as JSON Lines.

One JSON object per line, one line per unit, all languages in one file. It is a
few megabytes for the whole corpus, diffs cleanly, and needs no database, so
rung 1 and the evaluation harness run with nothing installed. PostgreSQL enters
at milestone M3, when vectors need a home.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

from eurlex_rag.corpus import INDEX_LANG
from eurlex_rag.types import Unit

DEFAULT_UNITS_PATH = Path("data/processed/units.jsonl")


def write_units(units: Iterable[Unit], path: Path | str) -> int:
    """Write units to ``path``, sorted so the file is stable. Returns the count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(units, key=lambda u: (u.lang, u.id))
    with path.open("w", encoding="utf-8") as handle:
        for unit in ordered:
            handle.write(json.dumps(asdict(unit), ensure_ascii=False) + "\n")
    return len(ordered)


def load_units(path: Path | str, lang: str | None = None) -> list[Unit]:
    """Read units from ``path``; keep one language if ``lang`` is given."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No units file at {path}. Run `eurlex ingest` to build it.")
    units: list[Unit] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                unit = Unit(**json.loads(line))
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: bad unit record ({error})") from error
            if lang is None or unit.lang == lang:
                units.append(unit)
    return units


def load_index_units(path: Path | str) -> list[Unit]:
    """The English units: the only ones that are ever indexed."""
    return load_units(path, lang=INDEX_LANG)


class ParallelTable:
    """Look up the official text of a unit in a given language.

    Used to show a cited article in the user's language. Falls back to English
    when the requested language is not stored.
    """

    def __init__(self, units: Iterable[Unit]) -> None:
        self._by_key: dict[tuple[str, str], Unit] = {(u.id, u.lang): u for u in units}

    def get(self, unit_id: str, lang: str) -> Unit | None:
        """The unit in ``lang``, else in English, else ``None``."""
        return self._by_key.get((unit_id, lang)) or self._by_key.get((unit_id, INDEX_LANG))

    def languages(self) -> set[str]:
        return {lang for _, lang in self._by_key}

    def ids(self, lang: str) -> set[str]:
        return {uid for uid, unit_lang in self._by_key if unit_lang == lang}
