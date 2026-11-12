"""The checks that the parsed corpus is complete and aligned across languages.

The whole project rests on one property: unit ``AIA:art:6`` exists in every
stored language and means the same article in each. The parallel table, the
citation display and the benchmark's ground truth all depend on it. A parser
bug that drops a paragraph in German, or merges two articles in French, would
break that silently. These checks make it loud.

``check_units`` returns a list of human-readable problems. An empty list means
the corpus passed. It never raises: the ingest report shows every problem at
once, so you can fix them in one go.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping

from eurlex_rag.corpus import INDEX_LANG
from eurlex_rag.ids import ARTICLE, RECITAL, parse_id
from eurlex_rag.index.chunking import sorted_ids
from eurlex_rag.types import Unit

# How many articles and recitals each regulation has.
# From the published texts; confirm in M1.
EXPECTED_COUNTS: dict[str, dict[str, int]] = {
    "GDPR": {"articles": 99, "recitals": 173},
    "DMA": {"articles": 54, "recitals": 109},
    "DSA": {"articles": 93, "recitals": 156},
    "DATA": {"articles": 50, "recitals": 119},
    "AIA": {"articles": 113, "recitals": 180},
}

# Long lists of IDs are cut to this many, so one bad parse cannot flood the report.
MAX_LISTED = 10


def _listing(ids: Iterable[str]) -> str:
    """A short, ordered list of IDs for a message: "AIA:art:1, AIA:art:2 (+5 more)"."""
    ordered = sorted_ids(list(ids))
    shown = ", ".join(ordered[:MAX_LISTED])
    hidden = len(ordered) - MAX_LISTED
    return f"{shown} (+{hidden} more)" if hidden > 0 else shown


def _is_whole_article(uid: str) -> bool:
    parsed = parse_id(uid)
    return parsed.kind == ARTICLE and parsed.sub is None


def check_alignment(units: list[Unit]) -> list[str]:
    """Every language has exactly the same IDs as English, per regulation."""
    ids: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for unit in units:
        ids[unit.reg][unit.lang].add(unit.id)

    problems: list[str] = []
    for reg in sorted(ids):
        by_lang = ids[reg]
        if INDEX_LANG not in by_lang:
            problems.append(f"{reg}: no {INDEX_LANG} units, so nothing to align the others with")
            continue
        english = by_lang[INDEX_LANG]
        for lang in sorted(by_lang):
            if lang == INDEX_LANG:
                continue
            missing = english - by_lang[lang]
            extra = by_lang[lang] - english
            if missing:
                problems.append(
                    f"{reg}/{lang}: {len(missing)} IDs present in {INDEX_LANG} are missing: "
                    f"{_listing(missing)}"
                )
            if extra:
                problems.append(
                    f"{reg}/{lang}: {len(extra)} IDs not present in {INDEX_LANG}: {_listing(extra)}"
                )
    return problems


def check_not_empty(units: list[Unit]) -> list[str]:
    """No unit has empty text: an empty unit means the parser missed its content."""
    empty = defaultdict(list)
    for unit in units:
        if not unit.text.strip():
            empty[unit.lang].append(unit.id)
    return [
        f"{lang}: {len(ids)} units with empty text: {_listing(ids)}"
        for lang, ids in sorted(empty.items())
    ]


def check_duplicates(units: list[Unit]) -> list[str]:
    """No (id, language) appears twice: two units would fight over one ID."""
    counts = Counter((unit.id, unit.lang) for unit in units)
    duplicates = defaultdict(list)
    for (uid, lang), count in counts.items():
        if count > 1:
            duplicates[lang].append(uid)
    return [
        f"{lang}: {len(ids)} IDs appear more than once: {_listing(ids)}"
        for lang, ids in sorted(duplicates.items())
    ]


def check_article_numbers(units: list[Unit]) -> list[str]:
    """Article numbers run 1, 2, 3, ... with no gap, per regulation (English).

    Only English is checked; ``check_alignment`` makes the other languages match
    it. Numbers with a letter suffix ("6a") are left out: base regulations do
    not have them, and they do not break the count.
    """
    numbers: dict[str, set[int]] = defaultdict(set)
    for unit in units:
        if unit.lang == INDEX_LANG and _is_whole_article(unit.id):
            number = parse_id(unit.id).number
            if number.isdigit():
                numbers[unit.reg].add(int(number))

    problems: list[str] = []
    for reg, found in sorted(numbers.items()):
        gaps = set(range(1, max(found) + 1)) - found
        if gaps:
            missing = [f"{reg}:{ARTICLE}:{n}" for n in sorted(gaps)]
            problems.append(f"{reg}: article numbers have gaps: {_listing(missing)}")
    return problems


def check_counts(units: list[Unit], expected: Mapping[str, Mapping[str, int]]) -> list[str]:
    """English article and recital counts match the published texts.

    Regulations not in ``expected`` are not checked. Pass ``{}`` to skip this
    check entirely (the tests do, for their tiny fixtures).
    """
    found: dict[str, Counter] = defaultdict(Counter)
    for unit in units:
        if unit.lang != INDEX_LANG:
            continue
        if _is_whole_article(unit.id):
            found[unit.reg]["articles"] += 1
        elif unit.kind == RECITAL:
            found[unit.reg]["recitals"] += 1

    problems: list[str] = []
    for reg in sorted(found):
        if reg not in expected:
            continue
        for what, want in expected[reg].items():
            got = found[reg][what]
            if got != want:
                problems.append(f"{reg}: parsed {got} {what} in {INDEX_LANG}, expected {want}")
    return problems


def check_units(
    units: Iterable[Unit],
    expected: Mapping[str, Mapping[str, int]] = EXPECTED_COUNTS,
) -> list[str]:
    """Run every check. Returns the problems found; an empty list means all is well."""
    units = list(units)
    return (
        check_alignment(units)
        + check_not_empty(units)
        + check_duplicates(units)
        + check_article_numbers(units)
        + check_counts(units, expected)
    )
