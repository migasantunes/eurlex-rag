"""Turning parsed units into the chunks that get indexed.

Two strategies, because chunk size is an experiment (milestone M3), not a given:

``article``    one chunk per article: the whole article in one passage. Simple,
               and the retrieved passage is exactly the thing we cite. Long
               articles dilute the match.
``paragraph``  one chunk per numbered paragraph where an article has them, one
               per article where it does not. Sharper matches, but each chunk
               needs its article's title to make sense on its own.

Either way the chunk text starts with the chapter heading and the article
title. Without them, paragraph 3 of an article about "high-risk AI systems"
might never contain those words.

Only English units come in here. Recitals and annexes are separate chunks and
can be switched off.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence

from eurlex_rag.corpus import INDEX_LANG
from eurlex_rag.ids import ANNEX, ARTICLE, RECITAL, parse_id
from eurlex_rag.types import Chunk, Unit

STRATEGIES = ("article", "paragraph")


def _context(unit: Unit, parent: Unit | None) -> str:
    """Heading and title lines to put in front of a chunk's text."""
    source = parent or unit
    lines = [line for line in (source.heading, source.title) if line]
    return "\n".join(lines)


def _with_context(context: str, text: str) -> str:
    return f"{context}\n{text}" if context else text


def build_chunks(
    units: Iterable[Unit],
    strategy: str = "article",
    include_recitals: bool = True,
    include_annexes: bool = True,
) -> list[Chunk]:
    """Build the chunks to index from English units.

    The input may hold whole articles, their paragraphs, or both. Whichever
    level the strategy asks for is used when present, and the other level is the
    fallback, so no article is ever dropped or indexed twice.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown chunking strategy {strategy!r}. Choose from {STRATEGIES}.")

    units = list(units)
    wrong_lang = {u.lang for u in units} - {INDEX_LANG}
    if wrong_lang:
        raise ValueError(
            f"Only {INDEX_LANG!r} units are indexed; got {sorted(wrong_lang)}. "
            "Non-English text belongs in the parallel table."
        )

    wholes: dict[str, Unit] = {}
    parts: dict[str, list[Unit]] = defaultdict(list)
    for unit in units:
        if parse_id(unit.id).sub is None:
            wholes[unit.id] = unit
        else:
            parts[unit.article_id].append(unit)

    chunks: list[Chunk] = []
    for article_id in sorted(set(wholes) | set(parts), key=_sort_key):
        kind = parse_id(article_id).kind
        if kind == RECITAL and not include_recitals:
            continue
        if kind == ANNEX and not include_annexes:
            continue

        whole = wholes.get(article_id)
        subs = sorted(parts.get(article_id, []), key=lambda u: _sort_key(u.id))
        use_parts = bool(subs) and (strategy == "paragraph" or whole is None)

        if use_parts and strategy == "paragraph":
            for sub in subs:
                text = _with_context(_context(sub, whole), sub.text)
                chunks.append(Chunk(id=sub.id, article_id=article_id, text=text))
        elif use_parts:
            # Article strategy, but only paragraphs were parsed: join them.
            joined = "\n".join(sub.text for sub in subs)
            text = _with_context(_context(subs[0], whole), joined)
            chunks.append(Chunk(id=article_id, article_id=article_id, text=text))
        elif whole is not None:
            text = _with_context(_context(whole, None), whole.text)
            chunks.append(Chunk(id=article_id, article_id=article_id, text=text))
    return chunks


_KIND_ORDER = {RECITAL: 0, ARTICLE: 1, ANNEX: 2}
_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


def _roman_to_int(numeral: str) -> int:
    total = 0
    for symbol, following in zip(numeral, numeral[1:] + " ", strict=True):
        value = _ROMAN[symbol]
        total += -value if following in _ROMAN and _ROMAN[following] > value else value
    return total


def _number_key(number: str) -> tuple[int, str]:
    if number[0].isdigit():
        digits = "".join(ch for ch in number if ch.isdigit())
        return int(digits), number[len(digits) :]
    if all(ch in _ROMAN for ch in number):
        return _roman_to_int(number), ""
    return 0, number


def _sort_key(uid: str) -> tuple:
    """Order IDs the way the regulation does: recitals, articles, annexes."""
    parsed = parse_id(uid)
    sub = _number_key(parsed.sub) if parsed.sub else (-1, "")
    return parsed.reg, _KIND_ORDER[parsed.kind], _number_key(parsed.number), sub


def sorted_ids(ids: Sequence[str]) -> list[str]:
    """IDs in document order."""
    return sorted(ids, key=_sort_key)
