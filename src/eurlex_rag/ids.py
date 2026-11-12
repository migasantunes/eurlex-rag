"""Stable, language-independent IDs for pieces of a regulation.

Every unit of text gets an ID built from the regulation's short name, the kind
of unit and its number::

    AIA:art:6        article 6 of the AI Act
    AIA:art:6:2      paragraph 2 of article 6
    AIA:rec:47       recital 47
    AIA:anx:III      annex III

The numbering of a regulation is the same in every language version. That one
fact carries the whole project: it lets us show a cited article in the user's
language, and it gives the benchmark its ground truth (a question written from
the Portuguese text of ``AIA:art:6`` must retrieve the English ``AIA:art:6``).

Retrieval is always scored at *article level*: a hit on ``AIA:art:6:2`` counts
as a hit on ``AIA:art:6``. ``article_level`` does that roll-up.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

ARTICLE = "art"
RECITAL = "rec"
ANNEX = "anx"
KINDS = (ARTICLE, RECITAL, ANNEX)

# Regulation short name: upper-case letters and digits, starting with a letter.
_REG = r"[A-Z][A-Z0-9]*"
# Article and recital numbers are digits, optionally with a letter suffix ("6a").
# Annexes are numbered with roman numerals; a regulation with a single annex
# uses the bare word, which we store as "I".
_NUMBER = r"[0-9]+[a-z]?|[IVXLC]+"
_SUB = r"[0-9]+[a-z]?|[A-Za-z]"
_ID_RE = re.compile(rf"^({_REG}):({'|'.join(KINDS)}):({_NUMBER})(?::({_SUB}))?$")


@dataclass(frozen=True)
class UnitId:
    """A parsed unit ID."""

    reg: str
    kind: str
    number: str
    sub: str | None = None

    def __str__(self) -> str:
        base = f"{self.reg}:{self.kind}:{self.number}"
        return f"{base}:{self.sub}" if self.sub else base


def make_id(reg: str, kind: str, number: str | int, sub: str | int | None = None) -> str:
    """Build an ID string and check that it is well formed."""
    uid = str(UnitId(reg, kind, str(number), None if sub is None else str(sub)))
    parse_id(uid)
    return uid


def parse_id(uid: str) -> UnitId:
    """Split an ID string into its parts. Raises ``ValueError`` if malformed."""
    match = _ID_RE.match(uid)
    if not match:
        raise ValueError(f"Malformed unit ID: {uid!r}")
    reg, kind, number, sub = match.groups()
    return UnitId(reg, kind, number, sub)


def is_valid_id(uid: str) -> bool:
    return _ID_RE.match(uid) is not None


def article_level(uid: str) -> str:
    """Drop the paragraph part: ``AIA:art:6:2`` becomes ``AIA:art:6``.

    Recital and annex IDs are returned without their sub-part too, so every unit
    rolls up to the thing a person would cite.
    """
    parsed = parse_id(uid)
    return f"{parsed.reg}:{parsed.kind}:{parsed.number}"
