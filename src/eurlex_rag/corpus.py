"""The regulations in the corpus and the languages we keep.

English is the only language that is ever indexed or searched. The other
languages are stored as a lookup table keyed by unit ID, for two uses only:
showing a cited article in the user's language, and building the benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass

INDEX_LANG = "en"
PARALLEL_LANGS = ("pt", "fr", "es", "de", "it")
ALL_LANGS = (INDEX_LANG, *PARALLEL_LANGS)

# Languages whose benchmark questions are checked by a human.
VERIFIED_LANGS = ("en", "pt", "fr")


@dataclass(frozen=True)
class Regulation:
    """One regulation in the corpus."""

    short: str  # used as the first part of every unit ID
    celex: str  # the EUR-Lex document number
    name: str


REGULATIONS: tuple[Regulation, ...] = (
    Regulation("GDPR", "32016R0679", "General Data Protection Regulation"),
    Regulation("DMA", "32022R1925", "Digital Markets Act"),
    Regulation("DSA", "32022R2065", "Digital Services Act"),
    Regulation("DATA", "32023R2854", "Data Act"),
    Regulation("AIA", "32024R1689", "Artificial Intelligence Act"),
)

_BY_SHORT = {r.short: r for r in REGULATIONS}
_BY_CELEX = {r.celex: r for r in REGULATIONS}


def by_short(short: str) -> Regulation:
    try:
        return _BY_SHORT[short]
    except KeyError:
        raise ValueError(f"Unknown regulation {short!r}. Known: {', '.join(_BY_SHORT)}") from None


def by_celex(celex: str) -> Regulation:
    try:
        return _BY_CELEX[celex]
    except KeyError:
        raise ValueError(f"Unknown CELEX number {celex!r}") from None


def eurlex_url(celex: str, lang: str) -> str:
    """Link to the official text of a regulation in one language."""
    return f"https://eur-lex.europa.eu/legal-content/{lang.upper()}/TXT/?uri=CELEX:{celex}"
