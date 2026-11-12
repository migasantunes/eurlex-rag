"""Turning one EUR-Lex HTML document into ``Unit`` records for one language.

A regulation on EUR-Lex is one long HTML page. We cut it into the pieces a
person would cite: articles, the numbered paragraphs inside them, recitals and
annexes. Each piece becomes a ``Unit`` with a language-independent ID such as
``AIA:art:6`` or ``AIA:art:6:2`` (see ``eurlex_rag.ids``).

How the page is laid out
------------------------
EUR-Lex marks up the Official Journal with CSS classes, and recent documents
also wrap every piece in a ``div`` with a structural ``id``:

=====================  =====================================================
``div#rct_47``          recital 47
``div#cpt_III``         chapter III (``cpt_III.sct_1`` is a section inside it)
``div#art_6``           article 6
``div#006.002``         paragraph 2 of article 6
``div#anx_III``         annex III
``p.oj-ti-art``         the "Article 6" line
``p.oj-sti-art``        the article's own title
``p.oj-ti-section-1``   a "CHAPTER III" / "SECTION 1" line
``p.oj-ti-section-2``   the chapter or section name
``p.oj-doc-ti``         document title, and the "ANNEX III" lines
``p.oj-normal``         ordinary text
=====================  =====================================================

Older documents (the 2016 GDPR, for example) use the same class names without
the ``oj-`` prefix and may have no structural ids at all. So there are two ways
of reading a page:

1. **By id** (``_..._by_id`` functions): find ``div#art_6`` and take what is
   inside it. Exact, and used whenever the ids are there.
2. **By walking** (``_walk``): go through the paragraphs in document order and
   keep track of where we are: a "title of article" line opens a new article,
   ordinary text goes into whatever is open. Used for any kind of unit
   (articles, recitals, annexes) the ids do not give us.

Language independence
---------------------
The parser never looks for the word "Article": it is "Artigo", "Artikel",
"Articolo"... It relies on the structure (ids and classes) and on numbers. Two
traps it handles: French calls article 1 "Article premier" (no digit, so the
walker counts instead), and German numbers paragraphs "(1)" rather than "1.".

Honesty note
------------
This module was written against the documented structure of EUR-Lex pages and
tested on synthetic fixtures, without access to the real files. It is verified
against real files in milestone M1. Until then, failures are made loud: a
document with no articles raises ``ParseError`` instead of quietly returning an
empty list, and ``eurlex_rag.ingest.checks`` compares the counts with the
published ones.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup, Tag

from eurlex_rag.ids import ANNEX, ARTICLE, RECITAL, make_id
from eurlex_rag.types import Unit


class ParseError(ValueError):
    """The document could not be cut into units (for example: no articles found)."""


# --- Small text helpers ---------------------------------------------------------

_SPACES = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Collapse every run of whitespace (including non-breaking spaces) to one space.

    EUR-Lex pads numbers with non-breaking spaces ("1.\\xa0\\xa0\\xa0Text") and
    the HTML source is full of line breaks. Neither carries meaning, and both
    would make the same text look different to a search engine or a test.
    """
    return _SPACES.sub(" ", text.replace("\xa0", " ")).strip()


# A paragraph number at the start of a line: "2. Text" or, in German, "(2) Text".
_PARAGRAPH_NUMBER = re.compile(r"^(?:(\d+)\.|\((\d+)\))\s+")
# A recital number at the start of a line: "(47) Text".
_RECITAL_NUMBER = re.compile(r"^\((\d+)\)\s*")
# The number in an "Article 6" line, in any language: the first run of digits,
# with an optional letter suffix ("6a"). "Artigo 6.º" gives "6".
_ARTICLE_NUMBER = re.compile(r"(\d+)([a-z])?(?![a-z])")
_ROMAN_LETTERS = set("IVXLCDM")


def _is_roman(word: str) -> bool:
    return bool(word) and set(word) <= _ROMAN_LETTERS


def _to_roman(number: int) -> str:
    """1 -> "I", 4 -> "IV", 14 -> "XIV". Annex numbers are small."""
    values = [(100, "C"), (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX")]
    values += [(5, "V"), (4, "IV"), (1, "I")]
    out = ""
    for value, symbol in values:
        while number >= value:
            out += symbol
            number -= value
    return out


def _css_kind(tag: Tag) -> str:
    """The tag's EUR-Lex class with any ``oj-`` prefix removed ("oj-ti-art" -> "ti-art").

    This one line is what lets the same code read old and new documents.
    """
    for css_class in tag.get("class") or []:
        return css_class.removeprefix("oj-")
    return ""


def _has_kind(tag: Tag, kind: str) -> bool:
    """True if any of the tag's classes, without ``oj-``, is ``kind``."""
    return any(c.removeprefix("oj-") == kind for c in tag.get("class") or [])


# Classes that are titles, not body text, and so are left out of a unit's text.
_TITLE_KINDS = {"ti-art", "sti-art", "doc-ti", "ti-section-1", "ti-section-2"}
# Footnotes. They are numbered "(1)" like recitals, so they must never be mistaken
# for body text.
_NOTE_KINDS = {"note"}


def _is_in_signature(tag: Tag) -> bool:
    """True inside the closing "Done at Brussels ... For the Council" block."""
    for parent in tag.parents:
        kind = _css_kind(parent) if isinstance(parent, Tag) else ""
        if kind in ("final", "signatory"):
            return True
    return False


def _row_ancestor(tag: Tag, stop: Tag | None = None) -> Tag | None:
    """The table row a tag sits in, if any (searching no higher than ``stop``)."""
    for parent in tag.parents:
        if parent is stop:
            return None
        if parent.name == "tr":
            return parent
    return None


def _blocks(element: Tag) -> list[str]:
    """The text lines of an element, one per paragraph or table row.

    EUR-Lex lays out lists as tables: one row per point, the label "(a)" in the
    first cell and the text in the second. Reading a row as one line keeps
    "(a) the text of point a" together, so a list stays part of the paragraph
    it belongs to. Nested lists sit inside a row and come along with it.
    Title lines and footnotes are skipped.
    """
    lines: list[str] = []
    for tag in element.find_all(["p", "tr"]):
        if _row_ancestor(tag, stop=element) is not None:
            continue  # inside a row: the row's own line already has this text
        if tag.name == "p" and _css_kind(tag) in _TITLE_KINDS | _NOTE_KINDS:
            continue
        if _is_in_signature(tag):
            continue
        text = normalise(tag.get_text(" "))
        if text:
            lines.append(text)
    return lines


def _strip_number(pattern: re.Pattern[str], line: str) -> str:
    return pattern.sub("", line, count=1)


# --- What the two readers produce -----------------------------------------------


@dataclass
class _Article:
    number: str
    title: str = ""
    heading: str = ""
    lines: list[str] = field(default_factory=list)
    # (paragraph number, its lines), in order. Empty if the article is not numbered.
    paragraphs: list[tuple[str, list[str]]] = field(default_factory=list)


@dataclass
class _Simple:
    """A recital or an annex: a number, maybe a title, and text."""

    number: str
    title: str = ""
    lines: list[str] = field(default_factory=list)


# --- Reader 1: by structural id -------------------------------------------------

_ART_ID = re.compile(r"^art_(\d+[a-z]?)$")
_RCT_ID = re.compile(r"^rct_(\d+)$")
_ANX_ID = re.compile(r"^anx_([^.]+)$")
_PARA_ID = re.compile(r"^(\d{3})\.(\d{3})$")
_DIVISION_ID = re.compile(r"^cpt_[^.]+(\.sct_[^.]+)?$")


def _first_text(element: Tag, kind: str) -> str:
    for tag in element.find_all("p"):
        if _css_kind(tag) == kind:
            return normalise(tag.get_text(" "))
    return ""


def _heading_by_id(article_div: Tag) -> str:
    """The chapter (and section) an article sits in, e.g. "CHAPTER III — HIGH-RISK".

    Walk up from the article; every ``div#cpt_*`` or ``div#cpt_*.sct_*`` on the
    way is a level. Its own label and name are the first section lines inside
    it, because they come before anything nested.
    """
    levels: list[str] = []
    for parent in article_div.parents:
        if parent.name == "div" and _DIVISION_ID.match(parent.get("id", "")):
            label = _first_text(parent, "ti-section-1")
            name = _first_text(parent, "ti-section-2")
            levels.append(" — ".join(part for part in (label, name) if part))
    return " / ".join(reversed(levels))


def _articles_by_id(soup: BeautifulSoup) -> list[_Article]:
    articles: list[_Article] = []
    for div in soup.find_all("div", id=_ART_ID):
        article = _Article(
            number=_ART_ID.match(div["id"]).group(1),
            title=_first_text(div, "sti-art"),
            heading=_heading_by_id(div),
            lines=_blocks(div),
        )
        for para_div in div.find_all("div", id=_PARA_ID):
            number = str(int(_PARA_ID.match(para_div["id"]).group(2)))
            lines = _blocks(para_div)
            if lines:
                lines[0] = _strip_number(_PARAGRAPH_NUMBER, lines[0])
            article.paragraphs.append((number, lines))
        articles.append(article)
    return articles


def _recitals_by_id(soup: BeautifulSoup) -> list[_Simple]:
    recitals: list[_Simple] = []
    for div in soup.find_all("div", id=_RCT_ID):
        lines = _blocks(div)
        if lines:
            lines[0] = _strip_number(_RECITAL_NUMBER, lines[0])
        recitals.append(_Simple(number=_RCT_ID.match(div["id"]).group(1), lines=lines))
    return recitals


def _annexes_by_id(soup: BeautifulSoup) -> list[_Simple]:
    divs = soup.find_all("div", id=_ANX_ID)
    annexes: list[_Simple] = []
    for position, div in enumerate(divs, start=1):
        suffix = _ANX_ID.match(div["id"]).group(1)
        if _is_roman(suffix):
            number = suffix
        elif suffix.isdigit():
            number = _to_roman(int(suffix))
        else:
            number = _to_roman(position)
        # The first "doc-ti" line is the label ("ANNEX III"); any after it is the title.
        title_lines = [
            normalise(p.get_text(" ")) for p in div.find_all("p") if _css_kind(p) == "doc-ti"
        ]
        title = " ".join(title_lines[1:])
        annexes.append(_Simple(number=number, title=title, lines=_blocks(div)))
    return annexes


# --- Reader 2: walking the page in order ----------------------------------------


@dataclass
class _Walk:
    articles: list[_Article] = field(default_factory=list)
    recitals: list[_Simple] = field(default_factory=list)
    annexes: list[_Simple] = field(default_factory=list)


def _add_to_article(article: _Article, line: str, in_table: bool) -> None:
    """Add a line of text to an article, starting a new paragraph if it is numbered.

    A line opens paragraph N only if it is plain text (not a list row) and N is
    the next number in sequence. That keeps numbered definitions ("(1) 'personal
    data' means ...", laid out as table rows) from being taken for paragraphs.
    """
    article.lines.append(line)
    match = None if in_table else _PARAGRAPH_NUMBER.match(line)
    if match:
        number = match.group(1) or match.group(2)
        if int(number) == len(article.paragraphs) + 1:
            article.paragraphs.append((number, [_strip_number(_PARAGRAPH_NUMBER, line)]))
            return
    if article.paragraphs:
        article.paragraphs[-1][1].append(line)


def _walk(soup: BeautifulSoup) -> _Walk:
    """Read the page top to bottom, keeping track of the open article or annex.

    Before the first article we are in the preamble, where a line "(N) text" is
    recital N. After the signature block, nothing is collected until an annex
    title line opens an annex.
    """
    found = _Walk()
    heading = ""
    current: _Article | _Simple | None = None
    in_preamble = True

    for tag in soup.find_all(["p", "tr"]):
        if _row_ancestor(tag) is not None:
            continue  # its text is read with its row
        if _is_in_signature(tag):
            current = None  # the enacting terms are over
            continue
        kind = _css_kind(tag) if tag.name == "p" else ""
        text = normalise(tag.get_text(" "))
        if not text or kind in _NOTE_KINDS:
            continue

        if kind == "ti-section-1":
            heading = text
        elif kind == "ti-section-2":
            heading = f"{heading} — {text}" if heading else text
        elif kind == "ti-art":
            in_preamble = False
            match = _ARTICLE_NUMBER.search(text)
            if match:
                number = match.group(1) + (match.group(2) or "")
            else:  # "Article premier": no digit, so count
                number = str(len(found.articles) + 1)
            current = _Article(number=number, heading=heading)
            found.articles.append(current)
        elif kind == "sti-art" and isinstance(current, _Article):
            current.title = f"{current.title} {text}".strip()
        elif kind == "doc-ti":
            if in_preamble:
                continue  # the document's own title
            words = text.split()
            if len(words) <= 2 and _is_roman(words[-1]):
                current = _Simple(number=words[-1])  # "ANNEX III"
                found.annexes.append(current)
            elif isinstance(current, _Simple) and not current.title:
                current.title = text  # the line after "ANNEX III"
            else:
                current = _Simple(number=_to_roman(len(found.annexes) + 1))  # "ANNEX"
                found.annexes.append(current)
        elif in_preamble:
            match = _RECITAL_NUMBER.match(text)
            if match and tag.name == "tr":
                line = _strip_number(_RECITAL_NUMBER, text)
                found.recitals.append(_Simple(number=match.group(1), lines=[line]))
        elif isinstance(current, _Article):
            _add_to_article(current, text, in_table=tag.name == "tr")
        elif isinstance(current, _Simple):
            current.lines.append(text)
    return found


# --- Putting it together --------------------------------------------------------


def _article_units(reg: str, lang: str, article: _Article) -> list[Unit]:
    units = [
        Unit(
            id=make_id(reg, ARTICLE, article.number),
            lang=lang,
            text="\n".join(article.lines),
            title=article.title,
            heading=article.heading,
        )
    ]
    # Paragraphs carry no title or heading: chunking takes those from the article.
    for number, lines in article.paragraphs:
        units.append(
            Unit(id=make_id(reg, ARTICLE, article.number, number), lang=lang, text="\n".join(lines))
        )
    return units


def parse_document(html: str, reg: str, lang: str) -> list[Unit]:
    """Cut one regulation in one language into units.

    Args:
        html: The page as downloaded from EUR-Lex / Cellar.
        reg: The regulation's short name, e.g. ``"AIA"``; the first part of every ID.
        lang: Two-letter code of the page's language, e.g. ``"pt"``.

    Returns:
        Recitals, then articles (each followed by its paragraphs), then annexes.

    Raises:
        ParseError: if no article is found. A regulation always has articles, so
            this means the page is not what we expected (an error page, a
            different layout) and must be looked at, not silently skipped.
    """
    soup = BeautifulSoup(html, "lxml")
    # Footnote markers ("...houseplants.(1)") sit inside the text; the footnotes
    # themselves are not part of the law's wording, so the markers go too.
    for marker in soup.find_all(lambda tag: _has_kind(tag, "note-tag")):
        marker.decompose()
    walked = _walk(soup)

    # For each kind of unit, the ids win when the page has them.
    articles = _articles_by_id(soup) or walked.articles
    recitals = _recitals_by_id(soup) or walked.recitals
    annexes = _annexes_by_id(soup) or walked.annexes

    if not articles:
        raise ParseError(
            f"No articles found in the {reg} document ({lang}). The file may be an error "
            "page or use a layout this parser does not know; open it in a browser to check."
        )

    units: list[Unit] = []
    for recital in recitals:
        units.append(
            Unit(id=make_id(reg, RECITAL, recital.number), lang=lang, text="\n".join(recital.lines))
        )
    for article in articles:
        units.extend(_article_units(reg, lang, article))
    for annex in annexes:
        units.append(
            Unit(
                id=make_id(reg, ANNEX, annex.number),
                lang=lang,
                text="\n".join(annex.lines),
                title=annex.title,
            )
        )
    return units
