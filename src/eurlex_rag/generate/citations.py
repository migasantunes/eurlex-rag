"""Reading, checking, scoring and displaying the citations in an answer.

The language model (added in milestone M5) is told to cite its sources inline,
as unit IDs in square brackets::

    The provider must assess the system [AIA:art:6:2] before ...
    This is covered twice [AIA:art:6][GDPR:art:22] or [AIA:art:6, GDPR:art:22].

This module does four things with those brackets, all with the standard
library only:

``extract_citations``          which articles does the answer cite?
``validate_citations``         were those articles actually retrieved? Remove
                               the ones that were not.
``citation_precision_recall``  how do the cited articles compare with the gold
                               articles of a benchmark question?
``build_citation``             the official text of a cited article, in the
                               user's language when we have it, for display.

Citations are always handled at *article level*, like retrieval scoring: a
citation of ``AIA:art:6:2`` is a citation of ``AIA:art:6``.

What counts as a citation
-------------------------
A bracket counts as a citation group only when *everything* inside it, split on
commas or semicolons, is a well-formed unit ID. Anything else in brackets
(``[1]``, ``[see above]``, the text of a Markdown link, ``[AIA:art:6, recital]``)
is ordinary text: it is neither extracted nor touched by the validator. Being
strict here is safer than guessing: the validator must never delete a piece of
the answer that was not a citation.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from eurlex_rag.corpus import by_short, eurlex_url
from eurlex_rag.ids import article_level, is_valid_id, parse_id
from eurlex_rag.index.chunking import sorted_ids
from eurlex_rag.store import ParallelTable

# One bracketed group, with the spaces or tabs just before and after it.
# The spaces are captured so that removing a citation can also remove the gap
# it leaves ("assessed [X] and" becomes "assessed and", not "assessed  and").
# [^\[\]]+ means "one or more characters that are not a bracket", so nested or
# empty brackets never match.
_GROUP_RE = re.compile(r"([ \t]*)\[([^\[\]]+)\]([ \t]*)")

# IDs inside one bracket are separated by commas or semicolons.
_SEPARATOR_RE = re.compile(r"[,;]")

# Characters that may directly follow a removed citation without a space.
_CLOSING_PUNCTUATION = ".,;:!?)"

# Where a removed citation was. A character from Unicode's private-use area,
# which never appears in real text, so it cannot clash with the answer.
_REMOVED = ""
# One gap: one or more removed citations in a row, with the spaces around them.
_GAP_RE = re.compile(rf"[ \t]*(?:{_REMOVED}[ \t]*)+")


def _close_gap(match: re.Match[str]) -> str:
    """What to leave where one or more citations were removed.

    ``"assessed [X] and"`` -> ``"assessed and"`` (one space stays),
    ``"risk [X] [Y]."``    -> ``"risk."``        (nothing before punctuation),
    ``"[X] The ..."``      -> ``"The ..."``      (nothing at the start of a line),
    ``"risk[X][Y]"``       -> ``"risk"``         (there was no space to keep).
    """
    text = match.string
    next_char = text[match.end() : match.end() + 1]
    at_line_start = match.start() == 0 or text[match.start() - 1] == "\n"
    at_line_end = next_char in ("", "\n")
    if at_line_start or at_line_end or next_char in _CLOSING_PUNCTUATION:
        return ""
    had_space = any(ch in " \t" for ch in match.group(0))
    return " " if had_space else ""


def _parse_group(inner: str) -> list[str] | None:
    """The IDs in one bracket, as written, or ``None`` if it is not a citation.

    ``"AIA:art:6, GDPR:art:22"`` gives ``["AIA:art:6", "GDPR:art:22"]``.
    ``"see above"`` gives ``None``.
    """
    parts = [part.strip() for part in _SEPARATOR_RE.split(inner)]
    if not parts or not all(is_valid_id(part) for part in parts):
        return None
    return parts


def _unique_in_order(items: Iterable[str]) -> list[str]:
    """Drop repeats, keeping the first occurrence of each item."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def extract_citations(answer: str) -> list[str]:
    """The article IDs cited in ``answer``, in order of first appearance.

    Paragraph citations are rolled up to their article, and an article cited
    several times is listed once::

        >>> extract_citations("See [AIA:art:6:2] and [AIA:art:6, GDPR:art:22].")
        ['AIA:art:6', 'GDPR:art:22']

    Bracketed text that is not made only of well-formed IDs is ignored.
    """
    cited: list[str] = []
    for match in _GROUP_RE.finditer(answer):
        ids = _parse_group(match.group(2))
        if ids is not None:
            cited.extend(article_level(uid) for uid in ids)
    return _unique_in_order(cited)


@dataclass(frozen=True)
class CitationCheck:
    """The result of ``validate_citations``.

    Attributes:
        valid: Cited article IDs that were retrieved, in order of first appearance.
        invalid: Cited article IDs that were *not* retrieved, in the same order.
            Their count is what the invalid-citation rate in docs/EVAL.md counts.
        cleaned_answer: The answer with the invalid citations removed.
    """

    valid: list[str]
    invalid: list[str]
    cleaned_answer: str


def validate_citations(answer: str, retrieved_article_ids: Iterable[str]) -> CitationCheck:
    """Keep only the citations of articles that were actually retrieved.

    Why this is code and not an instruction in the prompt: the prompt already
    tells the model to cite only the passages it was given, and a model can
    still write "[GDPR:art:99]" - an article it never saw, or one that does
    not exist. It writes plausible-looking IDs as easily as plausible-looking
    sentences. We cannot make the model reliable by asking, but we know exactly
    which articles were retrieved, so checking every citation against that list
    is a few lines of code that are right every time. What the check removes is
    counted, so an unreliable model shows up in the numbers instead of in front
    of a user.

    How the text is tidied: an invalid ID is removed from its bracket; a
    bracket left empty is removed with the space around it, so
    ``"assessed [GDPR:art:99] and"`` becomes ``"assessed and"`` and
    ``"risk [GDPR:art:99]."`` becomes ``"risk."`` (see ``_close_gap``). Valid IDs
    are kept as written, including paragraph IDs such as ``AIA:art:6:2``; a
    bracket that keeps some IDs is rewritten with ``", "`` between them.

    Args:
        answer: The answer text written by the model.
        retrieved_article_ids: IDs of what was retrieved. Paragraph IDs are fine;
            they are rolled up to article level before comparing.
    """
    retrieved = {article_level(uid) for uid in retrieved_article_ids}
    valid: list[str] = []
    invalid: list[str] = []

    def replace(match: re.Match[str]) -> str:
        before, inner, after = match.groups()
        ids = _parse_group(inner)
        if ids is None:
            return match.group(0)  # not a citation: leave it exactly as it was

        kept = []
        for uid in ids:
            article = article_level(uid)
            if article in retrieved:
                valid.append(article)
                kept.append(uid)
            else:
                invalid.append(article)

        if kept:
            return f"{before}[{', '.join(kept)}]{after}"
        # Every ID in this bracket was invalid. Leave a marker for now; the gap
        # is tidied in a second pass, once all neighbouring removals are known.
        return f"{before}{_REMOVED}{after}"

    marked = _GROUP_RE.sub(replace, answer)
    cleaned = _GAP_RE.sub(_close_gap, marked)
    return CitationCheck(
        valid=_unique_in_order(valid),
        invalid=_unique_in_order(invalid),
        cleaned_answer=cleaned,
    )


def citation_precision_recall(cited: Iterable[str], gold: Iterable[str]) -> tuple[float, float]:
    """Precision and recall of the cited articles against the gold articles.

    Both sides are rolled up to article level and treated as sets::

        precision = |cited and gold| / |cited|   "how many citations were right"
        recall    = |cited and gold| / |gold|    "how many right ones were cited"

    When a set is empty the fraction is 0/0, so a convention is needed. The one
    used here is that a fraction over nothing is 1.0 ("nothing went wrong"):

    ============  ============  =========  ======  ================================
    cited         gold          precision  recall  meaning
    ============  ============  =========  ======  ================================
    empty         empty         1.0        1.0     correctly cited nothing
                                                   (e.g. an unanswerable question)
    empty         not empty     1.0        0.0     cited nothing it should have
    not empty     empty         0.0        1.0     every citation was unwarranted
    ============  ============  =========  ======  ================================

    Why precision is 1.0, not 0.0, when nothing is cited: precision answers
    "can the citations that were shown be trusted?", and no untrustworthy
    citation was shown. The failure - missing the right article - is exactly
    what recall measures, and it scores 0.0. Scoring it 0.0 on precision as well
    would count the same mistake twice and blur the two metrics. Answers that
    cite nothing are not rewarded overall: their recall is 0, and abstention is
    measured separately (docs/EVAL.md §1). The cost of this choice is that
    averaged precision on its own flatters a model that rarely cites, so
    precision is always reported next to recall.
    """
    cited_set = {article_level(uid) for uid in cited}
    gold_set = {article_level(uid) for uid in gold}
    correct = len(cited_set & gold_set)
    precision = correct / len(cited_set) if cited_set else 1.0
    recall = correct / len(gold_set) if gold_set else 1.0
    return precision, recall


def _article_text(article_id: str, lang: str, table: ParallelTable) -> tuple[str, str, str] | None:
    """``(title, text, language)`` of an article in exactly ``lang``, or ``None``.

    Uses the whole-article unit when it is stored. When only the article's
    paragraphs were parsed, joins them in document order.
    """
    whole = table.get(article_id, lang)
    if whole is not None and whole.lang == lang:
        return whole.title, whole.text, lang

    paragraph_ids = [uid for uid in table.ids(lang) if article_level(uid) == article_id]
    if not paragraph_ids:
        return None
    paragraphs = [table.get(uid, lang) for uid in sorted_ids(paragraph_ids)]
    text = "\n".join(unit.text for unit in paragraphs if unit is not None)
    title = next((unit.title for unit in paragraphs if unit is not None and unit.title), "")
    return title, text, lang


def build_citation(article_id: str, lang: str, table: ParallelTable) -> dict | None:
    """What the user sees for one cited article.

    Returns ``{"id", "title", "text", "text_language", "url"}``, or ``None`` if
    the article is not in the table at all.

    The text is the *official* text in ``lang`` when that language version is
    stored, and the official English text otherwise. ``text_language`` says
    which one was used, so the interface can tell the user "shown in English".
    Legal text is never machine-translated: a translation could change the
    meaning of a legal provision, and the official versions already exist. The
    link points to EUR-Lex in the language actually shown.

    Args:
        article_id: The cited ID. A paragraph ID is rolled up to its article.
        lang: Two-letter code of the language the user asked in, e.g. ``"pt"``.
        table: The parallel table holding every stored language version.
    """
    article_id = article_level(article_id)
    lang = lang.lower()
    found = _article_text(article_id, lang, table)
    if found is None and lang != "en":
        found = _article_text(article_id, "en", table)
    if found is None:
        return None

    title, text, shown_lang = found
    celex = by_short(parse_id(article_id).reg).celex
    return {
        "id": article_id,
        "title": title,
        "text": text,
        "text_language": shown_lang,
        "url": eurlex_url(celex, shown_lang),
    }
