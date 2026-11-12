"""The benchmark file: questions with the articles that answer them.

``data/gold/questions.jsonl`` holds one JSON object per line (see docs/EVAL.md
§3)::

    {"qid": "q0042", "tier": "A", "split": "test", "gold": ["AIA:art:6"],
     "source_lang": "pt", "text": {"en": "...", "pt": "...", "fr": "..."},
     "verified_by": "miguel", "notes": ""}

Every number the project publishes is computed against this file, so it is
checked strictly when loaded. A typo in a gold ID would silently score every
retriever as wrong on that question; it is better to refuse the file and say
which line is broken.

Tiers:

* ``A`` - the answer sits in one article (one gold ID).
* ``B`` - the answer needs two or more articles (several gold IDs).
* ``C`` - unanswerable from the corpus (no gold IDs). These have no retrieval
  score; they exist to test whether the answering stage abstains.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from eurlex_rag.ids import is_valid_id, parse_id

DEFAULT_GOLD_PATH = Path("data/gold/questions.jsonl")

TIERS = ("A", "B", "C")
SPLITS = ("dev", "test")
_FIELDS = ("qid", "tier", "split", "gold", "source_lang", "text", "verified_by", "notes")


@dataclass(frozen=True)
class Question:
    """One benchmark question, in every language it was written in.

    Attributes:
        qid: Stable identifier, e.g. ``q0042``.
        tier: ``A`` (one article), ``B`` (several) or ``C`` (unanswerable).
        split: ``dev`` (for choosing settings) or ``test`` (for quoting).
        gold: Article-level IDs that answer the question. Empty for tier C.
        source_lang: Language of the article text the question was drafted from.
        text: The question per language code; always has ``en``.
        verified_by: Who checked the item by hand; empty if nobody yet.
        notes: Free text (edits, reasons for a rewrite).
    """

    qid: str
    tier: str
    split: str
    gold: tuple[str, ...]
    source_lang: str
    text: dict[str, str]
    verified_by: str = ""
    notes: str = ""

    @property
    def answerable(self) -> bool:
        """Tier A and B questions have gold articles; tier C ones do not."""
        return bool(self.gold)


def _check_record(record: object) -> Question:
    """Turn one parsed JSON value into a ``Question``, or raise ``ValueError``.

    The message says what is wrong; the caller adds the file and line.
    """
    if not isinstance(record, dict):
        raise ValueError("record must be a JSON object")
    missing = [name for name in ("qid", "tier", "split", "gold", "text") if name not in record]
    if missing:
        raise ValueError(f"missing field(s): {', '.join(missing)}")
    unknown = sorted(set(record) - set(_FIELDS))
    if unknown:
        raise ValueError(f"unknown field(s): {', '.join(unknown)}")

    qid = record["qid"]
    if not isinstance(qid, str) or not qid.strip():
        raise ValueError("qid must be a non-empty string")
    tier = record["tier"]
    if tier not in TIERS:
        raise ValueError(f"{qid}: tier must be one of {'/'.join(TIERS)}, got {tier!r}")
    split = record["split"]
    if split not in SPLITS:
        raise ValueError(f"{qid}: split must be one of {'/'.join(SPLITS)}, got {split!r}")

    gold = record["gold"]
    if not isinstance(gold, list) or not all(isinstance(g, str) for g in gold):
        raise ValueError(f"{qid}: gold must be a list of ID strings")
    for gid in gold:
        if not is_valid_id(gid):
            raise ValueError(f"{qid}: malformed gold ID {gid!r}")
        if parse_id(gid).sub is not None:
            # Scoring is at article level; a paragraph gold ID could never match.
            raise ValueError(
                f"{qid}: gold ID {gid!r} has a paragraph part; use the article-level ID"
            )
    if len(set(gold)) != len(gold):
        raise ValueError(f"{qid}: gold has duplicate IDs")
    if tier == "C" and gold:
        raise ValueError(f"{qid}: tier C (unanswerable) must have empty gold")
    if tier in ("A", "B") and not gold:
        raise ValueError(f"{qid}: tier {tier} needs at least one gold ID")

    text = record["text"]
    if not isinstance(text, dict) or "en" not in text:
        raise ValueError(f"{qid}: text must be an object with at least an 'en' entry")
    for lang, value in text.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{qid}: text[{lang!r}] must be a non-empty string")

    source_lang = record.get("source_lang", "")
    verified_by = record.get("verified_by", "")
    notes = record.get("notes", "")
    for name, value in (
        ("source_lang", source_lang),
        ("verified_by", verified_by),
        ("notes", notes),
    ):
        if not isinstance(value, str):
            raise ValueError(f"{qid}: {name} must be a string")

    return Question(
        qid=qid,
        tier=tier,
        split=split,
        gold=tuple(gold),
        source_lang=source_lang,
        text=dict(text),
        verified_by=verified_by,
        notes=notes,
    )


def load_gold(path: Path | str) -> list[Question]:
    """Read and validate a gold file. Raises ``ValueError`` naming ``file:line``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No gold file at {path}. The benchmark is built in milestone M2 "
            "(see docs/EVAL.md and data/gold/README.md)."
        )
    questions: list[Question] = []
    seen: dict[str, int] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            where = f"{path}:{line_number}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{where}: not valid JSON ({error.msg})") from error
            try:
                question = _check_record(record)
            except ValueError as error:
                raise ValueError(f"{where}: {error}") from None
            if question.qid in seen:
                raise ValueError(
                    f"{where}: duplicate qid {question.qid!r} (first on line {seen[question.qid]})"
                )
            seen[question.qid] = line_number
            questions.append(question)
    return questions


def write_gold(questions: Iterable[Question], path: Path | str) -> int:
    """Write questions as JSON Lines in the order given. Returns the count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for question in questions:
            record = asdict(question)
            record["gold"] = list(question.gold)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def validate_against_corpus(
    questions: Iterable[Question], known_article_ids: Iterable[str]
) -> list[str]:
    """Problems for gold IDs that are not in the corpus. Empty list means fine.

    A well-formed ID can still point at nothing (``GDPR:art:999``). Such a
    question can never be answered correctly, so the evaluation refuses to run.
    """
    known = set(known_article_ids)
    problems: list[str] = []
    for question in questions:
        for gid in question.gold:
            if gid not in known:
                problems.append(f"{question.qid}: gold ID {gid} is not in the corpus")
    return problems


def by_split(questions: Iterable[Question], split: str) -> list[Question]:
    """Questions of one split (``dev`` or ``test``)."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}. Choose from {SPLITS}.")
    return [q for q in questions if q.split == split]


def by_tier(questions: Iterable[Question], tiers: Sequence[str]) -> list[Question]:
    """Questions whose tier is in ``tiers``."""
    return [q for q in questions if q.tier in tiers]


def query_pairs(questions: Iterable[Question], langs: Sequence[str]) -> list[tuple[Question, str]]:
    """Every (question, language) pair to run, skipping languages a question lacks.

    One question in three languages becomes three queries with the same gold.
    """
    return [(q, lang) for q in questions for lang in langs if lang in q.text]
