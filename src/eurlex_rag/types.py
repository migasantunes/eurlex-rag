"""The three records that move through the system.

``Unit``  - a piece of a regulation in one language, as parsed from the source.
``Chunk`` - what actually gets indexed. Built from English units only.
``Hit``   - one search result.
"""

from __future__ import annotations

from dataclasses import dataclass

from eurlex_rag.ids import article_level, parse_id


@dataclass(frozen=True)
class Unit:
    """A piece of a regulation in one language.

    Attributes:
        id: Unit ID, e.g. ``AIA:art:6`` or ``AIA:art:6:2``. See ``eurlex_rag.ids``.
        lang: Two-letter language code of ``text``.
        title: The article's own title ("Classification rules for high-risk AI
            systems"). Empty for recitals and paragraphs.
        heading: The chapter or section the unit sits under. May be empty.
        text: The text of the unit itself, without its title.
    """

    id: str
    lang: str
    text: str
    title: str = ""
    heading: str = ""

    def __post_init__(self) -> None:
        parse_id(self.id)  # fail early on a malformed ID

    @property
    def reg(self) -> str:
        return parse_id(self.id).reg

    @property
    def kind(self) -> str:
        return parse_id(self.id).kind

    @property
    def article_id(self) -> str:
        return article_level(self.id)


@dataclass(frozen=True)
class Chunk:
    """One indexed passage. Always English.

    Attributes:
        id: ID of the unit the chunk was built from.
        article_id: The article-level ID it rolls up to; what gets scored.
        text: What the retriever sees: the unit text with its heading and title
            prepended, so a paragraph still "knows" which article it belongs to.
    """

    id: str
    article_id: str
    text: str

    @property
    def reg(self) -> str:
        return parse_id(self.id).reg

    @property
    def kind(self) -> str:
        return parse_id(self.id).kind


@dataclass(frozen=True)
class Hit:
    """One search result. ``rank`` starts at 1; higher ``score`` is better."""

    chunk_id: str
    article_id: str
    score: float
    rank: int
