"""The whole ingest in one call: fetch, parse, check, write.

``run_ingest`` is what ``eurlex ingest`` runs. It goes through every
(regulation, language) pair, keeps going when one fails, and collects every
problem into one ``IngestReport``, so a single run tells you everything that
needs fixing.

The units file is written even when there are problems, so you can open it and
see what the parser produced. ``report.ok`` is what says whether the corpus can
be trusted; the CLI turns a failed report into a non-zero exit.

``fetch`` and ``parse`` need the optional ``ingest`` dependencies (httpx,
BeautifulSoup, lxml). They are imported inside ``run_ingest``, so importing this
module, for instance from the CLI, does not require them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from eurlex_rag.corpus import ALL_LANGS, REGULATIONS, by_short
from eurlex_rag.ingest.checks import EXPECTED_COUNTS, check_units
from eurlex_rag.store import write_units
from eurlex_rag.types import Unit

if TYPE_CHECKING:
    import httpx


@dataclass
class IngestReport:
    """What an ingest produced.

    Attributes:
        counts: Number of units per regulation and language, e.g.
            ``{"AIA": {"en": 812, "pt": 812}}``.
        problems: Everything that went wrong, in plain language. Empty if all is well.
        out_path: Where the units were written.
    """

    counts: dict[str, dict[str, int]] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    out_path: Path = Path()

    @property
    def ok(self) -> bool:
        return not self.problems


def run_ingest(
    raw_dir: Path,
    out_path: Path,
    langs: Sequence[str] = ALL_LANGS,
    fetch: bool = True,
    regs: Sequence[str] | None = None,
    *,
    expected: Mapping[str, Mapping[str, int]] = EXPECTED_COUNTS,
    client: httpx.Client | None = None,
    delay: float | None = None,
) -> IngestReport:
    """Build the units file from the raw documents.

    Args:
        raw_dir: Folder of downloaded files (``{CELEX}.{lang}.html``) and the manifest.
        out_path: The units file to write (JSON Lines, see ``eurlex_rag.store``).
        langs: Languages to ingest. English must be among them: it is the
            reference the other languages are checked against.
        fetch: Download files that are missing. With ``False``, only files
            already in ``raw_dir`` are used and missing ones are reported.
        regs: Short names of the regulations to ingest; all of them by default.
        expected: Published article and recital counts to check against. Pass
            ``{}`` to skip that check (for test fixtures).
        client: An ``httpx.Client`` for downloading; tests pass a fake one.
        delay: Seconds between requests; ``None`` uses the default in ``fetch``.
    """
    from eurlex_rag.ingest.fetch import DELAY_SECONDS, FetchError, fetch_document, raw_path
    from eurlex_rag.ingest.parse import ParseError, parse_document

    raw_dir, out_path = Path(raw_dir), Path(out_path)
    regulations = [by_short(short) for short in regs] if regs else list(REGULATIONS)
    report = IngestReport(out_path=out_path)
    units: list[Unit] = []

    for reg in regulations:
        report.counts[reg.short] = {}
        for lang in langs:
            path = raw_path(raw_dir, reg.celex, lang)
            if fetch:
                try:
                    fetch_document(
                        reg.celex,
                        lang,
                        raw_dir,
                        client=client,
                        delay=DELAY_SECONDS if delay is None else delay,
                    )
                except FetchError as error:
                    report.problems.append(str(error))
                    continue
            elif not path.exists():
                report.problems.append(
                    f"{reg.short}/{lang}: missing {path}. Run the ingest with fetching on, "
                    "or download the document by hand and save it there."
                )
                continue

            try:
                parsed = parse_document(path.read_text(encoding="utf-8"), reg.short, lang)
            except ParseError as error:
                report.problems.append(f"{path}: {error}")
                continue
            report.counts[reg.short][lang] = len(parsed)
            units.extend(parsed)

    report.problems.extend(check_units(units, expected=expected))
    write_units(units, out_path)
    return report
