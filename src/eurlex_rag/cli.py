"""The ``eurlex`` command line.

    eurlex rungs                    which rungs exist, and which are planned
    eurlex ingest                   download and parse the regulations
    eurlex search "question"        try one rung on one question
    eurlex eval --rung 01_bm25      score a rung on the benchmark, write a result file
    eurlex report                   rebuild results/table.md from the result files

Built on ``argparse`` from the standard library, so the CLI works with nothing
installed. Imports that need optional packages (the ingest pipeline, later the
embedding models) happen inside the command that uses them: listing rungs never
loads a model.

Every command returns an exit code: 0 for success, 1 for a problem the user can
fix (a missing file, a bad gold record). Problems are printed as one readable
line on stderr, not as a traceback.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from eurlex_rag.corpus import ALL_LANGS, INDEX_LANG, VERIFIED_LANGS
from eurlex_rag.eval.gold import DEFAULT_GOLD_PATH, SPLITS
from eurlex_rag.eval.results import DEFAULT_RESULTS_DIR
from eurlex_rag.index.chunking import STRATEGIES
from eurlex_rag.registry import BASELINE_RUNG
from eurlex_rag.store import DEFAULT_UNITS_PATH

SNIPPET_CHARS = 100


class CliError(Exception):
    """A problem to report in one line, without a traceback."""


def _fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


# --------------------------------------------------------------------------- rungs


def cmd_rungs(args: argparse.Namespace) -> int:
    from eurlex_rag.registry import PLANNED_RUNGS, available_rungs

    print("Available:")
    for rung in available_rungs():
        print(f"  {rung}")
    print("Planned:")
    for rung, milestone in sorted(PLANNED_RUNGS.items()):
        print(f"  {rung:<22} {milestone}")
    return 0


# -------------------------------------------------------------------------- ingest


def cmd_ingest(args: argparse.Namespace) -> int:
    try:
        from eurlex_rag.ingest.pipeline import run_ingest
    except ImportError as error:
        return _fail(
            f"the ingest pipeline could not be imported ({error}). "
            'Install its dependencies with: pip install -e ".[ingest]"'
        )

    report = run_ingest(
        raw_dir=Path(args.raw_dir),
        out_path=Path(args.out),
        langs=args.langs,
        fetch=not args.no_fetch,
        regs=args.regs,
    )
    langs = sorted({lang for per_lang in report.counts.values() for lang in per_lang})
    print("Units per regulation and language:")
    print("  " + "reg".ljust(8) + "".join(lang.rjust(8) for lang in langs))
    for reg, per_lang in sorted(report.counts.items()):
        print("  " + reg.ljust(8) + "".join(str(per_lang.get(lang, 0)).rjust(8) for lang in langs))
    if report.problems:
        print(f"Problems ({len(report.problems)}):")
        for problem in report.problems:
            print(f"  - {problem}")
    print(f"Wrote {report.out_path}")
    return 0 if report.ok else 1


# ------------------------------------------------------------------- shared setup


def _load_corpus(units_path: Path) -> list:
    """All units, every language. Raises ``CliError`` when the file is missing."""
    from eurlex_rag.store import load_units

    try:
        units = load_units(units_path)
    except FileNotFoundError:
        raise CliError(f"no units file at {units_path}. Build it with `eurlex ingest`.") from None
    except ValueError as error:
        raise CliError(str(error)) from None
    if not any(unit.lang == INDEX_LANG for unit in units):
        raise CliError(f"{units_path} has no {INDEX_LANG!r} units to index.")
    return units


def _indexed_retriever(rung: str, units: list, chunking: str, recitals: bool, annexes: bool):
    """Build the chunks from the English units and index them with ``rung``."""
    from eurlex_rag.index.chunking import build_chunks
    from eurlex_rag.registry import build_retriever

    try:
        retriever = build_retriever(rung)
    except ValueError as error:
        raise CliError(str(error)) from None
    english = [unit for unit in units if unit.lang == INDEX_LANG]
    chunks = build_chunks(
        english, strategy=chunking, include_recitals=recitals, include_annexes=annexes
    )
    retriever.index(chunks)
    return retriever, chunks


# -------------------------------------------------------------------------- search


def _snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def cmd_search(args: argparse.Namespace) -> int:
    from eurlex_rag.store import ParallelTable

    units = _load_corpus(Path(args.units))
    retriever, _ = _indexed_retriever(args.rung, units, args.chunking, True, True)
    table = ParallelTable(units)

    hits = retriever.search(args.query, args.k)
    if not hits:
        print("No hits.")
        return 0
    for hit in hits:
        # Show the article in the requested language; with paragraph chunks the
        # whole article may not be stored, so fall back to the chunk's own unit.
        unit = table.get(hit.article_id, args.lang) or table.get(hit.chunk_id, args.lang)
        text = unit.text if unit else ""
        title = f"{unit.title} - " if unit and unit.title else ""
        print(f"{hit.rank:>3}  {hit.score:7.3f}  {hit.article_id:<14} {_snippet(title + text)}")
    return 0


# ---------------------------------------------------------------------------- eval


def cmd_eval(args: argparse.Namespace) -> int:
    from eurlex_rag.eval.gold import by_split, load_gold, validate_against_corpus
    from eurlex_rag.eval.results import sha256_file, write_result
    from eurlex_rag.eval.runner import run_retrieval_eval
    from eurlex_rag.ids import article_level

    units_path, gold_path = Path(args.units), Path(args.gold)
    if args.split == "test":
        print(
            "Reminder: test is run once per rung, on a frozen configuration. "
            "Choose settings on dev (CLAUDE.md rule 2)."
        )

    units = _load_corpus(units_path)
    try:
        questions = load_gold(gold_path)
    except FileNotFoundError as error:
        raise CliError(str(error)) from None
    except ValueError as error:
        raise CliError(f"invalid gold file: {error}") from None

    known = {article_level(unit.id) for unit in units if unit.lang == INDEX_LANG}
    problems = validate_against_corpus(questions, known)
    if problems:
        details = "\n  ".join(problems)
        raise CliError(f"gold file does not match the corpus:\n  {details}")

    selected = by_split(questions, args.split)
    if not selected:
        raise CliError(f"no {args.split} questions in {gold_path}.")

    retriever, chunks = _indexed_retriever(
        args.rung, units, args.chunking, not args.no_recitals, not args.no_annexes
    )
    evaluation = run_retrieval_eval(retriever, selected, args.langs, k=args.k)
    if not evaluation["per_query"]:
        raise CliError("no answerable questions to score in the selected split and languages.")

    settings = {
        "chunking": args.chunking,
        "include_recitals": not args.no_recitals,
        "include_annexes": not args.no_annexes,
        "k": args.k,
        "langs": list(args.langs),
        "units_path": str(units_path),
        "units_sha256": sha256_file(units_path),
        "gold_path": str(gold_path),
        "gold_sha256": sha256_file(gold_path),
        "n_units_indexed_lang": sum(1 for unit in units if unit.lang == INDEX_LANG),
        "n_chunks": len(chunks),
        "n_questions_in_split": len(selected),
    }
    path = write_result(args.out, args.rung, args.split, evaluation, retriever.describe(), settings)

    print(f"{args.rung} on {args.split}: {evaluation['n_queries']} queries", end="")
    print(f", {evaluation['excluded_unanswerable']} unanswerable question(s) excluded")
    for lang, summary in evaluation["by_lang"].items():
        r5, mrr = summary["recall@5"], summary["mrr@10"]
        interval = f"[{r5['low']:.2f}, {r5['high']:.2f}]"
        print(
            f"  {lang}  n={summary['n']:<4} R@5 {r5['mean']:.2f} {interval}"
            f"  MRR@10 {mrr['mean']:.2f}"
        )
    print(f"  p95 latency {evaluation['latency_ms']['p95']:.1f} ms")
    print(f"Wrote {path}")
    return 0


# -------------------------------------------------------------------------- report


def cmd_report(args: argparse.Namespace) -> int:
    from eurlex_rag.eval.report import write_report
    from eurlex_rag.eval.results import load_results

    results_dir = Path(args.results)
    try:
        results = load_results(results_dir)
    except ValueError as error:
        raise CliError(str(error)) from None
    if not results:
        raise CliError(f"no result files in {results_dir}. Run `eurlex eval` first.")
    path = write_report(results_dir)
    print(path.read_text(encoding="utf-8"), end="")
    print(f"Wrote {path}")
    return 0


# -------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eurlex", description="Cross-lingual RAG over EU regulation, built as a ladder."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    rungs = sub.add_parser("rungs", help="list available and planned rungs")
    rungs.set_defaults(func=cmd_rungs)

    ingest = sub.add_parser("ingest", help="fetch, parse and write the units file")
    ingest.add_argument("--raw-dir", default="data/raw")
    ingest.add_argument("--out", default=str(DEFAULT_UNITS_PATH))
    ingest.add_argument("--langs", nargs="+", default=list(ALL_LANGS))
    ingest.add_argument("--regs", nargs="+", default=None, help="regulation short names")
    ingest.add_argument("--no-fetch", action="store_true", help="only parse files already in raw")
    ingest.set_defaults(func=cmd_ingest)

    search = sub.add_parser("search", help="run one query against one rung")
    search.add_argument("query")
    search.add_argument("--rung", default=BASELINE_RUNG)
    search.add_argument("--k", type=int, default=5)
    search.add_argument("--units", default=str(DEFAULT_UNITS_PATH))
    search.add_argument("--chunking", choices=STRATEGIES, default="article")
    search.add_argument("--lang", default=INDEX_LANG, help="language to display articles in")
    search.set_defaults(func=cmd_search)

    evaluate = sub.add_parser("eval", help="score a rung on the benchmark")
    evaluate.add_argument("--rung", default=BASELINE_RUNG)
    evaluate.add_argument("--split", choices=SPLITS, default="dev")
    evaluate.add_argument("--units", default=str(DEFAULT_UNITS_PATH))
    evaluate.add_argument("--gold", default=str(DEFAULT_GOLD_PATH))
    evaluate.add_argument("--langs", nargs="+", default=list(VERIFIED_LANGS))
    evaluate.add_argument("--k", type=int, default=10)
    evaluate.add_argument("--chunking", choices=STRATEGIES, default="article")
    evaluate.add_argument("--no-recitals", action="store_true")
    evaluate.add_argument("--no-annexes", action="store_true")
    evaluate.add_argument("--out", default=str(DEFAULT_RESULTS_DIR))
    evaluate.set_defaults(func=cmd_eval)

    report = sub.add_parser("report", help="rebuild results/table.md")
    report.add_argument("--results", default=str(DEFAULT_RESULTS_DIR))
    report.set_defaults(func=cmd_report)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``eurlex`` console script. Returns the exit code."""
    args = build_parser().parse_args(argv)
    if getattr(args, "k", 1) <= 0:
        return _fail("--k must be positive")
    try:
        return args.func(args)
    except CliError as error:
        return _fail(str(error))


if __name__ == "__main__":
    sys.exit(main())
