# eurlex-rag

An evaluation-driven, cross-lingual RAG system over EU law. Ask in any
language; it retrieves from English sources and answers in your language, with
article-level citations shown in the official text of your language.

**Status: scaffold.** The ingestion code, the BM25 baseline, the evaluation
harness, the command line and the retrieval service are implemented and
unit-tested on a small synthetic corpus. The real regulations have not been
ingested and the benchmark does not exist yet, so there are no results to show.
The results table will appear here once the runs in milestone M2 onwards are
done.

## Why

Most "chat with your documents" projects have no numbers. This one is built as
a ladder: each retrieval technique is a rung, measured on a held-out benchmark,
and kept only if it earns its place.

The benchmark uses a property of EU law: every regulation is published in 24
languages with identical article numbering. A question written from the
Portuguese text has a known correct English article, so cross-lingual
retrieval can be scored without manual relevance labels.

The index is English only, because that is the realistic case: most document
collections exist in one language and their users do not all speak it.

## The ladder

| Rung | What it adds | State |
|---|---|---|
| `00_no_retrieval` | The LLM alone, as a control | Planned (M5) |
| `01_bm25` | Keyword search, written by hand | Implemented, not measured |
| `02_dense` | Multilingual embeddings in pgvector | Planned (M3) |
| `03_hybrid` | Rank fusion of keyword and dense | Planned (M3) |
| `04_rerank` | Cross-encoder reranking | Planned (M4) |
| `05_query_transform` | Translate or rewrite the query first | Planned (M4) |
| `06_agentic` | Multi-hop agent for questions spanning articles | Planned (M6) |

Corpus: GDPR, Digital Markets Act, Digital Services Act, Data Act, AI Act.

## Where things are

| Path | Contents |
|---|---|
| [`docs/PLAN.md`](docs/PLAN.md) | Milestones, what is done, acceptance criteria |
| [`docs/SPEC.md`](docs/SPEC.md) | What the system is and how it is built |
| [`docs/EVAL.md`](docs/EVAL.md) | The benchmark, the metrics, the protocol |
| [`docs/rungs/`](docs/rungs) | One spec per rung |
| [`docs/defend/`](docs/defend) | Plain-language notes on each idea |
| [`docs/decisions/`](docs/decisions) | One note per milestone on what was chosen and why |
| [`docs/m1-parser-checklist.md`](docs/m1-parser-checklist.md) | What to verify against the real EUR-Lex files |
| `src/eurlex_rag/ingest/` | Download, parse, alignment checks |
| `src/eurlex_rag/retrieve/` | One module per way of searching |
| `src/eurlex_rag/eval/` | Gold set, metrics, bootstrap intervals, runner, report |
| `src/eurlex_rag/generate/` | Citation extraction and validation |
| `src/eurlex_rag/api/` | FastAPI service |
| `results/` | Result files and the generated table (once runs exist) |

## Quick start

    python -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev,serve,ingest]"
    pytest                       # unit tests, no downloads

Build the corpus (downloads the five regulations in six languages):

    eurlex ingest
    eurlex search "notify a data breach" --lang pt

Measure a rung (needs the benchmark from milestone M2):

    eurlex eval --rung 01_bm25 --split dev
    eurlex report                # writes results/table.md

Run the service:

    docker compose up --build
    curl -s localhost:8000/retrieve -H 'content-type: application/json' \
         -d '{"query": "notify a data breach", "k": 3, "lang": "pt"}'

## Measurement in one paragraph

Retrieval is scored at article level against gold articles that come from the
alignment between language versions. Recall@5, MRR@10 and nDCG@10 are reported
per query language, with 95% bootstrap intervals, because the gap between
English and the other languages is the result. Settings are chosen on a dev
split; the test split is run once per rung. Every number in this README is
generated from a file under `results/`.

## Source of the texts

Legal texts © European Union, from EUR-Lex. Not legal advice.
