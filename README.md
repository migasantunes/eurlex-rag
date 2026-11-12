# eurlex-rag

An evaluation-driven, cross-lingual RAG system over EU law. Ask in any
language; it retrieves from English sources and answers in your language, with
article-level citations shown in the official text of your language.

**Status:** planning complete, build not started. No results yet.

## Why

Most "chat with your documents" projects have no numbers. This one is built as
a ladder: keyword search, vector search, hybrid, reranking, query
transformation, an agent. Each rung is measured on a held-out benchmark and
kept only if it earns its place.

The benchmark uses a property of EU law: every regulation is published in 24
languages with identical article numbering. A question written from the
Portuguese text has a known correct English article, so cross-lingual
retrieval can be scored without manual relevance labels.

## Corpus

GDPR, Digital Markets Act, Digital Services Act, Data Act, AI Act.

## Documents

- [docs/SPEC.md](docs/SPEC.md) — what the system is and how it is built
- [docs/EVAL.md](docs/EVAL.md) — the benchmark and the metrics
- [docs/PLAN.md](docs/PLAN.md) — milestones
- [docs/decisions/](docs/decisions/) — one note per milestone on what was chosen and why

## Results

To be generated from `results/` as each rung is measured.

## Source and licence of the texts

Legal texts © European Union, from EUR-Lex. Not legal advice.
