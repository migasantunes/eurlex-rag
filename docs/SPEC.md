# eurlex-rag — System specification

Status: draft v1 (12 Nov 2026). Owner: Miguel Antunes.
Companion documents: [EVAL.md](EVAL.md) (how we measure), [PLAN.md](PLAN.md) (milestones).

## 1. What this is

A question-answering system over EU digital regulation. A user asks in any
language; the system searches an **English-only** index, answers **in the
user's language**, and cites the exact articles it relied on, showing each
cited article in the **official text of the user's language** when one exists.

The project is built as a ladder. Each retrieval technique is a rung, and a
rung stays in the system only if it improves a measured number on a held-out
benchmark. The deliverable is the working system *and* the results table that
justifies every part of it.

### Why English-only retrieval

Most real corpora exist in one language: company policies, case law, guidance
notes, contracts. Their users do not all speak it. EU regulations happen to be
published in 24 languages, and we use that fact for *evaluation only* (see
EVAL.md §2): the translations give us ground truth for free, while the system
itself solves the realistic problem.

### Non-goals

- Not legal advice, and the UI says so.
- Not all of EUR-Lex. Five regulations, fixed versions.
- No fine-tuning of embedding models or LLMs. Off-the-shelf models, measured.
- No user accounts, chat history or multi-turn memory in v1.

## 2. Corpus

| Short name | Regulation | CELEX |
|---|---|---|
| GDPR | Regulation (EU) 2016/679 | 32016R0679 |
| DMA | Regulation (EU) 2022/1925 | 32022R1925 |
| DSA | Regulation (EU) 2022/2065 | 32022R2065 |
| DATA | Regulation (EU) 2023/2854 (Data Act) | 32023R2854 |
| AIA | Regulation (EU) 2024/1689 (AI Act) | 32024R1689 |

Roughly 400 articles and 700 recitals, plus annexes: a few megabytes of text
and a few thousand chunks.

- **Version pinning.** We ingest the text as published in the Official Journal
  and record the retrieval date and a content hash per document. Later
  amendments are out of scope; the README states the cut-off.
- **Source.** The Publications Office (EUR-Lex / Cellar). EUR-Lex content is
  free to reuse subject to its copyright notice; M1 confirms the exact
  attribution wording and the supported programmatic route (Cellar content
  negotiation by CELEX and language is the expected one; scraping the website
  UI is the fallback, not the plan).
- **Languages stored.** EN is indexed. PT, FR, ES, DE and IT are stored as a
  **parallel lookup table** keyed by article ID. They are never embedded or
  searched. They serve two purposes: showing citations in the user's language,
  and building the benchmark.

### Units and IDs

Every piece of text gets a stable, language-independent ID:

```
AIA:art:6          article
AIA:art:6:2        paragraph 2 of article 6
AIA:rec:47         recital 47
AIA:anx:III        annex III
```

Article numbering is identical across language versions, which is what makes
the parallel table and the benchmark possible. The ingestion tests assert it
(same set of IDs in every stored language).

### Chunking

- Default chunk = one article, with its title and chapter heading prepended.
- Articles above a token threshold are split by numbered paragraph; each
  sub-chunk keeps the article title as context and points to its parent.
- Recitals and annex sections are separate chunks, tagged by kind so they can
  be included or excluded per experiment.
- Retrieval is always scored at **article level** (a hit on `AIA:art:6:2`
  counts as `AIA:art:6`).

Chunking strategy is itself an experiment variable in M3, not a fixed choice.

## 3. Architecture

```
            ┌────────────── offline ──────────────┐
 Cellar ──► fetch ──► parse ──► units (all langs) ──► chunks (EN) ──► embed
                                     │                     │            │
                                     ▼                     ▼            ▼
                              parallel table          BM25 index     pgvector

            ┌────────────── online ───────────────┐
 query ──► detect language ──► [transform] ──► retrieve ──► [rerank]
                                                               │
                                     answer in query language ◄┘ generate
                                                │
                              validate citations against retrieved set
                                                │
                       attach official text of cited articles (user's language)
```

Square brackets mark stages that only exist from a given rung onwards. The
pipeline is assembled from a config file, so that every rung can be run and
re-run on the benchmark by name.

### Components

| Component | Choice | Note |
|---|---|---|
| Language | Python 3.12 | `uv` for environments, `ruff`, `pytest` |
| Store | PostgreSQL + pgvector | one database for text, metadata and vectors |
| Lexical | BM25 (in-process, `bm25s`) | true BM25; Postgres full-text as a compared alternative |
| Embeddings | multilingual open-weight model | BGE-M3 as the starting candidate; re-check the field at M3 and compare at least two |
| Reranker | multilingual cross-encoder | candidate chosen at M4 |
| LLM | behind a provider interface | hosted open model by default; model and provider fixed per results run and recorded |
| Agent | LangGraph | rung 6 only; nothing else depends on it |
| API | FastAPI | `/ask`, `/retrieve`, `/health` |
| UI | one page | question box, answer, cited articles side by side |
| Packaging | Docker + Compose | app + database |
| CI | GitHub Actions | lint, tests, retrieval smoke-eval with thresholds |

Model names are deliberately not frozen here: this field moves in months, so
each is chosen at its milestone, written into the decision note, and pinned in
the run config.

## 4. The ladder

| Rung | Adds | Question it answers |
|---|---|---|
| 0 | No retrieval (LLM alone) | How much does the model already know, and how often does it invent article numbers? |
| 1 | BM25 | What does plain keyword search get, on English queries and on non-English ones? |
| 2 | Dense retrieval | Do multilingual embeddings cross the language gap on their own? |
| 3 | Hybrid (rank fusion) | Does lexical + dense beat either alone? |
| 4 | Cross-encoder reranking | Is the extra latency worth the precision? |
| 5 | Query transformation | Embed the foreign-language query directly, or translate to English first? Does rewriting help? |
| 6 | Agentic multi-hop | Can decomposition and following cross-references answer questions spanning several articles or regulations? |

Rungs 1–5 are judged on retrieval metrics; rungs 0 and 6 and the final system
on answer metrics as well. A rung that does not earn its place is **reported
and left out** of the default pipeline. A negative result is a result.

## 5. Generation and citations

- The LLM receives the question and the retrieved chunks, each labelled with
  its ID, and is instructed to answer only from them, in the language of the
  question, citing IDs inline.
- **Citation validation** is code, not prompt: every cited ID must be in the
  retrieved set. Invalid citations are stripped and counted.
- **Abstention.** If the retrieved context does not support an answer, the
  system says so. The benchmark includes unanswerable questions to measure
  this.
- **Citation display.** For each cited ID, the response carries the official
  text in the user's language from the parallel table (falling back to
  English), plus the EUR-Lex link for that language. Legal text is never
  machine-translated by the system.

Response shape:

```json
{
  "answer": "…",
  "language": "pt",
  "citations": [
    {"id": "AIA:art:6", "title": "…", "text": "…", "text_language": "pt", "url": "…"}
  ],
  "abstained": false,
  "trace": {"pipeline": "rung4", "retrieved": ["…"], "latency_ms": {}, "tokens": {}}
}
```

## 6. Repository layout

```
src/eurlex_rag/
  ingest/      fetch, parse, units, parallel table
  index/       chunking, BM25, embeddings, pgvector
  retrieve/    one module per retriever + fusion + rerank
  transform/   language detection, translation, rewriting
  generate/    prompts, provider interface, citation validation
  agent/       rung 6
  eval/        metrics, runner, reports
  api/         FastAPI app
configs/       one YAML per rung
data/gold/     benchmark (committed)
data/raw/      downloaded sources (git-ignored, reproducible via make)
results/       one JSON per run + the summary table
docs/          this spec, eval design, plan, decision notes
tests/
```

## 7. Risks

| Risk | Mitigation |
|---|---|
| Programmatic access to the source is awkward or rate-limited | Settle in M1 before anything depends on it; cache raw files with hashes; fall back to one-off manual download (10 files per language) |
| Benchmark questions leak the answer's wording | Questions are drafted from non-English text and paraphrased; a lexical-overlap check flags suspicious items (EVAL.md §3) |
| Tuning on the test set | Dev/test split; test is run once per rung, at the end |
| LLM-as-judge is unreliable | Judge is validated against hand labels on a sample before its numbers are reported |
| Results depend on a hosted model that changes | Model ID and date recorded per run; retrieval metrics do not depend on the LLM at all |
| The owner cannot explain the system | Decision note per rung, written by Miguel (PLAN.md) |
