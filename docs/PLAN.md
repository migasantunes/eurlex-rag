# eurlex-rag — Milestone plan

Status: draft v1 (12 Nov 2026). Read [SPEC.md](SPEC.md) and [EVAL.md](EVAL.md) first.

Eight milestones, each about one focused Claude Code session plus Miguel's
part. Each one ends with: a merged pull request, a results file where
applicable, and a decision note.

## How the work is run

- **Lead (Fable 5.2):** breaks the milestone into tasks, reviews every diff,
  runs the evaluation, writes the PR summary.
- **Implementers (Opus sub-agents):** one task each, on the milestone branch.
- **Miguel:** the items under "Your part", and the decision note. These are not
  optional: they are what lets you defend the project in an interview.

Branch per milestone: `m<N>-<slug>` → pull request → `main`.

## Decision notes

One short file per milestone in `docs/decisions/`, **written by Miguel**, from
the template there. The lead supplies the numbers; the reasoning is yours.
Three questions: what did we try, what did the numbers say, what did I decide
and why.

---

## M0 — Scaffold

**Goal:** an empty project that builds, tests and runs.

- `pyproject.toml` (uv), ruff, pytest, package skeleton per SPEC §6
- `compose.yaml` with PostgreSQL + pgvector; `Dockerfile` for the app
- `Makefile`: `setup`, `test`, `lint`, `up`, `ingest`, `eval`
- GitHub Actions: lint + tests
- `.env.example`; secrets never committed

**Done when:** `make up && make test` passes locally and in CI.
**Your part:** run it on your machine; read the Dockerfile line by line and ask
about anything unclear. Estimated 30 min.

## M1 — Ingestion

**Goal:** the five regulations as clean, ID-addressed units in six languages.

- Settle the access route and attribution wording (SPEC §2); cache raw files
  with hashes
- Parser: articles, paragraphs, recitals, annexes → `units` table
- Parallel lookup table (PT, FR, ES, DE, IT)
- Tests: same ID set in every language; no empty units; counts per regulation
  match the published structure

**Done when:** `make ingest` rebuilds the database from scratch and all
alignment tests pass.
**Your part:** open 10 random articles in EN, PT and FR side by side and
confirm they are the same article. Estimated 30 min.

## M2 — Benchmark, harness and rung 1

**Goal:** a trustworthy gold set, a runner, and the first row of the table.

- Gold set generation pipeline and automatic checks (EVAL §3)
- Metrics module with unit tests on hand-computed cases
- Runner: `make eval RUNG=1` → `results/*.json` → table
- Rung 1: BM25, on raw queries and on translated queries
- Review tool for verification (a simple CLI or page: question in three
  languages, gold article, accept / edit / reject)

**Done when:** rung 1 has dev and test numbers with confidence intervals.
**Your part:** verify the full test split and a 25% dev sample. This is the
largest block of your time in the project: estimated 2–3 hours. It is also the
part that makes every later number credible.

## M3 — Dense and hybrid (rungs 2–3)

**Goal:** vector search, then fusion.

- Embedding pipeline on GPU; pgvector index
- Compare at least two multilingual embedding models on dev
- Chunking experiment (article vs paragraph)
- Hybrid via reciprocal rank fusion
- CI smoke-eval with a committed threshold

**Done when:** rungs 2 and 3 are in the table, per language.
**Your part:** decision note. Which embedding model, which chunking, and does
hybrid earn its place? Estimated 1 hour.

## M4 — Reranking and query transformation (rungs 4–5)

**Goal:** the two classic precision levers, each with its latency bill.

- Cross-encoder reranker over the top-N candidates
- Rung 5: direct multilingual query vs translate-to-English, same retriever;
  optional query rewriting
- Latency and cost tracing per stage

**Done when:** rungs 4 and 5 are in the table with p95 latency.
**Your part:** decision note. This one carries the project's main finding:
translate first, or trust the multilingual model? Estimated 1 hour.

## M5 — Generation and citations (rung 0 and the full pipeline)

**Goal:** answers in the user's language with validated, article-level
citations.

- Provider interface; prompt; citation validation in code
- Abstention behaviour
- Citation display from the parallel table, with EUR-Lex links
- Answer metrics; rung 0 (no retrieval) as the baseline
- LLM judge and its agreement check (EVAL §5)

**Done when:** the answer columns of the table are filled for rung 0 and the
best retrieval pipeline.
**Your part:** hand-label ~40 answers for the judge check, and write the
decision note. Estimated 1.5 hours.

## M6 — Agentic multi-hop (rung 6)

**Goal:** answer tier-B questions that one retrieval pass cannot.

- LangGraph agent: decompose, retrieve per sub-question, follow
  cross-references between articles, stop conditions
- Evaluated on tier B specifically, and on tier A to check it does not make
  simple questions slower and worse

**Done when:** rung 6 is in the table, with cost per query next to the gain.
**Your part:** decision note. Is the agent worth it, and for which questions?
Estimated 1 hour.

## M7 — Serve, deploy, write up

**Goal:** something a recruiter can click, and a README that tells the story.

- FastAPI endpoints; one-page UI
- Container image; deployment to a small host (choice made here, with cost)
- README: problem, architecture diagram, results table generated from
  `results/`, three findings, limitations
- A short demo recording

**Done when:** the public URL answers a Portuguese question with Portuguese
citations, and the README table matches the results files.
**Your part:** deploy it yourself following the runbook, then write the
"findings" paragraph of the README in your own words. Estimated 1.5 hours.

---

## After M7

- CV bullets drafted from the final table (numbers only from `results/`)
- Optional LinkedIn write-up, one finding per post

## Out of scope unless asked

Fine-tuning, more regulations, multi-turn chat, authentication, user feedback
loop.
