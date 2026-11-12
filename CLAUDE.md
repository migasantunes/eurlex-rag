# Working in this repository

Cross-lingual RAG over EU regulation, built as a measured ladder.
Read `docs/SPEC.md`, `docs/EVAL.md` and `docs/PLAN.md` before changing anything.

## Roles

- The lead session plans each milestone from `docs/PLAN.md`, splits it into
  tasks, reviews every diff and runs the evaluation.
- Implementation is delegated to Opus sub-agents, one task each, on the
  milestone branch `m<N>-<slug>`. Work reaches `main` through a pull request.
- Miguel owns the "Your part" items and the decision notes.

## Rules

1. **No number without a run.** Every figure in the README, a PR or a decision
   note comes from a file in `results/`. Never write an expected or estimated
   metric as if it were measured.
2. **Never tune on the test split.** Choose on dev. Test runs once per rung,
   after the config is frozen. If that is broken, say so in the PR.
3. **Decision notes are Miguel's.** Do not write `docs/decisions/M*.md` for
   him. Supply the numbers and, if asked, questions to think about.
4. **Explain as you go.** When a milestone introduces something new to Miguel
   (Dockerfiles, pgvector, rank fusion, cross-encoders, LangGraph), the PR
   description includes a short plain-language explanation of what it is and
   why it is there.
5. **Non-English legal text is never embedded, indexed or machine-translated.**
   It lives in the parallel table for citation display and benchmark building.
6. **A rung that does not help is reported and left out** of the default
   pipeline, not quietly dropped or kept.
7. Model IDs, providers and dates are pinned in `configs/` and recorded in
   every results file. Check what is current before choosing a model; do not
   rely on memory.
8. Secrets live in `.env` (git-ignored). `data/raw/` is git-ignored and must be
   reproducible with `make ingest`.
9. Keep dependencies few. LangGraph is used in `agent/` only.

## Commands

Defined in M0; keep this list current.

```
make setup     install
make up        start database
make ingest    fetch, parse, load
make test      unit tests
make lint      ruff
make eval RUNG=<n> SPLIT=dev|test
```
