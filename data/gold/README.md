# Gold benchmark

This directory will hold `questions.jsonl`, the benchmark every rung is scored
on. **It does not exist yet**: the real benchmark is built in milestone M2.
Do not add invented questions here; tests use their own fixtures.

## Format

One JSON object per line:

```json
{"qid": "q0042", "tier": "A", "split": "test", "gold": ["AIA:art:6"],
 "source_lang": "pt", "text": {"en": "…", "pt": "…", "fr": "…"},
 "verified_by": "miguel", "notes": ""}
```

- `tier`: `A` one article, `B` several articles, `C` unanswerable (empty `gold`).
- `split`: `dev` for choosing settings, `test` for the numbers that are quoted.
  All language versions of a question sit on the same side.
- `gold`: article-level IDs (`AIA:art:6`, not `AIA:art:6:2`) that must exist
  in the corpus.
- `text`: the question per language; `en` is required.

`eurlex eval` refuses a file that breaks any of these rules and names the line.

## How it is built and verified

Questions are drafted from the Portuguese or French text of an article, then
translated, checked automatically, and verified by hand (every test item, a 25%
sample of dev). The full procedure is in [docs/EVAL.md](../../docs/EVAL.md) §3–4.
