# Rung 05 - Query transformation

Status: not started (milestone M4).

## What it is

Change the query before searching. Two transformations, measured separately:

1. **Translate** a non-English query to English, then search.
2. **Rewrite** the query into the vocabulary of the documents (for example,
   turning "can I make a company delete my data?" into something closer to the
   regulation's own terms).

## The question this rung answers

For a Portuguese question against an English index there are two designs:

| Design | How |
|---|---|
| Direct | Embed the Portuguese question with the multilingual model |
| Translate first | Translate to English with an LLM, then embed (and BM25) the English |

Teams building multilingual search have to pick one. Translate-first adds an
LLM call (latency, cost, a new way to fail) but lets BM25 take part and hands
the embedding model the language it was trained on most. Direct is simpler and
depends entirely on how well the model aligns languages. Which is better on
legal text is an empirical question, and this project answers it for one model
and one corpus.

## Implementation notes

- **Keep the comparison clean.** Same retriever, same chunks, same `k` on both
  sides. The only thing that changes is the query string.
- A `TransformingRetriever` wraps any `Retriever` and a transform function, so
  the harness needs no change.
- Language detection: needed to decide whether to translate. A wrong guess on a
  short query is a real failure mode; count them.
- Translation must not answer the question or add terms. Prompt it to translate
  only, and spot-check.
- Cache translations by query text so re-running an evaluation does not pay
  twice, and so results are repeatable. Commit the cache used for reported
  runs.
- This rung also supplies `01_bm25_translated`, the fair lexical baseline.
- Rewriting uses an LLM too. Evaluate it on English queries as well: it can
  help or hurt independently of language.
- Translation quality differs by language. Report per language; do not pool.

## Acceptance

- Rows for: direct, translate-first, and (if tried) rewrite, per language, with
  added latency and cost per query.
- `01_bm25_translated` row filled in.
- A stated recommendation with the trade-off, in the decision note.

## Questions the defend note must answer

- Translate first or trust the multilingual model? What did the numbers say,
  and how confident are we given the sample size?
- What does translation cost per query, and what breaks when the translation
  service is down?
- Why is BM25-on-translated-query the right baseline to compare dense against?
- Show a query where translation changed the meaning.
