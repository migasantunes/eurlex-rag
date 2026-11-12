# Rung 03 - Hybrid (rank fusion)

Status: not started (milestone M3).

## What it is

Run BM25 and dense retrieval for the same query and merge the two ranked lists
into one.

## Theory

The two methods fail on different questions. BM25 finds exact terms and misses
paraphrases and other languages. Dense finds meaning and misses exact
identifiers. If their mistakes are not the same mistakes, combining them
should beat both.

Their scores cannot be added directly: a BM25 score of 12 and a cosine of 0.7
are on unrelated scales. **Reciprocal rank fusion (RRF)** avoids the problem by
using only the *rank* each method gave a chunk:

    rrf(d) = sum over methods m of  1 / (k + rank_m(d))

A chunk ranked 1st by one method and 3rd by the other scores
`1/(k+1) + 1/(k+3)`. A chunk that only one method returned gets one term.
`k` (commonly 60) controls how much the top ranks dominate. There is nothing
to train, and it is about ten lines of code.

## Implementation notes

- `HybridRetriever` takes two already-built retrievers and a `k`. It asks each
  for more candidates than it returns (e.g. 50 each for a final 10).
- For non-English queries the BM25 half contributes little or noise. Measure
  it per language: hybrid may help English and do nothing, or harm, elsewhere.
  If so, say so; one option to evaluate is fusing only when the query language
  is English, another is feeding BM25 the translated query (rung 05).
- A weighted variant (score normalisation plus a weight) is an optional second
  row. Tune any weight on dev.

## Acceptance

- Result files for dev and test, per language.
- A paired bootstrap against the better of rungs 01 and 02, per language.
- A short error analysis: for the dev questions where the three rungs disagree,
  which method was right, and is there a pattern?

## Questions the defend note must answer

- Why can't the two scores be added?
- Work an RRF example by hand.
- When does fusion hurt?
- Did hybrid earn its place for Portuguese and French queries, or only for
  English?
