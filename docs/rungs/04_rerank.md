# Rung 04 - Cross-encoder reranking

Status: not started (milestone M4).

## What it is

Take the top candidates from the best retriever so far (say 50) and re-score
each one with a slower, more accurate model that reads the query and the chunk
together. Return the best 10 by the new score.

## Theory

The embedding model of rung 02 is a **bi-encoder**: it encodes the query and
the chunk separately, and compares two vectors. That is what makes it fast (the
chunk vectors are computed once), and also what limits it: everything about a
long article has to be squeezed into one vector before the question is known.

A **cross-encoder** takes the query and one chunk as a single input and outputs
one relevance score. The model can attend from each query word to each chunk
word, so it judges the pair far better. The cost: one full model pass per
candidate, at query time, and nothing can be precomputed. Running it over the
whole corpus would be too slow, so it only re-orders a short list. Hence the
two stages: a cheap retriever for recall, an expensive reranker for precision.

A reranker cannot fix what was never retrieved. If the right article is not in
the candidate list, it stays lost. Recall@50 of the first stage is the ceiling
for Recall@10 after reranking.

## Implementation notes

- The reranker must be multilingual: it sees a Portuguese query next to English
  text. Choose the model at M4 from what is current, and pin the revision.
- `RerankRetriever` wraps any `Retriever`. Parameters: candidates in, results
  out. Evaluate two or three candidate depths on dev (20, 50, 100).
- This is the first rung where latency matters. Record per-stage timings
  (retrieve, rerank) at p50 and p95, on the owner's GPU and on CPU, since the
  deployment target has no GPU.
- Long chunks get truncated by the reranker too. Note the limit.
- Unit tests use a fake scorer; the real model is a `slow` test.

## Acceptance

- Result files for dev and test, with stage latencies.
- The gain is stated against its cost: Recall@5 and MRR change per added
  millisecond at p95.
- First-stage Recall@50 reported, as the ceiling.

## Questions the defend note must answer

- Bi-encoder versus cross-encoder: what does each see?
- Why not use the cross-encoder for everything?
- What is the ceiling on what reranking can achieve, and how close did we get?
- Is it worth it on a CPU-only server?
