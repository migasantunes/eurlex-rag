# Rung 02 - Dense retrieval

Status: not started (milestone M3).

## What it is

Every chunk is turned into a vector by an embedding model, once, offline. A
query is turned into a vector by the same model, and the nearest chunk vectors
are the results.

## Theory

An embedding model is a neural network trained so that texts with similar
meaning land close together in a space of a few hundred or thousand dimensions.
"Close" is measured with cosine similarity: the cosine of the angle between two
vectors, 1 for the same direction, 0 for unrelated.

A **multilingual** embedding model is trained on many languages at once, with
translation pairs pulled together. A Portuguese question and the English
article that answers it should then be near each other, although they share no
words. This is the property the whole project tests.

What dense retrieval is bad at: exact identifiers and rare terms. "Article
6(2)" or an unusual defined term may be a blur to the model while being a
perfect match for BM25. That weakness is the reason rung 03 exists.

## Implementation notes

- **Model choice is an experiment.** Compare at least two open-weight
  multilingual models on dev. BGE-M3 is the starting candidate; check what is
  current before choosing, and pin the exact model revision in the config.
- Some models expect a prefix or instruction on queries and a different one on
  passages. Getting this wrong silently costs accuracy. Read the model card.
- Embeddings are normalised to unit length so cosine similarity is a dot
  product.
- Storage: PostgreSQL with the pgvector extension (`compose.yaml` already
  starts it). One table: chunk ID, article ID, text, vector. Start with exact
  search (no index): a few thousand vectors are scanned in milliseconds, and
  exact search removes one source of error. An approximate index (HNSW) is only
  worth adding to show the recall/latency trade-off, as its own row.
- Embedding runs on the owner's GPU and the vectors are cached under
  `artifacts/`, keyed by model revision and chunking strategy.
- Chunk length matters more here than for BM25: models have a maximum input
  length and truncate silently. Log how many chunks exceed it.
- Implement `Retriever` in `src/eurlex_rag/retrieve/dense.py` and register it.
  Unit tests use a tiny fake embedder (e.g. hashed bag-of-words vectors), never
  a downloaded model; a test with the real model is marked `slow`.

## Acceptance

- Result files for dev and test, per language.
- The gap between English and non-English Recall@5 is reported next to rung
  01's. Whether dense "crosses the language gap" is read off that comparison.
- CI smoke-eval gate added (EVAL §6), with a small committed vector fixture or
  the fake embedder so CI does not download a model.

## Questions the defend note must answer

- What is an embedding, and what was the model trained to do?
- Why can a Portuguese query match an English article?
- Why cosine similarity and not Euclidean distance? (And when are they the same?)
- What is pgvector doing, and why no approximate index at this size?
- Give one question where dense lost to BM25, and explain why.
