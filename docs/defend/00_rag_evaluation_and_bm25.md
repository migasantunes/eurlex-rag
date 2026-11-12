# Defend it: RAG, how we measure it, and BM25

Notes for explaining the foundation of the project out loud. Numbers get added
after the M2 runs; until then this covers the reasoning only.

## What RAG is

A language model answers from what it memorised during training. That fails in
three ways: it does not know your documents, its knowledge stops at a date, and
when it does not know it may produce something fluent and wrong.

Retrieval-augmented generation puts a search step in front. Find the passages
relevant to the question, paste them into the prompt, and tell the model to
answer from them and say which ones it used. The model's job changes from
"remember" to "read and summarise", which it is much better at, and every claim
can be traced to a source.

So a RAG system is two systems: a **search engine** and a **reader**. Most of
the quality is decided by the search engine. If the right passage is not
retrieved, the best model in the world cannot cite it.

## Why this project is a ladder

There are many retrieval techniques and each adds cost and complexity. The
ladder adds one at a time and measures each. A technique stays only if the
benchmark says it helps. That turns "I used hybrid search and a reranker" into
"hybrid search moved Recall@5 from X to Y for Portuguese queries, and the
reranker added Z at a cost of N milliseconds".

## The cross-lingual setup

Users ask in any language. The index is English only, because that is the
realistic case: most document collections exist in one language.

EU regulations are published in every EU language with identical article
numbering. We use the translations for two things and never search them:

- to show a cited article in the user's language, in its official wording;
- to build the benchmark: a question written from the Portuguese text of an
  article has that article's number as its correct answer.

## Why chunk, and why by article

Search returns passages, not whole documents. The unit you index is the unit
you can find and cite. Legal text has a natural unit, the article, so we start
there. Long articles can be split by numbered paragraph. Each chunk carries its
chapter heading and article title, because a paragraph on its own often never
mentions what it is about.

## The metrics

All retrieval metrics compare a ranked list of articles against the gold
articles for a question.

- **Recall@k:** of the gold articles, what fraction appears in the top k? With
  one gold article it is simply 1 or 0. Recall@5 is the headline number,
  because about five passages is what we hand to the reader.
- **MRR (mean reciprocal rank):** 1 divided by the rank of the first correct
  article, averaged. First place scores 1, second 0.5, third 0.33. It rewards
  putting the right answer near the top.
- **nDCG@k:** like recall, but each hit is discounted by how far down it is,
  and the total is divided by the best achievable. It matters when a question
  has several gold articles.

Why more than one: Recall@10 can be perfect while the right article sits at
rank 10, where the reader may ignore it. MRR and nDCG catch that.

## Dev, test, and why the split matters

Every choice (chunk size, parameters, which model) is made by looking at
numbers. If you choose on the same questions you report on, the reported
number is optimistic: you have fitted the system to those questions. So the
benchmark is split. Choices are made on **dev**. **Test** is run once per rung,
after the configuration is frozen.

## Confidence intervals

With about 60 test questions, one question is worth nearly two percentage
points. A difference of three points between two rungs could be luck. The
bootstrap estimates how much: resample the questions with replacement a couple
of thousand times, recompute the metric each time, and take the middle 95% of
the results. If two rungs' intervals overlap heavily, the honest statement is
"no measurable difference at this sample size".

## BM25

Keyword search with three ideas, each fixing a flaw in simply counting matching
words:

1. **Rare words matter more (IDF).** "the" appears everywhere and tells us
   nothing. "biometric" appears in few articles and tells us a lot. Each word's
   weight falls as the number of chunks containing it rises.
2. **Repetition has diminishing returns (`k1`).** A chunk mentioning
   "processing" ten times is not ten times as relevant as one mentioning it
   once. The contribution of each extra occurrence shrinks.
3. **Long chunks are not favoured (`b`).** A long article contains more words
   by chance. Matches are scaled down by the chunk's length relative to the
   average. `b = 0` switches this off; `b = 1` applies it fully.

The code is in `src/eurlex_rag/retrieve/bm25.py`, about a hundred lines, with
the formula in the docstring and a test that checks one score by hand.

It works through an **inverted index**: for each word, the list of chunks that
contain it. A query only touches the lists of its own words, so search time
depends on how common the query's words are, not on the size of the corpus.

## Likely interview questions

- *Why not fine-tune the model on the regulations instead?* Fine-tuning does
  not give citations, has to be redone when the text changes, and does not stop
  the model inventing article numbers. Retrieval gives traceable answers from
  the current text.
- *Why does BM25 fail on a Portuguese question?* It matches words. The
  Portuguese and English texts share almost none. This is a property of the
  method, and measuring it is the baseline.
- *Why keep BM25 at all then?* It is strong on exact terms and identifiers,
  costs nothing, and on English queries it is a real competitor. It also shows
  what the embedding model is adding.
- *Why score at article level when you index paragraphs?* A citation is to an
  article. Two paragraphs of the same article are one correct answer, not two.
- *How do you know the benchmark is any good?* The gold labels come from the
  alignment between languages, not from a model's opinion, and every test
  question is checked by hand in three languages.
- *Couldn't the questions leak the answer?* If a question copies the article's
  wording, keyword search looks better than it is. Questions are written from
  the non-English text and paraphrased, and a word-overlap check flags
  suspicious ones.
- *What does a 95% interval of [0.50, 0.73] tell you?* That with this many
  questions the true value is plausibly anywhere in that range, so only large
  differences between rungs mean something.
