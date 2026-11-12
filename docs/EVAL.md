# eurlex-rag — Evaluation design

Status: draft v1 (12 Nov 2026). Read [SPEC.md](SPEC.md) first.

The benchmark is built before the retrieval rungs, and no rung is merged
without a run on it.

## 1. What we measure

### Retrieval (rungs 1–5, no LLM cost)

Scored at article level against the gold article IDs.

| Metric | Why |
|---|---|
| Recall@1, @5, @10 | Did the right article reach the LLM at all? Recall@5 is the headline number. |
| MRR@10 | How high the first correct article ranks. |
| nDCG@10 | Ranking quality when several articles are relevant. |

### Answers (rungs 0, 6 and the final pipeline)

| Metric | How |
|---|---|
| Citation precision / recall | Cited article IDs vs gold IDs. Pure set arithmetic. |
| Invalid-citation rate | Citations to IDs that were not retrieved (or do not exist). For rung 0 this measures invented article numbers. |
| Faithfulness | Share of answer claims supported by the cited text, judged by an LLM (see §5). |
| Abstention accuracy | On unanswerable questions: did it decline? On answerable ones: did it wrongly decline? |
| Answer-language match | Is the answer in the language of the question? Automatic language ID. |

### Cost of each rung

p50 and p95 latency per stage, tokens in and out, and estimated cost per query.
A rung is judged on the metric it buys per millisecond and per cent.

## 2. Ground truth from parallel texts

Article numbers are the same in every language version of a regulation. So a
question written from the **Portuguese** text of AI Act Article 6 has the gold
label `AIA:art:6`, and the system, which only indexes English, must find it.
No human needs to label relevance; the alignment does it.

This is the only use of the non-English texts in evaluation. They are never in
the index.

## 3. The gold set

About 150 questions, each existing in three human-checkable languages
(EN, PT, FR), so about 450 queries.

| Tier | Share | Description | Gold |
|---|---|---|---|
| A — single article | ~60% | Answer sits in one article | 1 ID |
| B — multi-article | ~20% | Needs 2+ articles, or two regulations (e.g. GDPR and AI Act on automated decisions) | 2–4 IDs |
| C — unanswerable | ~20% | Plausible but outside the corpus (other EU laws, national law, invented obligations) | none; expected behaviour is abstention |

Coverage is stratified across the five regulations so no one of them dominates.

### How questions are made

1. An LLM drafts a question from the **PT or FR** text of a sampled article,
   under instructions to paraphrase and to ask what a practitioner would ask,
   not to quote.
2. The question is translated into the other two languages.
3. Automatic checks: lexical overlap between the EN question and the EN gold
   article (items above a threshold are rewritten or dropped); duplicate
   detection; tier-C items are checked to really have no supporting article.
4. Human verification (below).

The model that drafts questions is different from the model that answers them.

### Human verification — Miguel

- **Test split: every item.** For each, confirm in all three languages that the
  question is natural, means the same thing, and that the gold article does
  answer it. Fix or reject.
- **Dev split: a 25% sample**, to estimate its error rate.
- Rejections and edits are logged; the acceptance rate is reported in the
  README.

A small optional **silver set** (ES, DE, IT; machine-translated, unverified) is
reported separately and labelled as such. It supports "works beyond the three
verified languages" without overclaiming.

### Format

`data/gold/questions.jsonl`, one object per question:

```json
{
  "qid": "q0042",
  "tier": "A",
  "split": "test",
  "gold": ["AIA:art:6"],
  "source_lang": "pt",
  "text": {"en": "…", "pt": "…", "fr": "…"},
  "verified_by": "miguel",
  "notes": ""
}
```

## 4. Protocol

- **Split.** ~60% dev, ~40% test, stratified by tier and regulation, split by
  question (all language versions of a question stay on the same side).
- **Dev** is for choosing: chunk size, k, fusion weights, which reranker.
- **Test** is run **once per rung**, after its configuration is frozen. If a
  rung is changed after seeing test numbers, that is written down.
- **Language grid.** Every metric is reported per query language (EN, PT, FR),
  not just pooled. The gap between EN and non-EN queries *is* the cross-lingual
  result.
- **Uncertainty.** 95% bootstrap confidence intervals on every headline number;
  paired bootstrap between consecutive rungs. With ~60 test questions the
  intervals will be wide, and the write-up says so instead of reporting
  differences that are inside the noise.
- **Reproducibility.** Each run writes `results/<rung>-<date>.json` with the
  config, model IDs, corpus hashes, git commit and per-query outputs. Numbers
  in the README are generated from these files, never typed by hand.

### Baselines that must be fair

- BM25 on a Portuguese query against English text will score near zero. That
  is reported, and next to it BM25 on the machine-translated query, which is
  the honest lexical baseline for the cross-lingual case.
- Rung 5 (translate-then-search vs direct multilingual embedding) uses the same
  retriever on both sides, so the only variable is the query.

## 5. The LLM judge

Faithfulness needs a judge, and a judge needs checking.

1. Miguel hand-labels ~40 answers (claim supported / not supported).
2. The judge labels the same answers. Agreement (Cohen's kappa) is reported.
3. Judge numbers are only published if agreement is acceptable; otherwise
   faithfulness is reported on the hand-labelled sample alone.

The judge model differs from the answering model.

## 6. Regression gate in CI

- Every pull request: unit tests + a **retrieval smoke-eval** on ~30 dev
  queries. No LLM calls, runs in under a minute, fails the build if Recall@5
  drops below the committed threshold.
- Full evaluation (`make eval RUNG=…`) is run by hand at the end of each
  milestone, because it costs tokens.

## 7. The results table (target shape)

| Rung | R@5 EN | R@5 PT | R@5 FR | MRR@10 | Citation P / R | Faithful | p95 ms | Cost / query |
|---|---|---|---|---|---|---|---|---|
| 0 none | – | – | – | – | | | | |
| 1 BM25 | | | | | – | – | | |
| 1 BM25 + translated query | | | | | – | – | | |
| 2 dense | | | | | – | – | | |
| 3 hybrid | | | | | – | – | | |
| 4 + rerank | | | | | – | – | | |
| 5 + query transform | | | | | – | – | | |
| 6 agentic | | | | | | | | |

Empty until measured.
