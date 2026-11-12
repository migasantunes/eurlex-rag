# Rung 00 - No retrieval

Status: not started (milestone M5).

## What it is

The question goes straight to the LLM with no documents. The model answers from
whatever it absorbed in training and is asked to cite article IDs.

## Why it is in the ladder

It is the control. If the bare model already answers these questions correctly,
retrieval is not earning anything. The regulations are public and well known,
so the model will know a lot. What it usually cannot do is cite reliably: it
tends to produce article numbers that look right and are not.

## What gets measured

- Citation precision and recall against the gold articles.
- **Invalid-citation rate:** here, citations to articles that do not exist in
  the corpus. This is the number that shows what retrieval is for.
- Abstention on the unanswerable tier (does it admit it does not know?).
- Faithfulness cannot be scored in the usual way (there is no retrieved text to
  be faithful to). Report it as "not applicable", not as zero.

There are no retrieval metrics for this rung.

## Implementation notes

- Same provider interface, model and prompt wording as the full pipeline, minus
  the context block, so the only difference is the missing documents.
- Belongs to the answer evaluation, built in M5. Until then it is only a row in
  the plan.

## Acceptance

- Row in the results table with the answer columns filled and retrieval columns
  marked "–".

## Questions the defend note must answer

- Why bother with retrieval if the model has read the regulations?
- What is the difference between a wrong answer and an unsupported one?
- Why can't prompt instructions alone stop invented citations?
