# Rung 06 - Agentic multi-hop

Status: not started (milestone M6).

## What it is

Instead of one search followed by one answer, an LLM-driven loop: split the
question into parts, search for each, read what came back, follow references
to other articles, and stop when it has enough.

## Why

Some questions cannot be answered from one retrieval pass:

- They span regulations: "does an AI system that profiles users fall under
  both the AI Act and GDPR's rules on automated decisions?"
- The answer sits behind a cross-reference: an article says "the systems
  referred to in Annex III", and the annex was not retrieved.

These are tier B of the benchmark. Tier A questions should not need any of
this, which is why both tiers are measured.

## Theory

An **agent** here means a program in which the LLM chooses the next step from
a fixed set of tools, inside a loop with a stopping rule. The tools are small:

| Tool | Does |
|---|---|
| `search(query)` | The best retrieval pipeline from rungs 01-05 |
| `get(article_id)` | Fetch one article by ID, for following a cross-reference |
| `finish(answer)` | Stop and answer |

LangGraph models this as a graph: nodes are steps (plan, search, read, answer),
edges say what can follow what, and a shared state object carries the question,
what was retrieved, and how many steps were taken. The graph makes the control
flow explicit and inspectable, where a hand-written loop would hide it in
if-statements.

The price is paid on every question: more LLM calls, more latency, more
tokens, and more ways to go wrong (looping, drifting off topic, retrieving so
much that the answer gets worse).

## Implementation notes

- LangGraph is used in `src/eurlex_rag/agent/` and nowhere else.
- Hard limits: maximum steps, maximum retrieved articles, a token budget. Hitting a
  limit ends with the best answer so far, flagged.
- Cross-references can be found without an LLM: the text says "Article 6",
  "Annex III", "Regulation (EU) 2016/679". A regular expression over the
  English text gives a reference graph for free. Evaluate that cheap version
  (retrieve, then add referenced articles) as its own row before the full
  agent: it may get most of the gain at none of the cost.
- Record the full trace of each run (steps, tool calls, tokens) in the result
  file. The trace is what makes failures explainable.
- Citation validation still applies: a cited article must have been retrieved
  or fetched during the run.

## Acceptance

- Tier B: citation recall and faithfulness against the best single-pass
  pipeline.
- Tier A: the same metrics, to show what the agent costs when it is not needed.
- Cost and latency per query next to every gain.
- The reference-expansion row.
- A recommendation: always, never, or route by question type.

## Questions the defend note must answer

- What makes this an agent and not a pipeline?
- What does the graph look like, and what is in the state?
- How does it know when to stop? What happens when it does not?
- How much of the gain came from simply following cross-references?
- What does it cost per question, and would you ship it?
