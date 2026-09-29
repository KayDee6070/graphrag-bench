# M11: giving the scout numbered sentences

The extractor is the scout who draws our map. If it invents a connection or misses
a useful one, even a perfect graph search starts with a bad map. This experiment
tests whether a smaller output task helps the existing local reader. It is still
M11 feasibility work; the full graph and real-paper comparison remain unfinished.

## The change, with two reports

Imagine this numbered report:

1. The Helix method uses the Comet model.
2. The weather was cold.

Previously, the model needed to invent local entity IDs, list the entities, connect
the IDs, and copy a supporting quote into its response. The new optional
`indexed-v1` prompt asks for one record:

```json
{
  "relations": [{
    "subject": "Helix",
    "subject_type": "Method",
    "predicate": "USES",
    "object": "Comet",
    "object_type": "Model",
    "sentence_id": 1
  }]
}
```

The program retrieves the exact text of sentence 1. It creates the local IDs itself
and sends the result through the existing source and ontology checks. There is no
generated quote to repair. The model still chooses the names, types, relationship,
direction, and sentence; the program does not supply missing facts.

This format requests relationships only. It does not separately extract isolated
entities, so a name with no proposed relationship may disappear from the graph.
That is another coverage tradeoff to assess before adopting it.

The [next bounded trial](local-model-trial.md) keeps this final indexed recipe and
changes the local reader from 0.5B to 3B. Its config and memory preflight are ready;
the larger model has not been downloaded or evaluated.

The parser also accepts a six-value array in the order subject, subject type,
predicate, object, object type, sentence ID. This was the first prototype format.
The final prompt and examples ask for named fields because the paper trials showed
the model repeatedly producing objects when asked for arrays. Neither format permits
extra fields, missing values, unknown JSON constants, duplicate keys, or more than
four relation rows. A sentence ID must be a positive integer, not a string or `true`.

## What the checks can establish

Sentence numbers come from exact slices of the original chunk. The splitter handles
common abbreviations such as “et al.” but remains a heuristic. It does not repair
PDF layout, hyphenation, or broken names. The default indexed config includes all
sentences; it does not use the previous cue filter.

A valid relationship must reference an existing sentence containing both exact
endpoint names. Entity types and predicate direction must fit the configured
ontology. The evidence quote must still occur uniquely in the original chunk.
The accepted evidence retains its original document, page-aware chunk, and character
coordinates. A cross-sentence guess cannot silently become a supported statement.

An invalid row is diagnosed; valid sibling rows can survive. Malformed JSON/schema
or an output-limit response rejects the entire response. Raw model outputs remain
unchanged in receipts. Profiles use different request/recipe fingerprints, while
the original M8 recipe continues to replay unchanged.

**These checks cannot establish understanding.** A model can point at a real sentence
but choose the wrong predicate or misread a negation. Such an assertion can pass
mechanical validation and still be wrong. Every accepted assertion remains unreviewed
until its meaning is checked. Numbered evidence reduces copying work; it does not
make the language model trustworthy.

## The small exam

We froze eight original paper chunks: five positive cases with six draft facts,
plus three negative cases. They include RAGAS using GPT-3.5, RAG using DPR and BART,
model ancestry, generic hyperparameters, a comparison, and an unsuitable-task example.
Both the previous focused prompt and the new indexed prompt receive the same full
chunks, cached model, eight-predicate ontology, and 320-token output allowance.

The draft facts were checked against source text. Independent review is still
pending. The selection is deliberately small and targeted; one negative
is an already known failure. It is not a held-out benchmark. Reference facts are
used for scoring only and never passed to the model. The baseline and candidate
change multiple prompt details, so their difference does not isolate one cause.

The [case file](../datasets/examples/extraction-study/m11-source-checks.json) explains
each choice. The [development report](../reports/m11-development.md) records actual
outputs, rejected proposals, and what can be concluded. A matched draft fact means
its names, direction, and predicate match an allowed reference spelling. Exact type
accuracy and full semantic correctness still require separate review.

## Reproduce and inspect

Use the existing local paper ingestion and model cache. Choose fresh output paths:

```bash
.venv/bin/python scripts/check_extraction_sample.py \
  datasets/processed/m10-expanded-03/ingestion \
  --checks datasets/examples/extraction-study/m11-source-checks.json \
  --config configs/llm-extraction-indexed.toml \
  --response-cache experiments/runs/m11-research-extraction-cache \
  --output experiments/runs/my-indexed-checks.json
```

Repeat with `configs/llm-extraction-focused.toml` and a different output file for
the baseline. Each run saves source hashes, the case-file hash, recipe, provider
settings, raw completions, accepted assertions/evidence, issues, and draft scores.
It creates no full paper graph and makes no full-corpus runtime projection. Cached
completion timings retain their historical values.

To check a saved diagnostic without inference:

```bash
.venv/bin/python scripts/check_extraction_sample.py \
  datasets/processed/m10-expanded-03/ingestion \
  --checks datasets/examples/extraction-study/m11-source-checks.json \
  --replay experiments/runs/my-indexed-checks.json
```

Replay rebuilds the source requests and recalculates validation and scores. It
rejects changed source, labels, requests, or inconsistent saved results. It does
not independently authenticate where a model response came from or prove a label true.

For the three fictional positive/negative controls, `build-llm-graph` also accepts
the indexed config. Their graph is a software/model control, not a graph of the
30-paper corpus.

**Check your understanding:** If the scout points at the correct report but writes
“Eren guards Lantern Squad” instead of “Eren belongs to Lantern Squad,” is the map
correct? No. A real citation and a correct relationship are separate requirements.
