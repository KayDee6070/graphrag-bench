# M11: teaching the reader with better examples

The 3B reader completed the first eight-case trial, but found only two of six
draft facts. A larger model helped it follow the output format; it did not reliably
find every relationship. This experiment asks whether more useful fictional
demonstrations help the same model read the same reports.

## What the missed facts revealed

The input did contain the missing evidence. In the RAGAS case, sentence 3 explicitly
names RAGAS and GPT-3.5 and says one uses the other. The model returned an empty list.
In the BERT initialization case, sentence 3 names the source and derived models.
Again, the model returned an empty list. Interpreting initialization as `BASED_ON`
remains a draft ontology decision awaiting independent review.

For the RAG-components case, the model proposed several relationships but omitted
the required sentence numbers. The parser correctly rejected the response. This
distinction matters: an empty response can be well formed and still miss facts;
a malformed response can contain recognizable names and still be unusable.

## One controlled change

[The candidate config](../configs/llm-extraction-indexed-3b-examples.toml) changes
only the examples compared with the original 3B config. The model revision, CPU
precision, four threads, prompt instructions, ontology, source chunks, scoring,
320-token output allowance, and validation rules stay fixed.

The original four short demonstrations become five fictional passages:

- One method uses two named components, with both rows pointing to sentence 2.
- A model is initialized from another model; unrelated shared-training text does
  not become a component relationship.
- A model is evaluated on a dataset after an unrelated sentence.
- A speed comparison and a negated evaluation produce no relationships.
- Generic hyperparameters and discussion produce no relationships.

Think of the scout learning to fill in a form. Previously, the practice report
usually had one sentence and one answer. Now a practice report has background text,
two possible answers, or a comparison that must be ignored. Every positive example
still includes the exact sentence number and original supporting text.

No real paper names, evaluation questions, or expected case answers were added to
the examples. However, the examples were designed **after inspecting the development
errors**. This is deliberate development adaptation, not an unseen test. It changes
several example properties together; any result cannot isolate which example helped.
The earlier outputs and labels remain intact for comparison.

Actual candidate prompt lengths are 988–1,085 tokens, below the 4,096-token input
limit. The existing cache separates requests when example messages change. No
model download, API service, new parser rule, or response repair is involved.

## Result: reject this candidate

The five-example candidate ran over the original eight cases and replayed
successfully. It still matched **2/6** draft facts, while schema-valid responses
fell from 7/8 to 6/8 and rejected responses rose from 1/8 to 2/8. Its mean recorded
completion time rose from 40.71 to 47.61 seconds. It did not improve extraction.

The RAGAS response copied the nearby ARES discussion and proposed an invalid
relationship instead of the explicit RAGAS–GPT-3.5 statement. The RAG-components
response still omitted sentence IDs. The Jina and SBERT relationships were the two
accepted matches. In the unsuitable-tasks negative, the model proposed an unsupported
relationship; validation removed it. This result is saved locally as
`m11-indexed-3b-examples-01.json` with recipe hash
`383bd8ad5be9b52a14442688a3c0be8b62f2ae2c853c5f6004be8d592148a66a`
and report hash
`82d66bc9d823c2fcbc7687d1480144645a4dbe2c9b52509b3b5a6fa651130275`.

More demonstrations increased prompt length but did not make the reader attend to
the right sentence. The original 3B indexed recipe remains the better observed
candidate. This test is not a held-out comparison and does not estimate corpus-wide
quality.

## Reproduce the comparison

Use the already cached 3B model and prepared paper ingestion. Choose a fresh output
path for a new run:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python scripts/check_extraction_sample.py \
  datasets/processed/m10-expanded-03/ingestion \
  --checks datasets/examples/extraction-study/m11-source-checks.json \
  --config configs/llm-extraction-indexed-3b-examples.toml \
  --min-available-gib 18 \
  --response-cache experiments/runs/m11-research-extraction-cache \
  --output experiments/runs/m11-indexed-3b-examples-01.json
```

Replay the resulting report without model loading:

```bash
.venv/bin/python scripts/check_extraction_sample.py \
  datasets/processed/m10-expanded-03/ingestion \
  --checks datasets/examples/extraction-study/m11-source-checks.json \
  --replay experiments/runs/m11-indexed-3b-examples-01.json
```

The original 3B report is
`experiments/runs/m11-indexed-3b-trial-01/diagnostic.json`. Its recipe and outputs
remain replayable. Compare draft facts matched, rejected responses, output limits,
negative cases, and every accepted assertion against its exact source sentence.
Repeated cache reads reproduce historical timings, not fresh speed measurements.

**Check your understanding:** If a practice example teaches two relationships and
the reader then invents a second relationship in every report, has it improved?
No. Recovering missing facts and avoiding unsupported facts are separate checks.
