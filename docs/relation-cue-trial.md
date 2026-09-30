# M11: reducing a report to relationship sentences

The first 3B trial found two of six draft facts. It often saw a long report chunk
with headings, tables, layout text, and several unrelated sentences. This trial
asks a narrower question: can the same reader extract a relationship after it sees
only sentences that contain explicit relationship language?

## The change

`relation-cues-v2` searches each source chunk for sentences containing words such
as `uses`, `based on`, `evaluated on`, `initializes from`, or `modification of`.
It sends at most the first two matching sentences, retaining their exact original
text and source coordinates. The model still chooses names, types, direction,
predicate, and sentence number. The program does not add a relationship because a
sentence matches a word pattern.

The older `relation-cues-v1` recipe remains unchanged. Version 2 adds only
initialization and modification phrases, which occur in two missed development
facts. It has its own request and recipe fingerprints, so outputs cannot be mixed
with the prior recipe.

For example, this report fragment:

```text
Background discussion. Aurora initializes from Quartz. More discussion.
```

becomes this model input:

```json
{"sentences":[{"id":1,"text":"Aurora initializes from Quartz."}]}
```

The model must still return the full relationship row. A wrong name, relation,
type, or sentence number fails the same validation as before.

## Why the full eight-case check cannot score this recipe

The original “generic hyperparameters” negative was selected because generic words
such as batch size and learning rate are not relationships. Its full source chunk
also contains a real statement that mE5 models initialize from named backbones.
The older cue list did not select that sentence; version 2 does. Counting a
source-supported initialization relationship as a false answer would be misleading.

The run therefore uses [a five-case positive slice](../datasets/examples/extraction-study/m11-cues-v2-positive-checks.json).
It copies the five positive cases and six draft facts from the original case file.
It does not measure negative-case behavior, precision, or abstention. Its labels
remain draft and are not held out.

This discovery is useful on its own: a label can be reasonable for one extraction
recipe yet incomplete for another because the source text sent to the model changes.
Before a final benchmark, every negative label needs review against the exact input
that recipe uses.

## Run and inspect

The candidate config keeps the cached Qwen2.5-3B-Instruct revision, CPU float32,
four threads, original examples, ontology, and 320-token limit. It changes only
sentence selection.

## Result: reject this candidate

The candidate ran the five-case positive slice and replayed successfully. It
matched **1/6** draft facts. The original full-context 3B recipe matched 2/6 over
the same five positive chunks. Narrowing input helped the RAGAS–GPT-3.5 case, but
lost the Jina relation and did not recover the RAG components or SBERT relation.

The model produced a well-formed but unmatched `BERT-FTbase USES E5` assertion.
The source says it uses the same fine-tuning data as E5, so this requires semantic
review rather than automatic classification as false. It still missed the draft
`BERT-FTbase BASED_ON BERTbase` relationship in the same sentence. The Jina output
mistook PDF figure layout text for the subject. The SBERT output used an incompatible
predicate. The RAG-components response again omitted sentence IDs and was rejected.

The result report SHA-256 is
`fb6c3bbb84350e34ed361979e2e6b11d2071dd2d57ad2ece098bd925cad8e705`;
its recipe SHA-256 is
`f6a846cc6bde07415f826f453ef9615f58723377d16cd2f544dd4e4b8f77e73b`.
The five calls took 4:21.52 wall time and peaked at 18,598,012 KiB RSS. These
single-run host measurements are descriptive, not performance benchmarks.

This recipe cannot be evaluated on the original three negative cases until their
labels are reviewed against the new selected sentences. It is not a candidate for
full-corpus extraction.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python scripts/check_extraction_sample.py \
  datasets/processed/m10-expanded-03/ingestion \
  --checks datasets/examples/extraction-study/m11-cues-v2-positive-checks.json \
  --config configs/llm-extraction-indexed-3b-cues-v2.toml \
  --min-available-gib 18 \
  --response-cache experiments/runs/m11-research-extraction-cache \
  --output experiments/runs/m11-indexed-3b-cues-v2-positive-01.json
```

Replay follows the normal source-check command. Compare this report only with
the five positive cases from the prior 3B report. Inspect each accepted assertion
against its original source sentence. A positive-only slice cannot establish that
the recipe avoids invented links across the corpus.

**Check your understanding:** Does finding a sentence with the word `uses` prove
that every name in the sentence has a relationship? No. The word only decides what
the model reads; source meaning and validation still decide what can be accepted.
