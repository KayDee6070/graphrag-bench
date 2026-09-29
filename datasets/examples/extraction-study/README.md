# M11 extraction development checks

`m11-source-checks.json` pins eight original chunks from the 30-paper ingestion.
It contains five positive cases with six draft reference relationships and three
negative cases. The positive cases cover component use and model ancestry; they
do not cover every predicate. The selection is purposive and small. The batch-size
negative was already inspected in an earlier failed extraction trial.

These draft labels were checked against the source text. They are **not independently
reviewed, representative, or held out**. Each
case explains its interpretation. In particular, interpreting model initialization
as `BASED_ON` should receive a second review. Name alternatives in a reference fact
are scoring alternatives only; they never enter the extraction prompt or registry.

The file records hashes of the complete original source artifacts and stable chunk
IDs. It does not change text, chunk boundaries, or offsets. Source PDFs and processed
ingestion remain local under the paths recorded in the M11 report. Attribution and
pinned paper metadata are in `datasets/papers/research`.

Run `scripts/check_extraction_sample.py` against the original ingestion and this
file to compare extraction profiles on the same inputs. The model receives only
the source chunk, ontology, and independently authored fictional prompt examples.
The scoring code sees the draft labels afterward. `--replay` recomputes requests,
source validation, and draft scores from saved completions without loading a model.
Replay checks consistency; it cannot prove a completion was historically generated
by the claimed model or validate the semantic correctness of a reference fact.

The script counts draft facts matched by names, direction, and predicate. It does
not grade exact entity types, establish precision, or label every unmatched assertion
false. The full accepted assertions and their source evidence are saved for review.
Its negative-case count requires zero accepted assertions and no whole-response
rejection; inspect individual issues before interpreting that as correct abstention.

See [the numbered-sentence lesson](../../../docs/indexed-extraction.md) for commands
and [the M11 development report](../../../reports/m11-development.md) for observations.
