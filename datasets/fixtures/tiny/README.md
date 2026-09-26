# Tiny deterministic fixture

These eight original, fictional technical notes are development fixtures under the repository's MIT license. They make no claims about real papers or models. No external data, model, or API was used to obtain them.

## Files

- `corpus/documents.jsonl`: source inputs, including exact text and SHA-256 checksums.
- `gold/chunks.jsonl`: 25 manually specified sentence chunks, reproduced by M2's sentence configuration at the source-span level.
- `gold/entities.jsonl`: 15 reference entities, aliases, and exact source mentions.
- `gold/relations.jsonl`: 23 manually specified, attributed relation assertions.
- `gold/questions.jsonl`: 20 questions with answers and alternative sufficient evidence sets.
- `ontology.json`: fixture-specific entity and relation vocabulary.
- `manifest.json`: SHA-256 checksums of the six data files above.

All fixture IDs and offsets are stored explicitly. The files do not need runtime generation. Editing a source requires reviewing dependent annotations and updating the manifest hashes. Keep JSONL records on one physical line and use UTF-8 with LF newlines.

## Intended cases

| Questions | Purpose |
| --- | --- |
| q01–q08 | Direct facts and relationships within one document |
| q09–q16 | Two-hop joins across two documents, with bridge entities absent from the query |
| q17–q18 | Comparisons across documents without two sequential reasoning hops |
| q19 | A two-hop question with both one-document and two-document sufficient evidence sets |
| q20 | Alias resolution: QZ names Quartz |

Both Birch and Elm have the alias `Base`; their identities and mentions remain separate. Lumen and Quartz connect multiple documents and will exercise shared-neighbor traversal. Cedar's reported Accuracy metric appears in two documents, giving repeated assertions separate provenance. Dataset sizes remain textual facts without artificial numeric entity nodes.

Reference chunks and graph annotations are test expectations, not an implemented extraction pipeline. Numeric questions will also expose the need to retrieve entity mention evidence rather than only relation statements. No claim of real retrieval quality should be drawn from this corpus.

M2's reproduction test compares document IDs, ordinals, offsets, and exact text. The fixture's manual chunk IDs and `Overview` section labels are not inferred from the plain source text. Runtime chunks instead receive IDs derived from document identity, content, algorithm version, configuration, source bounds, and section labels.

Gold data is isolated from source data. Production extractors and retrievers must not read `gold/questions.jsonl` or its relevant-entity labels. Every question has `split: "fixture"`; these examples are not development/test partitions for the real experiment.
