# Data contracts

## Identity and serialization

All records carry `schema_version: "1.0"`. Unknown fields and unsupported schema versions are rejected. IDs are stable, explicit strings, not Python hashes or runtime object addresses. IDs allow letters, digits, underscore, period, colon, and hyphen, with an alphanumeric first character.

Records use Pydantic's frozen assignment policy. Tuple collections prevent accidental list mutation, but dictionary fields are not deeply immutable. Revalidate records after transforming their serialized data. Do not use unchecked `model_construct` or `model_copy(update=...)` to ingest external data.

The document checksum is SHA-256 over the exact UTF-8 encoded `Document.text`. No trimming, Unicode normalization, or newline conversion occurs during validation. The fixture manifest separately hashes the exact bytes of six data files. These hashes detect accidental changes; they are not cryptographic signatures.

## Coordinates and provenance

`Chunk` and `EvidenceSpan` use half-open `[start, end)` Unicode character offsets into the stored document text. Offsets are not UTF-8 byte offsets or offsets into a PDF binary. A future parser must preserve a mapping from this stored text to source pages and sections where available.

Span text must have exactly `end - start` characters. The fixture audit additionally requires `document.text[start:end] == span.text`. Evidence with a `chunk_id` must refer to a chunk in the same document and fit completely inside that chunk. Chunk ordinals start at zero and must be unique within each document. Overlapping chunks are allowed.

Relation assertions require at least one evidence span, each with a source chunk. They retain subject, predicate, object, extraction method/version, and optional qualifiers. An assertion records a document's claim, not an automatically established truth. Multiple assertions with the same subject/predicate/object retain separate identities and provenance.

Entity and predicate labels are open strings in the reusable contracts. `ontology.json` supplies the fixture's permitted vocabulary. Entity aliases may overlap across different entities: the fixture deliberately gives both Birch and Elm the alias `Base`. The loader does not merge them. Future resolution must handle this ambiguity explicitly.

## Gold evidence semantics

`BenchmarkQuestion.sufficient_evidence_sets` is a list of alternatives:

1. Any one complete evidence set is sufficient.
2. Every fact in that set is required.
3. Every span attached to a fact is required to support that fact.

Each fact includes a statement for later answer evaluation. Its ID must be unique within its evidence set. Repeated alternative spans must not inflate future coverage scores. Future metrics will evaluate unique facts and source-span coverage, not counts of overlapping retrieved chunks.

Gold evidence normally omits `chunk_id`. This keeps relevance labels independent of a particular chunking scheme. The fixture's reference chunk boundaries are manually specified expected outputs for future ingestion tests; no chunker is implemented in M1.

`relevant_document_ids` is exactly the union of documents across all gold alternatives. `required_document_count` is the minimum number of distinct documents among the annotated sufficient sets. This is an annotation claim, not a proof that no unannotated shortcut exists. `relevant_entity_ids` includes bridge and answer entities and is evaluation-only metadata.

`reasoning_hops` records sequential semantic links along the required reasoning chain, not the number of source documents or facts. Comparing two directly stated facts can therefore have two documents and one hop. A two-hop chain can appear in one document. Human review is required to assess hop labels; M1 validates their type and positivity rather than pretending to infer them.

Question types are `factual`, `relationship`, and `comparison`. Splits are `fixture`, `dev`, or `test`. Questions sharing a `group_id` cannot cross splits in a loaded dataset. Group assignment itself requires manual review; the validator cannot discover semantic near-duplicates. The tiny synthetic corpus uses only `fixture` and must not be reported as a held-out research benchmark.

## Results and reproducibility

Retrieval results contain an ordered tuple of unique chunk hits. Tuple position gives rank; scores need not share a scale between strategies. An empty tuple represents a retrieval miss. Scores and elapsed time must be finite; elapsed time cannot be negative.

Traversal paths retain entity IDs and assertion IDs. A path has one more entity than assertions. M1 validates path shape only; checking graph connectivity and traversal direction belongs to the graph implementation.

Run manifests record a timezone-aware timestamp, seed, package/Python versions, optional Git commit, input checksums, configuration, artifact checksums, and provider versions. They establish the format for future experiment outputs; M1 does not execute experiments or populate manifests. Never store secrets in configuration or manifests.

## Fixture audit boundary

`load_fixture` checks manifest hashes, individual schemas, unique IDs, ontology membership, document/chunk references, exact source slices, entity mention surfaces, relation endpoints, benchmark evidence, and split-group consistency. Errors from JSONL parsing include filename and line number.

The audit validates consistency, not truth or entailment. It cannot prove that an assertion follows from its quote, that every relevant passage is annotated, or that a question genuinely requires two hops. Those remain annotation and evaluation responsibilities.
