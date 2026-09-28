# Data contracts

## Identity and serialization

Persisted project records carry `schema_version: "1.0"`. M8 and M9 raw model-proposal schemas deliberately omit that field; validated proposals are converted to versioned project records. Unknown fields and unsupported schema versions are rejected. IDs are stable, explicit strings, not Python hashes or runtime object addresses. IDs allow letters, digits, underscore, period, colon, and hyphen, with an alphanumeric first character.

Records use Pydantic's frozen assignment policy. Tuple collections prevent accidental list mutation, but dictionary fields are not deeply immutable. Revalidate records after transforming their serialized data. Do not use unchecked `model_construct` or `model_copy(update=...)` to ingest external data.

The document checksum is SHA-256 over the exact UTF-8 encoded `Document.text`. No trimming, Unicode normalization, or newline conversion occurs during validation. The fixture manifest separately hashes the exact bytes of six data files. These hashes detect accidental changes; they are not cryptographic signatures.

## Coordinates and provenance

`Chunk` and `EvidenceSpan` use half-open `[start, end)` Unicode character offsets into the stored document text. Offsets are not UTF-8 byte offsets or offsets into a PDF binary. M2 preserves text after optional UTF-8 BOM removal, records raw-file hashes and section ranges in `SourceRecord`, and keeps Markdown headings in the stored text. Original byte offsets can be recovered using the BOM size and UTF-8 encoded text prefixes. Page numbers remain unknown. A future PDF parser will need an explicit mapping to source pages and layout.

Span text must have exactly `end - start` characters. The fixture audit and M3's runtime `CorpusIndex` additionally require `document.text[start:end] == span.text`. Evidence with a `chunk_id` must refer to a chunk in the same document and fit completely inside that chunk. Chunk ordinals start at zero and must be contiguous within each document. Overlapping chunks are allowed. M3's graph builder requires source chunks for entity mentions as well as relation evidence; gold evidence can still omit chunks where permitted by its separate contracts.

Relation assertions require at least one evidence span, each with a source chunk. They retain subject, predicate, object, extraction method/version, and optional qualifiers. An assertion records a document's claim, not an automatically established truth. Multiple assertions with the same subject/predicate/object retain separate identities and provenance.

Entity and predicate labels are open strings in the reusable contracts. `ontology.json` supplies the fixture's permitted vocabulary. Entity aliases may overlap across different entities: the fixture deliberately gives both Birch and Elm the alias `Base`. The fixture loader does not merge them. M3's resolver likewise retains both candidates. Runtime extraction uses its own configurable syntax/type rules, not the fixture ontology or gold entity records.

M3 entity IDs fingerprint the exact type and canonical name normalized with NFC, case folding, and collapsed whitespace. Assertion IDs fingerprint the extraction/rules version, resolved triple, and source document/location/text. Assertion IDs exclude chunk IDs: one source statement appearing in multiple overlapping chunks remains one assertion with several evidence references. A second source location remains a distinct assertion. The graph keeps directed parallel edges keyed by assertion ID and rejects duplicate IDs and dangling endpoints. See the [M3 walkthrough](extraction-and-graph.md) for collision limitations and alias behavior.

## Gold evidence semantics

`BenchmarkQuestion.sufficient_evidence_sets` is a list of alternatives:

1. Any one complete evidence set is sufficient.
2. Every fact in that set is required.
3. Every span attached to a fact is required to support that fact.

Each fact includes a statement for later answer evaluation. Its ID must be unique within its evidence set. Repeated alternative spans must not inflate future coverage scores. Future metrics will evaluate unique facts and source-span coverage, not counts of overlapping retrieved chunks.

Gold evidence normally omits `chunk_id`. This keeps relevance labels independent of a particular chunking scheme. M1 introduced manually specified reference chunk boundaries. M2's sentence configuration reproduces all 25 reference spans; generated chunk IDs fingerprint content and configuration, while the fixture's manual IDs and `Overview` labels remain unchanged.

`relevant_document_ids` is exactly the union of documents across all gold alternatives. `required_document_count` is the minimum number of distinct documents among the annotated sufficient sets. This is an annotation claim, not a proof that no unannotated shortcut exists. `relevant_entity_ids` includes bridge and answer entities and is evaluation-only metadata.

`reasoning_hops` records sequential semantic links along the required reasoning chain, not the number of source documents or facts. Comparing two directly stated facts can therefore have two documents and one hop. A two-hop chain can appear in one document. Human review is required to assess hop labels; M1 validates their type and positivity rather than pretending to infer them.

Question types are `factual`, `relationship`, and `comparison`. Splits are `fixture`, `dev`, or `test`. Questions sharing a `group_id` cannot cross splits in a loaded dataset. Group assignment itself requires manual review; the validator cannot discover semantic near-duplicates. The tiny synthetic corpus uses only `fixture` and must not be reported as a held-out research benchmark.

## Results and reproducibility

Retrieval results contain an ordered tuple of unique chunk hits. Tuple position gives rank; scores need not share a scale between strategies. An empty tuple represents a retrieval miss. Scores and elapsed time must be finite; elapsed time cannot be negative.

M4 populates this contract for `vector` and `bm25`. Vector hits have cosine scores and no graph paths; BM25 hits have lexical scores and no paths. Both use descending score followed by ascending chunk ID for exact ties. Vector retrieval returns up to K nearest chunks even for unanswerable questions; BM25 returns only positive-score matches. Neither strategy generates an answer. CLI evidence records mirror hit order and retain original source spans.

Traversal paths retain entity IDs and assertion IDs. A path has one more entity than assertions. The record validates shape; M5's `KnowledgeGraph.validate_path` additionally checks existence, connectivity, and permitted traversal direction. The graph retriever verifies that each returned chunk supplies evidence for an edge on its path or a permitted terminal mention. Walking an incoming edge does not reverse the stored assertion.

Run manifests record a timezone-aware timestamp, seed, package/Python versions, optional Git commit, input checksums, configuration, artifact checksums, and provider versions. They establish the format for future experiment outputs; M2 does not execute retrieval experiments or populate `RunManifest`. Its separate `IngestionManifest` records deterministic source metadata, configuration, implementation versions, and output hashes without timestamps. Never store secrets in configuration or manifests.

M2 uses `ChunkingConfig` to validate positive character/unit limits and an overlap smaller than the unit limit. `Section` spans must partition the document when passed to the chunker. Every emitted chunk is a nonempty exact slice with a finite character bound; source text and all non-whitespace characters remain recoverable. See the [M2 walkthrough](ingestion.md) for algorithm details and limitations.

M3 adds `ExtractionRules`, `ExtractionIssue`, `ExtractionResult`, `GraphSnapshot`, and `GraphManifest`. The graph manifest stores input ingestion hashes, rules and their fingerprint, versions, counts, and graph/issue file hashes without timestamps. `load_ingestion` and `load_graph` verify hashes and cross-record provenance. Counts may be zero for an unsupported corpus; diagnostics explain skipped statements. These are construction artifacts, not retrieval experiments or populated `RunManifest` records.

M4 adds `EmbeddingConfig`, `EmbeddingSpec`, `VectorRows`, `VectorManifest`, and `BM25Config`. The vector manifest binds a normalized float32 matrix and sorted chunk IDs to the exact ingestion artifacts and embedding specification. Query providers must match the saved specification, including pinned revision, dimensions, text prefixes, execution settings, and relevant library versions. Numerical vectors must be finite and nonzero; persisted index rows must have unit length. M4 rejects oversized neural input rather than silently truncating source text. See the [M4 walkthrough](vector-retrieval.md) for storage, timing, and reproducibility boundaries.

M5 adds `GraphRetrievalConfig`, `QueryLink`, `GraphRank`, and `GraphSearchTrace`. A trace contains the standard `graph` result plus candidate groups, configuration, structural ranking details, path/candidate counts, and cap diagnostics. Graph scores encode final ordinal rank, not similarity. Surfaces with identical candidate sets share a seed group; ambiguous aliases preserve all candidates. Paths refer to the actual stored assertions even during reverse traversal. Audit metadata can contain sources outside the selected top K and must not silently become answer context. See the [M5 walkthrough](graph-retrieval.md).

M6 adds `HybridRetrievalConfig`, `FusionRank`, `FusionTrace`, and `HybridSearchTrace`. Hybrid results use strategy `hybrid` and RRF scores, not cosine similarities or probabilities. Each candidate records its one-based vector/graph ranks (null when absent), nonnegative contributions, and their sum. `ranking` covers the entire candidate union in fused order; `result.hits` applies the requested final `top_k`, which the hybrid trace records explicitly. The nested component results retain their candidate windows. `graph_trace` preserves the M5 configuration and cap diagnostics. The combined query time includes both branches, fusion, and selected-path validation; `fusion_ms` measures only the arithmetic/assembly step.

`HybridRetriever` requires exactly equal chunk records in both branches and a compatible embedding provider. The CLI additionally verifies both saved artifacts against the same ingestion hashes. Fusion deduplicates by chunk ID and preserves graph paths from candidates admitted by the graph branch. Unselected candidates, linked entities, and supporting assertion quotes remain audit metadata, not extra answer context. The pure `fuse_rrf` helper operates on already constructed result records; source validation belongs to the wrapper and artifact loaders. These diagnostic traces do not replace an experiment `RunManifest`. See the [M6 walkthrough](hybrid-retrieval.md).

M7 evaluates `BenchmarkQuestion` source coordinates independently of runtime chunk IDs and extracted entity IDs. All spans of a fact are required; all facts of one sufficient set are required; any complete sufficient set is accepted. `EvidenceSetScore` exposes supported and missing facts, while `EvidenceScore` records best-set coverage and complete evidence. `Fact Recall@K` uses the first K retrieved source chunks; context scores use only the source fragments actually selected under the budget.

`SelectedContext` records exact source pieces, rendered text, selected chunk IDs, overlap/budget skips, and the tokenizer count. `QuestionRun` records query/repeat identity, original retrieval and diagnostics, measured retrieval time, and cutoff evaluations. `BenchmarkSummary` includes descriptive aggregates, paired hybrid/vector differences, and a fingerprint excluding timing/diagnostics. Repeatability is null when only one repeat is available. Annotation group split checks precede filtering, and fixture labels are not promoted to held-out labels.

M7 now populates `RunManifest` with raw source/question input hashes, selected-source snapshots and output checksums, full comparison settings, extraction rules, model/tokenizer specification, implementation/library versions, code provenance, and timings. The manifest is written last. The verifier checks integrity and result fingerprints, not truth, independent annotation quality, or tokenizer equivalence to an answer model. See the [M7 walkthrough](benchmark.md).

M8 adds `LLMExtractionConfig`, `ProposedExtraction`, `ModelSpec`, `ExtractionRequest`, `Completion`, `ResponseRecord`, `LLMIssue`, and `LLMGraphManifest`. Responses use local entity IDs that are replaced by stable normalized-name/type IDs after validation. Quotes must match a unique contiguous interval of their chunk and contain both endpoint names. Invalid chunk-level output and rejected individual proposals are recorded separately. All retained assertions carry `review_status: unreviewed`; source checks establish coordinates, not semantic entailment.

The M8 manifest hashes the graph, issues, and complete inference receipts, and binds them to ingestion hashes, the extraction recipe, model revision, environment versions, and package source hash. The graph loader reconstructs the expected prompts and replays validation without loading a model. A cached response retains the original token usage and inference duration. Source/reference checks and replay consistency are not extraction accuracy metrics. See the [M8 walkthrough](llm-extraction.md).

M9 adds `GenerationConfig`, `AnswerRetrievalConfig`, `AnswerProposal`, `CitedClaim`, `Citation`, `Answer`, `TokenMeasurement`, `RetrievalReceipt`, `AnswerRequest`, `AnswerReceipt`, and `AnswerManifest`. It reuses the M8 local provider through the message-only `ChatRequest` protocol. Answer status distinguishes a structurally accepted proposal, insufficient evidence, and rejected output; every accepted claim remains `unreviewed`.

Citation labels such as `S1` address selected context pieces within one answer. They are not global IDs. Quotes resolve to unique intervals inside those pieces and then to `EvidenceSpan` document/chunk coordinates. A valid span elsewhere in the document is insufficient if it was not shown to the model. A single invalid citation rejects the whole answer; partial claims are not silently salvaged.

The answer manifest binds five artifacts to the original ingestion hashes and recorded provider/configuration. Replay uses the saved ranking and candidate token-count receipts to reconstruct selection and prompts, then rechecks citations and rendering. It does not rerun retrieval, tokenize independently, generate new text, or prove entailment. The nullable completion distinguishes deterministic empty-context abstention from a real model response. See the [M9 walkthrough](answer-generation.md).

## Fixture audit boundary

`load_fixture` checks manifest hashes, individual schemas, unique IDs, ontology membership, document/chunk references, exact source slices, entity mention surfaces, relation endpoints, benchmark evidence, and split-group consistency. Errors from JSONL parsing include filename and line number.

The audit validates consistency, not truth or entailment. It cannot prove that an assertion follows from its quote, that every relevant passage is annotated, or that a question genuinely requires two hops. Those remain annotation and evaluation responsibilities.
