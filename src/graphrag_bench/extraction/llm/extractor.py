"""Align model proposals to source coordinates, then construct auditable assertions."""

import json
import re
from collections.abc import Iterable
from pathlib import Path

from graphrag_bench.corpus import CorpusIndex
from graphrag_bench.extraction.llm.cache import get_completion
from graphrag_bench.extraction.llm.config import LLMError, LLMExtractionConfig
from graphrag_bench.extraction.llm.contracts import (
    EXTRACTOR_VERSION,
    Completion,
    EntityProposal,
    ExtractionProvider,
    IndexedExtraction,
    IndexedRelation,
    LLMExtractionResult,
    LLMIssue,
    ModelSpec,
    ProposedExtraction,
    RelationProposal,
    ResponseRecord,
)
from graphrag_bench.extraction.llm.prompt import (
    extraction_fingerprint,
    make_request,
    numbered_spans,
    request_fingerprint,
    selected_passage,
    selected_ranges,
)
from graphrag_bench.extraction.registry import EntityRegistry, stable_id
from graphrag_bench.models import Chunk, Document, EvidenceSpan, RelationAssertion, require_unique

ABSTENTION_COMPLETION = Completion(
    text='{"entities":[],"relations":[]}',
    input_tokens=0,
    output_tokens=0,
    finish_reason="eos",
    elapsed_ms=0,
)


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse(text: str, *, style: str = "standard") -> ProposedExtraction | IndexedExtraction:
    def invalid_constant(value: str) -> None:
        raise ValueError(f"invalid JSON constant: {value}")

    schema = IndexedExtraction if style == "indexed-v1" else ProposedExtraction
    return schema.model_validate(
        json.loads(text, object_pairs_hook=_unique_object, parse_constant=invalid_constant)
    )


def _expand_indexed(
    proposed: IndexedExtraction, chunk: Chunk, config: LLMExtractionConfig, issues: list[LLMIssue]
) -> ProposedExtraction:
    spans = numbered_spans(chunk.text, config)
    entities: dict[tuple[str, str], EntityProposal] = {}
    relations = []
    for row in proposed.relations:
        if isinstance(row, IndexedRelation):
            subject, subject_type, predicate, obj, object_type, sentence_id = (
                row.subject,
                row.subject_type,
                row.predicate,
                row.object,
                row.object_type,
                row.sentence_id,
            )
        else:
            subject, subject_type, predicate, obj, object_type, sentence_id = row
        if sentence_id > len(spans):
            issues.append(_issue("invalid_quote", "Unknown numbered source sentence.", chunk))
            continue
        start, end = spans[sentence_id - 1]
        quote = chunk.text[start:end]
        subject_surface = subject.strip() if config.strip_entity_whitespace else subject
        object_surface = obj.strip() if config.strip_entity_whitespace else obj
        if not _locations(quote, subject_surface) or not _locations(quote, object_surface):
            issues.append(
                _issue("invalid_quote", "Selected sentence lacks exact endpoint names.", chunk)
            )
            continue
        ids = []
        for name, entity_type in ((subject, subject_type), (obj, object_type)):
            key = (name, entity_type)
            if key not in entities:
                entities[key] = EntityProposal(
                    id=f"e{len(entities) + 1}", name=name, entity_type=entity_type
                )
            ids.append(entities[key].id)
        relations.append(
            RelationProposal(subject=ids[0], predicate=predicate, object=ids[1], quote=quote)
        )
    return ProposedExtraction(entities=tuple(entities.values()), relations=tuple(relations))


def _locations(text: str, surface: str) -> tuple[tuple[int, int], ...]:
    """Exact spelling, with word boundaries to avoid Elm matching Helmet."""
    left = r"(?<!\w)" if surface[0].isalnum() or surface[0] == "_" else ""
    right = r"(?!\w)" if surface[-1].isalnum() or surface[-1] == "_" else ""
    return tuple(m.span() for m in re.finditer(left + re.escape(surface) + right, text))


def _issue(code: str, message: str, chunk: Chunk) -> LLMIssue:
    return LLMIssue(
        code=code,
        message=message,
        span=EvidenceSpan(
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            start=chunk.start,
            end=chunk.end,
            text=chunk.text,
        ),
    )


def _entities(
    proposals: tuple[EntityProposal, ...],
    chunk: Chunk,
    config: LLMExtractionConfig,
    registry: EntityRegistry,
    corpus: CorpusIndex,
    issues: list[LLMIssue],
) -> dict[str, tuple[str, EntityProposal]]:
    accepted = {}
    for entity in proposals:
        if config.strip_entity_whitespace and entity.name != entity.name.strip():
            entity = entity.model_copy(update={"name": entity.name.strip()})
            issues.append(
                _issue("normalized_entity", f"{entity.id}: trimmed name whitespace.", chunk)
            )
        locations = tuple(
            (start, end)
            for start, end in _locations(chunk.text, entity.name)
            if any(a <= start and end <= b for a, b in selected_ranges(chunk.text, config))
        )
        if (
            entity.entity_type not in config.entity_types
            or entity.name != entity.name.strip()
            or not locations
        ):
            issues.append(
                _issue(
                    "invalid_entity",
                    f"{entity.id}: unknown type or name absent from source.",
                    chunk,
                )
            )
            continue
        identifier = registry.declare(entity.name, entity.entity_type)
        for start, end in locations:
            registry.mentions[identifier].update(
                corpus.evidence(chunk.document_id, chunk.start + start, chunk.start + end)
            )
        accepted[entity.id] = (identifier, entity)
    return accepted


def _relation(
    proposal: RelationProposal,
    chunk: Chunk,
    entities: dict[str, tuple[str, EntityProposal]],
    config: LLMExtractionConfig,
    corpus: CorpusIndex,
    version: str,
) -> RelationAssertion:
    if proposal.subject not in entities or proposal.object not in entities:
        raise ValueError("relation references an unknown or rejected local entity")
    subject_id, subject = entities[proposal.subject]
    object_id, object_entity = entities[proposal.object]
    allowed = next((r for r in config.relations if r.predicate == proposal.predicate), None)
    if (
        allowed is None
        or subject.entity_type not in allowed.subject_types
        or object_entity.entity_type not in allowed.object_types
        or subject_id == object_id
    ):
        raise ValueError("unknown predicate, incompatible endpoint types, or self-relation")
    start = chunk.text.find(proposal.quote)
    if start < 0 or chunk.text.find(proposal.quote, start + 1) >= 0:
        raise LLMError("quote must occur exactly once in the source chunk")
    if not any(
        a <= start and start + len(proposal.quote) <= b
        for a, b in selected_ranges(chunk.text, config)
    ):
        raise LLMError("quote must be contained in one selected source span")
    if not _locations(proposal.quote, subject.name) or not _locations(
        proposal.quote, object_entity.name
    ):
        raise LLMError("quote must contain both exact endpoint names")
    start += chunk.start
    end = start + len(proposal.quote)
    return RelationAssertion(
        assertion_id=stable_id(
            "assertion",
            version,
            chunk.document_id,
            str(start),
            str(end),
            subject_id,
            proposal.predicate,
            object_id,
        ),
        subject_id=subject_id,
        predicate=proposal.predicate,
        object_id=object_id,
        evidence=corpus.evidence(chunk.document_id, start, end),
        extraction_method="llm-assisted",
        extraction_version=version,
        qualifiers={
            "review_status": "unreviewed",
            "source_check": "exact-quote-and-endpoint-presence",
        },
    )


def validate_provider_spec(config: LLMExtractionConfig, spec: ModelSpec) -> None:
    if (spec.model_id, spec.revision) != (config.model.model_id, config.model.revision):
        raise LLMError("provider model differs from extraction configuration")
    if any(spec.settings.get(k) != v for k, v in config.model.model_dump(mode="json").items()):
        raise LLMError("provider settings differ from extraction configuration")


def complete_chunk(
    chunk: Chunk,
    config: LLMExtractionConfig,
    provider: ExtractionProvider,
    response_cache: Path | None = None,
) -> ResponseRecord:
    """Complete one source chunk, or record a deterministic cue-filter abstention."""
    spec = provider.spec
    validate_provider_spec(config, spec)
    request = make_request(chunk, config)
    if config.passage_selection != "full" and not selected_passage(chunk.text, config):
        return ResponseRecord(
            request_sha256=request_fingerprint(spec, request),
            request=request,
            completion=ABSTENTION_COMPLETION,
            origin="deterministic-abstention",
        )
    return get_completion(provider, request, response_cache)


def process_responses(
    corpus: CorpusIndex,
    records: tuple[ResponseRecord, ...],
    config: LLMExtractionConfig,
    spec: ModelSpec,
) -> LLMExtractionResult:
    """Replay validation without inference. Exact source matches do not establish entailment."""
    return _process_responses(corpus, records, config, spec, complete=True)


def validate_sample_responses(
    corpus: CorpusIndex,
    records: tuple[ResponseRecord, ...],
    config: LLMExtractionConfig,
    spec: ModelSpec,
) -> LLMExtractionResult:
    """Diagnose source-only samples; never substitute this for complete graph validation.

    The full corpus preserves original ordinals, IDs, and source coordinates.
    Graph artifact loading still requires one response for every source chunk.
    """
    return _process_responses(corpus, records, config, spec, complete=False)


def _process_responses(
    corpus: CorpusIndex,
    records: tuple[ResponseRecord, ...],
    config: LLMExtractionConfig,
    spec: ModelSpec,
    *,
    complete: bool,
) -> LLMExtractionResult:
    validate_provider_spec(config, spec)
    try:
        require_unique(tuple(r.request.chunk.chunk_id for r in records), "response chunk IDs")
    except ValueError as error:
        raise LLMError(str(error)) from error
    received = {r.request.chunk.chunk_id for r in records}
    if not received <= corpus.chunks.keys():
        raise LLMError("response references an unknown source chunk")
    if complete and received != corpus.chunks.keys():
        raise LLMError("responses must cover every source chunk exactly once")
    registry = EntityRegistry()
    relations: dict[str, RelationAssertion] = {}
    issues: list[LLMIssue] = []
    rejected = []
    version = f"{EXTRACTOR_VERSION}:{extraction_fingerprint(config, spec)}"
    records = tuple(sorted(records, key=lambda r: r.request.chunk.chunk_id))
    for record in records:
        chunk = corpus.chunks[record.request.chunk.chunk_id]
        if record.request != make_request(chunk, config):
            raise LLMError("saved request differs from source or extraction prompt")
        if record.request_sha256 != request_fingerprint(spec, record.request):
            raise LLMError("saved request fingerprint mismatch")
        abstains = config.passage_selection != "full" and not selected_ranges(chunk.text, config)
        if (record.origin == "deterministic-abstention") != abstains:
            raise LLMError("response origin differs from passage selection policy")
        if abstains and record.completion != ABSTENTION_COMPLETION:
            raise LLMError("deterministic abstention must have the exact empty, zero-cost receipt")
        if abstains:
            continue
        if record.completion.finish_reason == "length":
            issues.append(
                _issue("output_limit", "Output hit its token limit; chunk rejected.", chunk)
            )
            rejected.append(chunk.chunk_id)
            continue
        try:
            proposed = _parse(record.completion.text, style=config.prompt_style)
        except (ValueError, RecursionError) as error:
            issues.append(_issue("invalid_json", f"Response rejected: {str(error)[:240]}", chunk))
            rejected.append(chunk.chunk_id)
            continue
        if isinstance(proposed, IndexedExtraction):
            proposed = _expand_indexed(proposed, chunk, config, issues)
        entities = _entities(proposed.entities, chunk, config, registry, corpus, issues)
        for relation in proposed.relations:
            try:
                assertion = _relation(relation, chunk, entities, config, corpus, version)
                relations[assertion.assertion_id] = assertion
            except LLMError as error:
                issues.append(_issue("invalid_quote", str(error), chunk))
            except ValueError as error:
                issues.append(_issue("invalid_relation", str(error), chunk))
    return LLMExtractionResult(
        entities=registry.entities(),
        relations=tuple(relations[key] for key in sorted(relations)),
        issues=tuple(issues),
        rejected_chunks=tuple(rejected),
        records=records,
    )


def extract_with_llm(
    documents: Iterable[Document],
    chunks: Iterable[Chunk],
    config: LLMExtractionConfig,
    provider: ExtractionProvider,
    *,
    response_cache: Path | None = None,
) -> LLMExtractionResult:
    """Source records and ontology only; no benchmark questions, answers, or gold graph."""
    corpus = CorpusIndex(documents, chunks)
    spec = provider.spec
    # Check provider/config agreement before the first potentially expensive inference.
    validate_provider_spec(config, spec)
    records = []
    for chunk in corpus.chunks.values():
        records.append(complete_chunk(chunk, config, provider, response_cache))
    records = tuple(records)
    if provider.spec != spec:
        raise LLMError("provider specification changed during extraction")
    return process_responses(corpus, records, config, spec)
