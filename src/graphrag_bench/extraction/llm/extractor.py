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
    EntityProposal,
    ExtractionProvider,
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
    request_fingerprint,
)
from graphrag_bench.extraction.registry import EntityRegistry, stable_id
from graphrag_bench.models import Chunk, Document, EvidenceSpan, RelationAssertion, require_unique


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse(text: str) -> ProposedExtraction:
    def invalid_constant(value: str) -> None:
        raise ValueError(f"invalid JSON constant: {value}")

    return ProposedExtraction.model_validate(
        json.loads(text, object_pairs_hook=_unique_object, parse_constant=invalid_constant)
    )


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
        locations = _locations(chunk.text, entity.name)
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


def process_responses(
    corpus: CorpusIndex,
    records: tuple[ResponseRecord, ...],
    config: LLMExtractionConfig,
    spec: ModelSpec,
) -> LLMExtractionResult:
    """Replay validation without inference. Exact source matches do not establish entailment."""
    validate_provider_spec(config, spec)
    try:
        require_unique(tuple(r.request.chunk.chunk_id for r in records), "response chunk IDs")
    except ValueError as error:
        raise LLMError(str(error)) from error
    if {r.request.chunk.chunk_id for r in records} != corpus.chunks.keys():
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
        if record.completion.finish_reason == "length":
            issues.append(
                _issue("output_limit", "Output hit its token limit; chunk rejected.", chunk)
            )
            rejected.append(chunk.chunk_id)
            continue
        try:
            proposed = _parse(record.completion.text)
        except (ValueError, RecursionError) as error:
            issues.append(_issue("invalid_json", f"Response rejected: {str(error)[:240]}", chunk))
            rejected.append(chunk.chunk_id)
            continue
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
    records = tuple(
        get_completion(provider, make_request(chunk, config), response_cache)
        for chunk in corpus.chunks.values()
    )
    if provider.spec != spec:
        raise LLMError("provider specification changed during extraction")
    return process_responses(corpus, records, config, spec)
