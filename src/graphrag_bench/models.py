"""Versioned data contracts; no extraction, retrieval, or provider dependencies."""

from __future__ import annotations

from hashlib import sha256
from typing import Annotated, Literal, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    model_validator,
)

Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")]
Text = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
NonNegativeInt = Annotated[int, Field(ge=0, strict=True)]
PositiveInt = Annotated[int, Field(gt=0, strict=True)]


def text_sha256(text: str) -> str:
    """Hash the exact UTF-8 text, including whitespace and newlines."""
    return sha256(text.encode("utf-8")).hexdigest()


def require_unique(values: tuple[str, ...], label: str) -> None:
    """Reject ambiguous identifiers rather than silently overwriting records."""
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {label}")


class Record(BaseModel):
    """Reject unknown fields, non-finite numbers, and invalid field assignment."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    schema_version: Literal["1.0"] = "1.0"


class Document(Record):
    document_id: Identifier
    title: Text
    text: Text
    content_sha256: Sha256
    source_uri: Text

    @model_validator(mode="after")
    def verify_checksum(self) -> Self:
        if self.content_sha256 != text_sha256(self.text):
            raise ValueError("content_sha256 does not match document text")
        return self


class TextSpan(Record):
    """Half-open Unicode character offsets into the stored document text."""

    document_id: Identifier
    start: NonNegativeInt
    end: PositiveInt
    text: Text

    @model_validator(mode="after")
    def verify_length(self) -> Self:
        if self.end <= self.start or self.end - self.start != len(self.text):
            raise ValueError("span offsets must describe exactly len(text) characters")
        return self


class Chunk(TextSpan):
    chunk_id: Identifier
    ordinal: NonNegativeInt
    section: Text | None = None
    page: PositiveInt | None = None


class EvidenceSpan(TextSpan):
    # Gold evidence can omit chunk_id so it survives changes in chunking.
    chunk_id: Identifier | None = None


class Entity(Record):
    entity_id: Identifier
    name: Text
    entity_type: Identifier
    aliases: tuple[Text, ...] = ()
    mentions: tuple[EvidenceSpan, ...] = ()

    @model_validator(mode="after")
    def verify_aliases(self) -> Self:
        require_unique(tuple(alias.casefold() for alias in self.aliases), "entity aliases")
        if self.name.casefold() in {alias.casefold() for alias in self.aliases}:
            raise ValueError("aliases must not repeat the canonical name")
        return self


class RelationAssertion(Record):
    assertion_id: Identifier
    subject_id: Identifier
    predicate: Identifier
    object_id: Identifier
    evidence: tuple[EvidenceSpan, ...] = Field(min_length=1)
    extraction_method: Text
    extraction_version: Text
    qualifiers: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_source_chunks(self) -> Self:
        if any(span.chunk_id is None for span in self.evidence):
            raise ValueError("relation evidence requires chunk_id")
        return self


class EvidenceFact(Record):
    """An atomic reference fact; every supporting span is required."""

    fact_id: Identifier
    statement: Text
    spans: tuple[EvidenceSpan, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def verify_spans(self) -> Self:
        locations = tuple(f"{s.document_id}:{s.start}:{s.end}" for s in self.spans)
        require_unique(locations, "supporting spans")
        return self


class EvidenceSet(Record):
    """All facts in this set are necessary; one complete set is sufficient."""

    facts: tuple[EvidenceFact, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def verify_facts(self) -> Self:
        require_unique(tuple(fact.fact_id for fact in self.facts), "fact IDs")
        return self


class BenchmarkQuestion(Record):
    question_id: Identifier
    question: Text
    question_type: Literal["factual", "relationship", "comparison"]
    expected_answer: Text
    sufficient_evidence_sets: tuple[EvidenceSet, ...] = Field(min_length=1)
    relevant_document_ids: tuple[Identifier, ...] = Field(min_length=1)
    relevant_entity_ids: tuple[Identifier, ...] = ()
    required_document_count: PositiveInt
    reasoning_hops: PositiveInt
    difficulty: Literal["easy", "medium", "hard"]
    split: Literal["fixture", "dev", "test"]
    group_id: Identifier

    @model_validator(mode="after")
    def verify_labels(self) -> Self:
        require_unique(self.relevant_document_ids, "relevant document IDs")
        require_unique(self.relevant_entity_ids, "relevant entity IDs")
        document_sets = [
            {span.document_id for fact in evidence.facts for span in fact.spans}
            for evidence in self.sufficient_evidence_sets
        ]
        if set.union(*document_sets) != set(self.relevant_document_ids):
            raise ValueError(
                "relevant_document_ids must equal the union of gold evidence documents"
            )
        if min(map(len, document_sets)) != self.required_document_count:
            raise ValueError("required_document_count must equal the minimum across evidence sets")
        return self


class TraversalPath(Record):
    entity_ids: tuple[Identifier, ...] = Field(min_length=1)
    assertion_ids: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def verify_path_length(self) -> Self:
        if len(self.entity_ids) != len(self.assertion_ids) + 1:
            raise ValueError("a path requires one more entity than assertions")
        return self


class RetrievalHit(Record):
    chunk_id: Identifier
    score: float = Field(strict=True)
    paths: tuple[TraversalPath, ...] = ()


class RetrievalResult(Record):
    query: Text
    strategy: Identifier
    hits: tuple[RetrievalHit, ...] = ()
    elapsed_ms: float = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def verify_hits(self) -> Self:
        require_unique(tuple(hit.chunk_id for hit in self.hits), "retrieved chunk IDs")
        return self


class RunManifest(Record):
    """Record inputs and environment; only non-secret configuration belongs here."""

    run_id: Identifier
    created_at: AwareDatetime
    seed: NonNegativeInt
    python_version: Text
    package_version: Text
    git_commit: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")] | None = None
    corpus_sha256: Sha256
    benchmark_sha256: Sha256
    config: dict[str, JsonValue] = Field(default_factory=dict)
    artifact_hashes: dict[str, Sha256] = Field(default_factory=dict)
    provider_versions: dict[str, str] = Field(default_factory=dict)
