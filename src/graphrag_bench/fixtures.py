"""Load and audit gold development fixtures, never production retrieval inputs."""

from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from typing import Self, TypeVar

from pydantic import Field, ValidationError, model_validator

from graphrag_bench.models import (
    BenchmarkQuestion,
    Chunk,
    Document,
    Entity,
    EvidenceSpan,
    Identifier,
    Record,
    RelationAssertion,
    Sha256,
    require_unique,
)

T = TypeVar("T", bound=Record)
FIXTURE_FILES = (
    "corpus/documents.jsonl",
    "gold/chunks.jsonl",
    "gold/entities.jsonl",
    "gold/relations.jsonl",
    "gold/questions.jsonl",
    "ontology.json",
)


class FixtureError(ValueError):
    """A fixture cannot be read or violates its data contract."""


class Ontology(Record):
    entity_types: tuple[Identifier, ...] = Field(min_length=1)
    relation_types: tuple[Identifier, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def verify_labels(self) -> Self:
        require_unique(self.entity_types, "ontology entity types")
        require_unique(self.relation_types, "ontology relation types")
        return self


class FixtureManifest(Record):
    fixture_id: Identifier
    files: dict[str, Sha256]

    @model_validator(mode="after")
    def verify_file_list(self) -> Self:
        if set(self.files) != set(FIXTURE_FILES):
            raise ValueError("manifest must list exactly the six fixture data files")
        return self


def _unique_index(records: tuple[T, ...], key: Callable[[T], str]) -> dict[str, T]:
    keys = tuple(key(record) for record in records)
    require_unique(keys, f"{type(records[0]).__name__} IDs")
    return dict(zip(keys, records, strict=True))


def _validate_span(
    span: EvidenceSpan | Chunk,
    documents: dict[str, Document],
    chunks: dict[str, Chunk],
) -> None:
    document = documents.get(span.document_id)
    if document is None:
        raise ValueError(f"unknown document: {span.document_id}")
    if document.text[span.start : span.end] != span.text:
        raise ValueError(f"source text mismatch: {span.document_id}[{span.start}:{span.end}]")
    if isinstance(span, EvidenceSpan) and span.chunk_id is not None:
        chunk = chunks.get(span.chunk_id)
        if chunk is None:
            raise ValueError(f"unknown chunk: {span.chunk_id}")
        if chunk.document_id != span.document_id:
            raise ValueError(f"chunk/document mismatch: {span.chunk_id}")
        if not chunk.start <= span.start < span.end <= chunk.end:
            raise ValueError(f"evidence outside chunk: {span.chunk_id}")


class FixtureDataset(Record):
    """Sources plus gold annotations for tests; this is not an extractor result."""

    ontology: Ontology
    documents: tuple[Document, ...] = Field(min_length=1)
    chunks: tuple[Chunk, ...] = Field(min_length=1)
    entities: tuple[Entity, ...] = Field(min_length=1)
    relations: tuple[RelationAssertion, ...] = Field(min_length=1)
    questions: tuple[BenchmarkQuestion, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def verify_integrity(self) -> Self:
        documents = _unique_index(self.documents, lambda item: item.document_id)
        chunks = _unique_index(self.chunks, lambda item: item.chunk_id)
        entities = _unique_index(self.entities, lambda item: item.entity_id)
        _unique_index(self.relations, lambda item: item.assertion_id)
        _unique_index(self.questions, lambda item: item.question_id)
        require_unique(
            tuple(f"{chunk.document_id}:{chunk.ordinal}" for chunk in self.chunks),
            "chunk ordinals within a document",
        )
        for chunk in self.chunks:
            _validate_span(chunk, documents, chunks)
        for entity in self.entities:
            if entity.entity_type not in self.ontology.entity_types:
                raise ValueError(f"unknown entity type: {entity.entity_type}")
            names = {name.casefold() for name in (entity.name, *entity.aliases)}
            for span in entity.mentions:
                _validate_span(span, documents, chunks)
                if span.text.casefold() not in names:
                    raise ValueError(f"mention is not an entity name or alias: {entity.entity_id}")
        for relation in self.relations:
            if relation.subject_id not in entities or relation.object_id not in entities:
                raise ValueError(f"unknown relation endpoint: {relation.assertion_id}")
            if relation.predicate not in self.ontology.relation_types:
                raise ValueError(f"unknown relation type: {relation.predicate}")
            for span in relation.evidence:
                _validate_span(span, documents, chunks)
        self._verify_questions(documents, chunks, entities)
        return self

    def _verify_questions(
        self,
        documents: dict[str, Document],
        chunks: dict[str, Chunk],
        entities: dict[str, Entity],
    ) -> None:
        group_splits: dict[str, str] = {}
        for question in self.questions:
            previous = group_splits.setdefault(question.group_id, question.split)
            if previous != question.split:
                raise ValueError(f"question group crosses splits: {question.group_id}")
            if any(entity_id not in entities for entity_id in question.relevant_entity_ids):
                raise ValueError(f"unknown relevant entity: {question.question_id}")
            for evidence in question.sufficient_evidence_sets:
                for fact in evidence.facts:
                    for span in fact.spans:
                        _validate_span(span, documents, chunks)


def read_jsonl(path: Path, model: type[T]) -> tuple[T, ...]:
    """Read one record per line, retaining filename and line number on errors."""
    records: list[T] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    raise FixtureError(f"{path}:{line_number}: blank JSONL record")
                try:
                    records.append(model.model_validate_json(line))
                except ValidationError as error:
                    raise FixtureError(f"{path}:{line_number}: {error}") from error
    except (OSError, UnicodeError) as error:
        raise FixtureError(f"cannot read {path}: {error}") from error
    if not records:
        raise FixtureError(f"{path}: empty JSONL file")
    return tuple(records)


def load_fixture(root: Path) -> FixtureDataset:
    """Verify bytes, schema, references, provenance, and benchmark consistency."""
    try:
        manifest = FixtureManifest.model_validate_json((root / "manifest.json").read_bytes())
        for relative_path in FIXTURE_FILES:
            digest = sha256((root / relative_path).read_bytes()).hexdigest()
            if manifest.files[relative_path] != digest:
                raise FixtureError(f"checksum mismatch: {relative_path}")
        return FixtureDataset(
            ontology=Ontology.model_validate_json((root / "ontology.json").read_bytes()),
            documents=read_jsonl(root / "corpus/documents.jsonl", Document),
            chunks=read_jsonl(root / "gold/chunks.jsonl", Chunk),
            entities=read_jsonl(root / "gold/entities.jsonl", Entity),
            relations=read_jsonl(root / "gold/relations.jsonl", RelationAssertion),
            questions=read_jsonl(root / "gold/questions.jsonl", BenchmarkQuestion),
        )
    except (OSError, ValidationError) as error:
        raise FixtureError(f"invalid fixture at {root}: {error}") from error
