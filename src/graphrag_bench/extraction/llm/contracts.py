"""Separate model suggestions from validated graph records and inference receipts."""

from typing import Annotated, Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

from graphrag_bench.models import (
    Chunk,
    Entity,
    EvidenceSpan,
    Identifier,
    JsonValue,
    NonNegativeInt,
    Record,
    RelationAssertion,
    Sha256,
    Text,
    require_unique,
)

EXTRACTOR_VERSION = "llm-source-proposals-v1"
PROMPT_VERSION = "chunk-entities-relations-v1"


class Proposal(BaseModel):
    # No version/default fields to distract the model. Unknown keys are errors.
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class EntityProposal(Proposal):
    id: Identifier
    name: Text
    entity_type: Identifier


class RelationProposal(Proposal):
    subject: Identifier
    predicate: Identifier
    object: Identifier
    quote: Text


class ProposedExtraction(Proposal):
    entities: tuple[EntityProposal, ...] = Field(max_length=64)
    relations: tuple[RelationProposal, ...] = Field(max_length=128)

    @model_validator(mode="after")
    def check_ids(self) -> Self:
        require_unique(tuple(e.id for e in self.entities), "local entity IDs")
        return self


class IndexedRelation(Proposal):
    subject: Text
    subject_type: Identifier
    predicate: Identifier
    object: Text
    object_type: Identifier
    sentence_id: Annotated[int, Field(ge=1, strict=True)]


class IndexedExtraction(Proposal):
    """Compact model rows; the last field selects an exact numbered source sentence."""

    relations: tuple[
        IndexedRelation
        | tuple[
            Text, Identifier, Identifier, Text, Identifier, Annotated[int, Field(ge=1, strict=True)]
        ],
        ...,
    ] = Field(max_length=4)


class ModelSpec(Record):
    provider: Text
    model_id: Text
    revision: Text
    settings: dict[str, JsonValue]
    versions: dict[str, str]


class Message(Record):
    role: Literal["system", "user", "assistant"]
    content: Text


class ExtractionRequest(Record):
    prompt_version: Literal[
        "chunk-entities-relations-v1",
        "compact-chunk-relations-v1",
        "focused-chunk-relations-v1",
        "indexed-chunk-relations-v1",
    ] = PROMPT_VERSION
    chunk: Chunk
    messages: tuple[Message, ...] = Field(min_length=1)


class Completion(Record):
    text: str
    input_tokens: NonNegativeInt
    output_tokens: NonNegativeInt
    finish_reason: Literal["eos", "length"]
    elapsed_ms: float = Field(ge=0)


class ExtractionProvider(Protocol):
    @property
    def spec(self) -> ModelSpec: ...

    def complete(self, request: ExtractionRequest) -> Completion: ...


class ChatRequest(Protocol):
    """Only messages are needed by the shared local inference adapter."""

    @property
    def messages(self) -> tuple[Message, ...]: ...


class ResponseRecord(Record):
    request_sha256: Sha256
    request: ExtractionRequest
    completion: Completion
    origin: Literal["provider", "deterministic-abstention"] = "provider"

    @model_serializer(mode="wrap")
    def preserve_provider_receipt_shape(self, handler):
        value = handler(self)
        if self.origin == "provider":
            value.pop("origin", None)
        return value


class LLMIssue(Record):
    code: Literal[
        "output_limit",
        "invalid_json",
        "invalid_entity",
        "invalid_relation",
        "invalid_quote",
        "normalized_entity",
    ]
    message: Text
    span: EvidenceSpan


class LLMExtractionResult(Record):
    entities: tuple[Entity, ...]
    relations: tuple[RelationAssertion, ...]
    issues: tuple[LLMIssue, ...]
    rejected_chunks: tuple[Identifier, ...]
    records: tuple[ResponseRecord, ...]
