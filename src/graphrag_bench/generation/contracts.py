"""Model output is a proposal; resolved citations are versioned source records."""

from typing import Annotated, Literal, Protocol, Self

from pydantic import Field, StringConstraints, model_validator

from graphrag_bench.benchmark.context import SelectedContext
from graphrag_bench.extraction.llm.contracts import Completion, Message, ModelSpec, Proposal
from graphrag_bench.generation.config import AnswerRetrievalConfig
from graphrag_bench.models import (
    EvidenceSpan,
    JsonValue,
    PositiveInt,
    Record,
    RetrievalResult,
    Sha256,
    Text,
)

GENERATION_VERSION = "cited-source-claims-v1"
SourceID = Annotated[str, StringConstraints(pattern=r"^S[1-9][0-9]*$")]


class CitationProposal(Proposal):
    source_id: SourceID
    quote: Text


class ClaimProposal(Proposal):
    text: Text
    citations: tuple[CitationProposal, ...] = Field(min_length=1, max_length=16)


class AnswerProposal(Proposal):
    status: Literal["answered", "insufficient_evidence"]
    claims: tuple[ClaimProposal, ...] = Field(max_length=8)

    @model_validator(mode="after")
    def check_status(self) -> Self:
        if (self.status == "answered") != bool(self.claims):
            raise ValueError("answered requires claims; insufficient_evidence requires no claims")
        return self


class Citation(Record):
    source_id: SourceID
    span: EvidenceSpan
    title: Text
    source_uri: Text


class CitedClaim(Record):
    text: Text
    citations: tuple[Citation, ...] = Field(min_length=1)


class AnswerIssue(Record):
    code: Literal["output_limit", "invalid_response", "invalid_citation"]
    message: Text


class Answer(Record):
    status: Literal["answered", "insufficient_evidence", "invalid_output"]
    claims: tuple[CitedClaim, ...] = ()
    issues: tuple[AnswerIssue, ...] = ()
    review_status: Literal["unreviewed"] = "unreviewed"


class TokenMeasurement(Record):
    text_sha256: Sha256
    count: PositiveInt


class AnswerRequest(Record):
    prompt_version: Literal["cited-source-claims-v1"] = GENERATION_VERSION
    query: Text
    context: SelectedContext
    token_measurements: tuple[TokenMeasurement, ...]
    messages: tuple[Message, ...]


class AnswerProvider(Protocol):
    @property
    def spec(self) -> ModelSpec: ...

    def count_tokens(self, text: str) -> int: ...

    def complete(self, request: AnswerRequest) -> Completion: ...


class AnswerReceipt(Record):
    request_sha256: Sha256
    # None records a deterministic abstention when no evidence fits the budget.
    completion: Completion | None


class RetrievalReceipt(Record):
    result: RetrievalResult
    config: AnswerRetrievalConfig
    # Verified index manifests and any ranking trace; never sent to the answer model.
    index_hashes: dict[str, Sha256] = Field(default_factory=dict)
    audit: dict[str, JsonValue] = Field(default_factory=dict)
