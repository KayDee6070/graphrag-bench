"""Source metadata for ingestion; the reusable M1 document contract is unchanged."""

from dataclasses import dataclass
from typing import Literal, Self

from pydantic import Field, model_validator

from graphrag_bench.models import (
    Document,
    Identifier,
    NonNegativeInt,
    PositiveInt,
    Record,
    Sha256,
    Text,
)


class IngestionError(ValueError):
    """Input text or an ingestion output cannot be processed."""


class Section(Record):
    """A half-open document range, including its Markdown heading if present."""

    start: NonNegativeInt
    end: PositiveInt
    title: Text | None = None

    @model_validator(mode="after")
    def require_nonempty_range(self) -> Self:
        if self.end <= self.start:
            raise ValueError("section end must be greater than start")
        return self


class SourceRecord(Record):
    document_id: Identifier
    relative_path: Text
    format: Literal["text", "markdown"]
    raw_sha256: Sha256
    raw_bytes: PositiveInt
    had_utf8_bom: bool
    sections: tuple[Section, ...] = Field(min_length=1)


@dataclass(frozen=True)
class ParsedDocument:
    document: Document
    source: SourceRecord
