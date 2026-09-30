"""Source-bound review packets for draft extraction checks.

Packets expose labels and exact model input to a reviewer. They never approve a
relationship, authenticate a reviewer, or make a development set held out.
"""

import json
from collections import Counter
from datetime import date
from hashlib import sha256
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, StrictBool, model_validator

from graphrag_bench.corpus import CorpusIndex
from graphrag_bench.extraction.llm.config import LLMError, LLMExtractionConfig, load_llm_config
from graphrag_bench.extraction.llm.prompt import numbered_spans
from graphrag_bench.ingestion.reader import load_ingestion
from graphrag_bench.models import Identifier, PositiveInt, Record, Sha256, Text, require_unique
from graphrag_bench.serialization import json_bytes


class DraftFact(Record):
    subjects: tuple[Text, ...] = Field(min_length=1)
    predicate: Identifier
    objects: tuple[Text, ...] = Field(min_length=1)


class ReviewChecks(Record):
    source_support: StrictBool | None = None
    endpoint_direction_and_predicate: StrictBool | None = None
    ontology_interpretation: StrictBool | None = None
    negative_label_complete: StrictBool | None = None

    def decisions(self) -> tuple[bool | None, ...]:
        return tuple(self.model_dump(exclude={"schema_version"}).values())


class CheckReview(Record):
    case_id: Identifier
    decision: Literal["pending", "approved", "revise"] = "pending"
    reviewer: Text | None = None
    reviewed_on: date | None = None
    independent_of_label_author: StrictBool = False
    checks: ReviewChecks = Field(default_factory=ReviewChecks)
    notes: Text | None = None

    @model_validator(mode="after")
    def require_decision_fields(self) -> Self:
        decisions = self.checks.decisions()
        if self.decision == "pending":
            if (
                self.reviewer is not None
                or self.reviewed_on is not None
                or self.independent_of_label_author
                or any(value is not None for value in decisions)
                or self.notes is not None
            ):
                raise ValueError("pending review must not contain reviewer attestations or notes")
        elif self.reviewer is None or self.reviewed_on is None or self.notes is None:
            raise ValueError("a decided review needs reviewer, date, and notes")
        elif self.decision == "approved" and (
            not self.independent_of_label_author or not all(value is True for value in decisions)
        ):
            raise ValueError("approval requires all checks and an independence attestation")
        return self


class ReviewSentence(Record):
    sentence_id: PositiveInt
    text: Text


class ReviewCase(Record):
    case_id: Identifier
    chunk_id: Identifier
    rationale: Text
    expected: tuple[DraftFact, ...]
    selected_sentences: tuple[ReviewSentence, ...]


class ExtractionReviewPacket(Record):
    review_version: Literal["extraction-check-review-v1"] = "extraction-check-review-v1"
    checks_sha256: Sha256
    source_hashes: dict[str, Sha256]
    config_sha256: Sha256
    config: dict
    cases: tuple[ReviewCase, ...] = Field(min_length=1)
    reviews: tuple[CheckReview, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def matching_inventory(self) -> Self:
        case_ids = tuple(case.case_id for case in self.cases)
        review_ids = tuple(review.case_id for review in self.reviews)
        require_unique(case_ids, "review case IDs")
        require_unique(review_ids, "review decision IDs")
        if set(case_ids) != set(review_ids):
            raise ValueError("reviews must cover exactly the packet cases")
        return self


def _load_cases(
    source: Path, checks_path: Path, config: LLMExtractionConfig
) -> tuple[dict, CorpusIndex, dict]:
    try:
        raw = checks_path.read_bytes()
        checks = json.loads(raw)
        batch, hashes = load_ingestion(source)
        if checks["source_hashes"] != hashes:
            raise ValueError("draft checks refer to different source artifacts")
        corpus = CorpusIndex(batch.documents, batch.chunks)
        ids = [case["chunk_id"] for case in checks["cases"]]
        if len(set(ids)) != len(ids) or not set(ids) <= corpus.chunks.keys():
            raise ValueError("draft checks need unique, known source chunk IDs")
        return checks, corpus, hashes
    except (KeyError, OSError, UnicodeError, ValueError) as error:
        raise LLMError(f"cannot prepare extraction review: {error}") from error


def build_review_packet(
    source: Path, checks_path: Path, config_path: Path
) -> ExtractionReviewPacket:
    """Build pending review rows from source-bound draft extraction checks."""
    config = load_llm_config(config_path)
    checks, corpus, hashes = _load_cases(source, checks_path, config)
    review_cases = []
    for case in checks["cases"]:
        chunk = corpus.chunks[case["chunk_id"]]
        review_cases.append(
            ReviewCase(
                case_id=case["id"],
                chunk_id=chunk.chunk_id,
                rationale=case["reason"],
                expected=tuple(DraftFact.model_validate(fact) for fact in case["expected"]),
                selected_sentences=tuple(
                    ReviewSentence(sentence_id=index, text=chunk.text[start:end])
                    for index, (start, end) in enumerate(numbered_spans(chunk.text, config), 1)
                ),
            )
        )
    return ExtractionReviewPacket(
        checks_sha256=sha256(checks_path.read_bytes()).hexdigest(),
        source_hashes=hashes,
        config_sha256=sha256(config_path.read_bytes()).hexdigest(),
        config=config.model_dump(mode="json"),
        cases=tuple(review_cases),
        reviews=tuple(CheckReview(case_id=case.case_id) for case in review_cases),
    )


def export_review_packet(
    source: Path, checks_path: Path, config_path: Path, output: Path
) -> ExtractionReviewPacket:
    """Write a new pending packet; never overwrite a reviewer's decisions."""
    packet = build_review_packet(source, checks_path, config_path)
    try:
        with output.open("xb") as stream:
            stream.write(json_bytes(packet))
    except OSError as error:
        raise LLMError(f"cannot export extraction review: {error}") from error
    return packet


def verify_review_packet(
    source: Path, checks_path: Path, config_path: Path, review_path: Path
) -> dict[str, object]:
    """Verify source/config binding and declared review rows, not human identity."""
    try:
        expected = build_review_packet(source, checks_path, config_path)
        packet = ExtractionReviewPacket.model_validate_json(review_path.read_bytes())
        if (
            packet.checks_sha256 != expected.checks_sha256
            or packet.source_hashes != expected.source_hashes
            or packet.config_sha256 != expected.config_sha256
            or packet.config != expected.config
            or packet.cases != expected.cases
        ):
            raise ValueError("review packet differs from current source, checks, or config")
        decisions = Counter(review.decision for review in packet.reviews)
        return {
            "approved": decisions["approved"],
            "revise": decisions["revise"],
            "pending": decisions["pending"],
            "cases": len(packet.cases),
            "all_cases_approved": decisions["approved"] == len(packet.cases),
            "reviewer_identity_authenticated": False,
            "held_out_cases": 0,
        }
    except (OSError, UnicodeError, ValueError) as error:
        raise LLMError(f"cannot verify extraction review: {error}") from error
