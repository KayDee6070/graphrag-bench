"""Source-bound review receipts and coverage audits; never approve on a reviewer's behalf."""

from collections import Counter
from datetime import date
from hashlib import sha256
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, StrictBool, model_validator

from graphrag_bench.benchmark.dataset import load_benchmark
from graphrag_bench.benchmark.metrics import score_evidence
from graphrag_bench.ingestion.reader import load_ingestion
from graphrag_bench.models import (
    BenchmarkQuestion,
    Identifier,
    Record,
    Sha256,
    Text,
    require_unique,
)
from graphrag_bench.papers.catalog import PaperError
from graphrag_bench.papers.pipeline import verify_papers
from graphrag_bench.serialization import json_bytes


class ReviewChecks(Record):
    source_checked: StrictBool | None = None
    answer_supported: StrictBool | None = None
    evidence_complete: StrictBool | None = None
    alternatives_checked: StrictBool | None = None
    document_count_checked: StrictBool | None = None
    reasoning_hops_checked: StrictBool | None = None

    def decisions(self) -> tuple[bool | None, ...]:
        return tuple(self.model_dump(exclude={"schema_version"}).values())


class QuestionReview(Record):
    question_id: Identifier
    decision: Literal["pending", "approved", "revise"] = "pending"
    reviewer: Text | None = None
    reviewed_on: date | None = None
    independent_of_annotation_author: StrictBool = False
    checks: ReviewChecks = Field(default_factory=ReviewChecks)
    notes: Text | None = None

    @model_validator(mode="after")
    def require_real_decision_fields(self) -> Self:
        checks = self.checks.decisions()
        if self.decision == "pending":
            if (
                self.reviewer is not None
                or self.reviewed_on is not None
                or self.independent_of_annotation_author
                or any(c is not None for c in checks)
            ):
                raise ValueError("pending review must not contain reviewer attestations")
        elif self.reviewer is None or self.reviewed_on is None or self.notes is None:
            raise ValueError("decided review needs a reviewer, date, and notes")
        elif self.decision == "approved" and (
            not self.independent_of_annotation_author or not all(c is True for c in checks)
        ):
            raise ValueError("approval requires all checks and an independence attestation")
        return self


class ReviewSheet(Record):
    review_version: Literal["paper-review-v1"] = "paper-review-v1"
    bundle_sha256: Sha256
    questions: tuple[QuestionReview, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_questions(self) -> Self:
        require_unique(tuple(q.question_id for q in self.questions), "review question IDs")
        return self


def _verified_questions(directory: Path) -> tuple[BenchmarkQuestion, ...]:
    verify_papers(directory)
    return load_benchmark(
        directory / "ingestion/documents.jsonl", directory / "questions.jsonl", split="dev"
    ).questions


def export_review(directory: Path, output: Path) -> ReviewSheet:
    """Create only pending rows. The reviewer edits a separate, non-overwritten file."""
    try:
        questions = _verified_questions(directory)
        sheet = ReviewSheet(
            bundle_sha256=sha256((directory / "manifest.json").read_bytes()).hexdigest(),
            questions=tuple(QuestionReview(question_id=q.question_id) for q in questions),
        )
        with output.open("xb") as stream:
            stream.write(json_bytes(sheet))
        return sheet
    except (OSError, ValueError) as error:
        raise PaperError(f"cannot export paper review: {error}") from error


def verify_review(directory: Path, review_path: Path) -> dict[str, object]:
    """Validate self-reported reviewer decisions, not identity or actual independence."""
    try:
        questions = _verified_questions(directory)
        sheet = ReviewSheet.model_validate_json(review_path.read_bytes())
        expected = sha256((directory / "manifest.json").read_bytes()).hexdigest()
        if sheet.bundle_sha256 != expected:
            raise PaperError("review is stale: bundle fingerprint differs")
        if {q.question_id for q in sheet.questions} != {q.question_id for q in questions}:
            raise PaperError("review must cover exactly the frozen bundle questions")
        counts = Counter(q.decision for q in sheet.questions)
        return {
            "approved": counts["approved"],
            "revise": counts["revise"],
            "pending": counts["pending"],
            "questions": len(questions),
            "all_questions_approved": counts["approved"] == len(questions),
            "reviewer_identity_authenticated": False,
            "held_out_questions": 0,
        }
    except (OSError, ValueError) as error:
        raise PaperError(f"cannot verify paper review: {error}") from error


def audit_papers(directory: Path) -> dict[str, object]:
    """Check available-source coverage, not retrieval performance or semantic support."""
    questions = _verified_questions(directory)
    batch, _ = load_ingestion(directory / "ingestion")
    uncovered = [
        q.question_id for q in questions if not score_evidence(q, batch.chunks).complete_evidence
    ]
    used = {
        s.document_id
        for q in questions
        for e in q.sufficient_evidence_sets
        for f in e.facts
        for s in f.spans
    }
    return {
        "documents": len(batch.documents),
        "pages": sum(len(s.sections) for s in batch.sources),
        "chunks": len(batch.chunks),
        "questions": len(questions),
        "question_types": dict(sorted(Counter(q.question_type for q in questions).items())),
        "reasoning_hops": dict(sorted(Counter(str(q.reasoning_hops) for q in questions).items())),
        "required_documents": dict(
            sorted(Counter(str(q.required_document_count) for q in questions).items())
        ),
        "groups": len({q.group_id for q in questions}),
        "documents_with_reference_evidence": len(used),
        "uncovered_questions": uncovered,
        "all_source_evidence_coverable": not uncovered,
        "review_status": "pending-independent-review",
        "held_out_questions": 0,
    }
