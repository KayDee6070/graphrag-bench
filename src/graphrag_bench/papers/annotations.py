"""Compile page-local source quotations into chunk-independent benchmark labels."""

from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from graphrag_bench.ingestion.pipeline import IngestionBatch
from graphrag_bench.models import (
    BenchmarkQuestion,
    EvidenceFact,
    EvidenceSet,
    EvidenceSpan,
    Identifier,
    PositiveInt,
    Record,
    Text,
    require_unique,
)
from graphrag_bench.papers.catalog import PaperCatalog, PaperError


class QuoteLocator(Record):
    paper_id: Identifier
    page: PositiveInt
    quote: Text


class FactAnnotation(Record):
    fact_id: Identifier
    statement: Text
    spans: tuple[QuoteLocator, ...] = Field(min_length=1)


class QuestionAnnotation(Record):
    question_id: Identifier
    question: Text
    question_type: Literal["factual", "relationship", "comparison"]
    expected_answer: Text
    sufficient_evidence_sets: tuple[tuple[FactAnnotation, ...], ...] = Field(min_length=1)
    reasoning_hops: PositiveInt
    difficulty: Literal["easy", "medium", "hard"]
    group_id: Identifier
    rationale: Text


class AnnotationBook(Record):
    """Annotations for one catalog.

    `split` defaults to `dev`, and a `test` book carries an obligation no schema can
    enforce: it must be authored without access to any measured result, and nothing may
    be run against it until the result is reported once. `prepare_papers` therefore
    refuses a `test` book unless a caller opts in explicitly, so the label cannot be
    acquired by editing one field. See docs/held-out-questions.md.
    """

    catalog_id: Identifier
    split: Literal["dev", "test"] = "dev"
    authorship: Literal["agent-source-authored", "agent-authored-results-blinded"] = (
        "agent-source-authored"
    )
    review_status: Literal["pending-independent-review"] = "pending-independent-review"
    questions: tuple[QuestionAnnotation, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_questions(self) -> Self:
        require_unique(tuple(q.question_id for q in self.questions), "question IDs")
        return self


def load_annotations(path: Path) -> AnnotationBook:
    try:
        return AnnotationBook.model_validate_json(path.read_bytes())
    except (OSError, ValueError) as error:
        raise PaperError(f"cannot load paper annotations: {error}") from error


def compile_annotations(
    catalog: PaperCatalog, annotations: AnnotationBook, batch: IngestionBatch
) -> tuple[BenchmarkQuestion, ...]:
    """No graph, retrieval ranking, inference receipt, or model is consulted."""
    if annotations.catalog_id != catalog.catalog_id:
        raise PaperError("annotation/catalog identity mismatch")
    sources = {source.relative_path: source for source in batch.sources}
    documents = {document.document_id: document for document in batch.documents}
    by_paper = {paper.paper_id: sources[paper.filename] for paper in catalog.papers}

    def resolve(locator: QuoteLocator) -> EvidenceSpan:
        source = by_paper.get(locator.paper_id)
        if source is None or locator.page > len(source.sections):
            raise PaperError(f"unknown paper/page: {locator.paper_id}/{locator.page}")
        section = source.sections[locator.page - 1]
        document = documents[source.document_id]
        page_text = document.text[section.start : section.end]
        start = page_text.find(locator.quote)
        if start < 0 or page_text.find(locator.quote, start + 1) >= 0:
            raise PaperError(f"quote must occur exactly once: {locator.paper_id}/{locator.page}")
        return EvidenceSpan(
            document_id=document.document_id,
            start=section.start + start,
            end=section.start + start + len(locator.quote),
            text=locator.quote,
        )

    questions = []
    for question in sorted(annotations.questions, key=lambda q: q.question_id):
        evidence_sets = []
        facts_by_id = {}
        for annotated_set in question.sufficient_evidence_sets:
            facts = []
            for annotation in annotated_set:
                fact = EvidenceFact(
                    fact_id=annotation.fact_id,
                    statement=annotation.statement,
                    spans=tuple(resolve(locator) for locator in annotation.spans),
                )
                if facts_by_id.setdefault(fact.fact_id, fact) != fact:
                    raise PaperError(f"inconsistent fact ID: {question.question_id}/{fact.fact_id}")
                facts.append(fact)
            evidence_sets.append(EvidenceSet(facts=tuple(facts)))
        document_sets = [
            {s.document_id for f in evidence.facts for s in f.spans} for evidence in evidence_sets
        ]
        questions.append(
            BenchmarkQuestion(
                question_id=question.question_id,
                question=question.question,
                question_type=question.question_type,
                expected_answer=question.expected_answer,
                sufficient_evidence_sets=tuple(evidence_sets),
                relevant_document_ids=tuple(sorted(set.union(*document_sets))),
                required_document_count=min(map(len, document_sets)),
                reasoning_hops=question.reasoning_hops,
                difficulty=question.difficulty,
                split=annotations.split,
                group_id=question.group_id,
            )
        )
    return tuple(questions)


def render_review(catalog: PaperCatalog, annotations: AnnotationBook) -> str:
    papers = {paper.paper_id: paper for paper in catalog.papers}
    lines = [
        "# Source annotation review worksheet",
        "",
        "Status: pending independent review. Authored by an AI agent from paper text.",
        "These development labels are separate from extracted graph proposals.",
        "Exact quotation checks do not establish semantic support or exhaustive relevance.",
        "",
        "For each question, review support, missing facts, alternatives, shortcuts, and hop count.",
        "Record reviewer identity/date and disagreements separately; "
        "do not relabel these as test data.",
    ]
    for q in sorted(annotations.questions, key=lambda q: q.question_id):
        lines.extend(
            [
                "",
                f"## {q.question_id}: {q.question}",
                "",
                f"Reference answer: {q.expected_answer}",
                "",
                f"Type: {q.question_type}; reasoning hops: {q.reasoning_hops}; "
                f"group: {q.group_id}.",
                "",
                f"Annotation rationale: {q.rationale}",
                "",
            ]
        )
        for number, evidence in enumerate(q.sufficient_evidence_sets, 1):
            lines.append(f"Sufficient evidence option {number} (all listed facts required):")
            for fact in evidence:
                lines.extend(["", f"- {fact.fact_id}: {fact.statement}"])
                for locator in fact.spans:
                    paper = papers[locator.paper_id]
                    lines.extend(
                        [
                            "",
                            f"  [{paper.title}, PDF page {locator.page}]"
                            f"({paper.pdf_url}#page={locator.page}) ({paper.license_id}):",
                            "",
                            "  ```text",
                            *[f"  {line}" for line in locator.quote.splitlines()],
                            "  ```",
                        ]
                    )
        lines.extend(["", "Reviewer / date / decision / corrections: PENDING."])
    return "\n".join(lines) + "\n"
