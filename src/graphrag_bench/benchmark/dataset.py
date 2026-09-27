"""Read source documents and gold questions without importing a gold graph."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import ValidationError

from graphrag_bench.corpus import CorpusIndex
from graphrag_bench.models import BenchmarkQuestion, Document, Record, require_unique

T = TypeVar("T", bound=Record)


class BenchmarkError(ValueError):
    """Benchmark inputs or outputs violate the experiment contract."""


@dataclass(frozen=True)
class BenchmarkDataset:
    documents: tuple[Document, ...]
    questions: tuple[BenchmarkQuestion, ...]
    corpus_sha256: str
    benchmark_sha256: str


def _read(path: Path, model: type[T]) -> tuple[tuple[T, ...], str]:
    raw = path.read_bytes()
    records = []
    for number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        try:
            records.append(model.model_validate_json(line))
        except ValidationError as error:
            raise BenchmarkError(f"{path}:{number}: {error}") from error
    if not records:
        raise BenchmarkError(f"empty benchmark input: {path}")
    return tuple(records), sha256(raw).hexdigest()


def load_benchmark(
    documents_path: Path,
    questions_path: Path,
    *,
    split: Literal["fixture", "dev", "test"],
) -> BenchmarkDataset:
    """Validate every annotation and group split before selecting the requested split.

    Gold chunk IDs and entity IDs are annotation metadata, not runtime retrieval IDs.
    Gold evidence is checked against document coordinates independently of chunking.
    """
    try:
        if split not in {"fixture", "dev", "test"}:
            raise BenchmarkError("split must be fixture, dev, or test")
        documents, corpus_hash = _read(documents_path, Document)
        questions, benchmark_hash = _read(questions_path, BenchmarkQuestion)
        corpus = CorpusIndex(documents, ())
        require_unique(tuple(q.question_id for q in questions), "question IDs")
        groups: dict[str, str] = {}
        for question in questions:
            if groups.setdefault(question.group_id, question.split) != question.split:
                raise BenchmarkError(f"question group crosses splits: {question.group_id}")
            facts = {}
            for evidence_set in question.sufficient_evidence_sets:
                for fact in evidence_set.facts:
                    if facts.setdefault(fact.fact_id, fact) != fact:
                        raise BenchmarkError(
                            f"inconsistent fact ID: {question.question_id}/{fact.fact_id}"
                        )
                    for span in fact.spans:
                        document = corpus.documents.get(span.document_id)
                        if document is None or document.text[span.start : span.end] != span.text:
                            raise BenchmarkError(
                                f"gold source text mismatch: {question.question_id}"
                            )
        selected = tuple(
            sorted((q for q in questions if q.split == split), key=lambda q: q.question_id)
        )
        if not selected:
            raise BenchmarkError(f"no questions in requested split: {split}")
        return BenchmarkDataset(
            tuple(corpus.documents.values()), selected, corpus_hash, benchmark_hash
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise BenchmarkError(f"cannot load benchmark: {error}") from error
