"""Exact source-span coverage of complete facts and alternative sufficient sets."""

from collections import defaultdict
from collections.abc import Iterable

from pydantic import Field

from graphrag_bench.benchmark.context import merge_intervals
from graphrag_bench.models import BenchmarkQuestion, Identifier, NonNegativeInt, Record, TextSpan

METRICS_VERSION = "complete-facts-source-coverage-v1"


class EvidenceSetScore(Record):
    set_index: NonNegativeInt
    supported_fact_ids: tuple[Identifier, ...]
    missing_fact_ids: tuple[Identifier, ...]
    coverage: float = Field(ge=0, le=1)


class EvidenceScore(Record):
    evidence_coverage: float = Field(ge=0, le=1)
    complete_evidence: bool
    sets: tuple[EvidenceSetScore, ...]


def score_evidence(question: BenchmarkQuestion, spans: Iterable[TextSpan]) -> EvidenceScore:
    """All spans of a fact, all facts of one set; never mix partial alternative sets.

    A span may be covered by adjacent/overlapping selected fragments in the same
    document. Matching text at another source location earns no coordinate credit.
    Caller supplies source-validated spans; graph diagnostic quotes are not evidence.
    """
    intervals = defaultdict(list)
    for span in spans:
        intervals[span.document_id].append((span.start, span.end))
    covered = {identifier: merge_intervals(values) for identifier, values in intervals.items()}
    scores = []
    for index, evidence_set in enumerate(question.sufficient_evidence_sets):
        supported, missing = [], []
        for fact in evidence_set.facts:
            present = all(
                any(
                    start <= span.start and span.end <= end
                    for start, end in covered.get(span.document_id, ())
                )
                for span in fact.spans
            )
            (supported if present else missing).append(fact.fact_id)
        scores.append(
            EvidenceSetScore(
                set_index=index,
                supported_fact_ids=tuple(supported),
                missing_fact_ids=tuple(missing),
                coverage=len(supported) / len(evidence_set.facts),
            )
        )
    return EvidenceScore(
        evidence_coverage=max(s.coverage for s in scores),
        complete_evidence=any(not s.missing_fact_ids for s in scores),
        sets=tuple(scores),
    )
