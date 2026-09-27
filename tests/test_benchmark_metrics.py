import pytest

from graphrag_bench.benchmark.metrics import score_evidence
from graphrag_bench.models import (
    BenchmarkQuestion,
    EvidenceFact,
    EvidenceSet,
    EvidenceSpan,
    TextSpan,
)


def span(start=0, end=4, document_id="doc"):
    return EvidenceSpan(document_id=document_id, start=start, end=end, text="x" * (end - start))


def question(sets):
    doc_sets = [{s.document_id for fact in group for s in fact.spans} for group in sets]
    return BenchmarkQuestion(
        question_id="q",
        question="Which evidence?",
        expected_answer="Known answer",
        question_type="relationship",
        sufficient_evidence_sets=tuple(EvidenceSet(facts=tuple(s)) for s in sets),
        relevant_document_ids=tuple(sorted(set.union(*doc_sets))),
        required_document_count=min(map(len, doc_sets)),
        reasoning_hops=2,
        difficulty="medium",
        split="fixture",
        group_id="group",
    )


def fact(identifier, *spans):
    return EvidenceFact(fact_id=identifier, statement=identifier, spans=spans)


def test_partial_and_complete_two_document_evidence():
    first, second = span(document_id="one"), span(document_id="two")
    gold = question(((fact("a", first), fact("b", second)),))
    partial = score_evidence(gold, (first,))
    assert partial.evidence_coverage == 0.5 and not partial.complete_evidence
    assert partial.sets[0].supported_fact_ids == ("a",)
    assert partial.sets[0].missing_fact_ids == ("b",)
    assert score_evidence(gold, (first, second)).complete_evidence
    assert score_evidence(gold, ()).evidence_coverage == 0


def test_every_span_of_a_fact_is_required_without_partial_fact_credit():
    a, b = span(0, 4), span(8, 12)
    gold = question(((fact("joined", a, b),),))
    assert score_evidence(gold, (a,)).evidence_coverage == 0
    assert score_evidence(gold, (a, b)).evidence_coverage == 1


def test_alternative_sets_are_scored_independently_without_mixing():
    a, b, c, d = (span(i * 5, i * 5 + 4) for i in range(4))
    gold = question(((fact("a", a), fact("b", b)), (fact("c", c), fact("d", d))))
    mixed = score_evidence(gold, (a, d))
    assert mixed.evidence_coverage == 0.5 and not mixed.complete_evidence
    assert score_evidence(gold, (c, d)).complete_evidence


@pytest.mark.parametrize(
    "intervals,expected",
    [
        ([(0, 2), (2, 4)], True),
        ([(0, 3), (1, 4)], True),
        ([(0, 1), (2, 4)], False),
        ([(0, 3)], False),
    ],
)
def test_union_covers_adjacent_and_overlapping_fragments_but_not_gaps(intervals, expected):
    gold = question(((fact("a", span()),),))
    scored = score_evidence(gold, tuple(span(start, end) for start, end in intervals))
    assert scored.complete_evidence is expected


def test_duplicate_text_in_another_document_or_offset_is_not_gold_evidence():
    gold = question(((fact("a", span()),),))
    assert not score_evidence(gold, (span(document_id="other"), span(10, 14))).complete_evidence
    assert score_evidence(gold, (span(), span())).evidence_coverage == 1


def test_fixture_q19_accepts_either_complete_source_set(corpus):
    gold = next(q for q in corpus.questions if q.question_id == "q19")
    for evidence_set in gold.sufficient_evidence_sets:
        selected = [
            TextSpan(**s.model_dump(exclude={"chunk_id"}))
            for f in evidence_set.facts
            for s in f.spans
        ]
        assert score_evidence(gold, selected).complete_evidence
