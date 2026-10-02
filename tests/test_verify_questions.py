"""Mechanical question checks: exact decisions only, no judgement of proof."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "verify_questions", ROOT / "scripts" / "verify_questions.py"
)
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


def chunk(chunk_id, start, end, document_id="doc-1"):
    return {"chunk_id": chunk_id, "document_id": document_id, "start": start, "end": end}


def span(start, end, document_id="doc-1"):
    return {"document_id": document_id, "start": start, "end": end}


def question(sets, hops, documents=1, question_id="q1"):
    return {
        "question_id": question_id,
        "reasoning_hops": hops,
        "required_document_count": documents,
        "sufficient_evidence_sets": [{"facts": facts} for facts in sets],
    }


def fact(fact_id, spans, statement="a statement"):
    return {"fact_id": fact_id, "statement": statement, "spans": spans}


CHUNKS = [chunk("a", 0, 100), chunk("b", 95, 200), chunk("far", 400, 500)]


def test_a_span_inside_one_chunk_needs_one_chunk():
    result = verify.check(question([[fact("f1", [span(10, 50)])]], hops=1), CHUNKS)

    assert result["problems"] == []
    assert result["chunks_required"] == {"f1": 1}
    assert result["needs_adjacent_chunks"] == []


def test_a_span_crossing_a_boundary_needs_two_and_is_reported_not_failed():
    result = verify.check(question([[fact("f1", [span(80, 150)])]], hops=1), CHUNKS)

    assert result["problems"] == []
    assert result["chunks_required"] == {"f1": 2}
    assert result["needs_adjacent_chunks"] == ["f1"]


def test_a_span_in_a_gap_is_uncoverable_and_fails():
    result = verify.check(question([[fact("f1", [span(250, 300)])]], hops=1), CHUNKS)

    assert any("uncoverable" in p for p in result["problems"])


def test_hops_must_equal_the_smallest_sufficient_set():
    two_routes = [
        [fact("f1", [span(10, 20)]), fact("f2", [span(30, 40)])],
        [fact("f3", [span(50, 60)])],
    ]

    assert verify.check(question(two_routes, hops=1), CHUNKS)["problems"] == []
    problems = verify.check(question(two_routes, hops=2), CHUNKS)["problems"]
    assert any("smallest sufficient set holds 1" in p for p in problems)


def test_document_count_must_equal_the_documents_in_that_set():
    cross = [[fact("f1", [span(10, 20)]), fact("f2", [span(10, 20, "doc-2")])]]
    chunks = CHUNKS + [chunk("c", 0, 100, "doc-2")]

    assert verify.check(question(cross, hops=2, documents=2), chunks)["problems"] == []
    problems = verify.check(question(cross, hops=2, documents=1), chunks)["problems"]
    assert any("spans 2 documents" in p for p in problems)


def test_covering_run_prefers_the_fewest_chunks():
    chunks = [chunk("wide", 0, 200), chunk("left", 0, 100), chunk("right", 95, 200)]

    assert verify.covering_run([span(10, 150)], chunks) == ["wide"]


def test_covering_run_returns_none_when_no_chunk_starts_at_the_span():
    assert verify.covering_run([span(10, 50)], [chunk("late", 20, 100)]) is None


def test_the_shipped_annotations_pass_every_exact_check():
    """Regression: a real fact_id collision and three boundary cases were found this way."""
    bundle = ROOT / "datasets/processed/m10-expanded-05"
    if not bundle.exists():
        import pytest

        pytest.skip("prepared bundle not present in this checkout")
    import json

    chunks = [json.loads(line) for line in (bundle / "ingestion/chunks.jsonl").open()]
    questions = [json.loads(line) for line in (bundle / "questions.jsonl").open()]

    failures = [r for r in (verify.check(q, chunks) for q in questions) if r["problems"]]

    assert failures == []
