"""Draft questions are checked against the corpus before a bundle is ever built."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "verify_draft_questions", ROOT / "scripts" / "verify_draft_questions.py"
)
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)

DOCUMENT = {"document_id": "doc-1", "title": "A Paper", "text": "alpha beta gamma delta repeated"}
CHUNKS = [
    {"chunk_id": "c1", "document_id": "doc-1", "start": 0, "end": 16, "page": 1},
    {"chunk_id": "c2", "document_id": "doc-1", "start": 11, "end": 30, "page": 2},
]
BY_PAPER = {"a-paper": DOCUMENT}


def span(quote, page=1, paper_id="a-paper"):
    return {"paper_id": paper_id, "page": page, "quote": quote}


def test_a_quote_on_its_declared_page_passes():
    problems, location = verify.check_span(span("alpha"), BY_PAPER, CHUNKS)

    assert problems == []
    assert location == ("doc-1", 0, 5)


def test_a_missing_quote_is_reported_and_unresolved():
    problems, location = verify.check_span(span("omega"), BY_PAPER, CHUNKS)

    assert problems == ["quote not found in a-paper"]
    assert location is None


def test_an_unknown_paper_id_is_reported():
    problems, _ = verify.check_span(span("alpha", paper_id="nope"), BY_PAPER, CHUNKS)

    assert problems == ["unknown paper_id 'nope'"]


def test_a_wrong_page_is_reported():
    problems, _ = verify.check_span(span("alpha", page=2), BY_PAPER, CHUNKS)

    assert any("not the declared 2" in p for p in problems)


def test_a_quote_appearing_twice_is_rejected_because_prepare_requires_uniqueness():
    document = {"document_id": "doc-1", "title": "A Paper", "text": "beta and beta again"}

    problems, _ = verify.check_span(span("beta"), {"a-paper": document}, CHUNKS)

    assert any("occurs 2 times" in p for p in problems)


def test_a_span_spanning_two_chunks_needs_both():
    assert verify.covering_run([("doc-1", 0, 25)], CHUNKS) == ["c1", "c2"]
    assert verify.covering_run([("doc-1", 0, 5)], CHUNKS) == ["c1"]


def test_a_span_beyond_every_chunk_is_uncoverable():
    assert verify.covering_run([("doc-1", 40, 60)], CHUNKS) is None


def question(hops=1, documents=1, extra_fact=False):
    facts = [{"fact_id": "f1", "statement": "s", "spans": [span("alpha")]}]
    if extra_fact:
        facts.append({"fact_id": "f2", "statement": "s", "spans": [span("gamma", page=2)]})
    return {
        "question_id": "d1",
        "reasoning_hops": hops,
        "required_document_count": documents,
        "sufficient_evidence_sets": [facts],
    }


def test_a_consistent_single_fact_question_passes():
    assert verify.check_question(question(), BY_PAPER, CHUNKS)["problems"] == []


@pytest.mark.parametrize(("hops", "fragment"), [(2, "holds 1 facts"), (0, "holds 1 facts")])
def test_a_wrong_hop_count_is_reported(hops, fragment):
    problems = verify.check_question(question(hops=hops), BY_PAPER, CHUNKS)["problems"]

    assert any(fragment in p for p in problems)


def test_a_wrong_document_count_is_reported():
    problems = verify.check_question(question(documents=3), BY_PAPER, CHUNKS)["problems"]

    assert any("spans 1 papers" in p for p in problems)


def test_the_corpus_loader_maps_catalog_ids_to_documents():
    bundle = ROOT / "datasets/processed/m10-expanded-05"
    if not bundle.exists():
        pytest.skip("prepared bundle not present in this checkout")

    by_paper, titles, chunks = verify.load_corpus(bundle)

    assert len(by_paper) == 30
    assert all("text" in document for document in by_paper.values())
    assert len(chunks) == 6996
    assert len(titles) == 30
