import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from graphrag_bench.extraction.llm.config import LLMError
from graphrag_bench.extraction.llm.review import (
    CheckReview,
    ExtractionReviewPacket,
    build_review_packet,
    export_review_packet,
    verify_review_packet,
)
from graphrag_bench.ingestion.reader import load_ingestion

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "llm-extraction.toml"


@pytest.fixture
def checks_path(tmp_path, llm_ingestion):
    batch, hashes = load_ingestion(llm_ingestion)
    chunk_ids = [chunk.chunk_id for chunk in batch.chunks[:2]]
    payload = {
        "kind": "draft-extraction-checks",
        "review_status": "draft",
        "selection": "hand-picked for this test, not sampled",
        "source_hashes": hashes,
        "cases": [
            {
                "id": "orion-uses-nova",
                "chunk_id": chunk_ids[0],
                "reason": "the first source sentence states the relationship",
                "expected": [{"subjects": ["Orion"], "predicate": "USES", "objects": ["Nova"]}],
            },
            {
                "id": "nova-on-harbor",
                "chunk_id": chunk_ids[1],
                "reason": "the second source sentence states the evaluation",
                "expected": [
                    {"subjects": ["Nova"], "predicate": "EVALUATED_ON", "objects": ["Harbor"]}
                ],
            },
        ],
    }
    path = tmp_path / "checks.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_packet_starts_pending_with_exact_source_sentences(llm_ingestion, checks_path):
    packet = build_review_packet(llm_ingestion, checks_path, CONFIG)

    assert [case.case_id for case in packet.cases] == ["orion-uses-nova", "nova-on-harbor"]
    assert all(review.decision == "pending" for review in packet.reviews)
    assert all(review.reviewer is None for review in packet.reviews)
    batch, _ = load_ingestion(llm_ingestion)
    texts = {chunk.chunk_id: chunk.text for chunk in batch.chunks}
    for case in packet.cases:
        assert case.selected_sentences
        for sentence in case.selected_sentences:
            assert sentence.text in texts[case.chunk_id]


def test_export_writes_packet_and_never_overwrites(tmp_path, llm_ingestion, checks_path):
    output = tmp_path / "review.json"
    export_review_packet(llm_ingestion, checks_path, CONFIG, output)
    original = output.read_bytes()

    with pytest.raises(LLMError, match="cannot export extraction review"):
        export_review_packet(llm_ingestion, checks_path, CONFIG, output)
    assert output.read_bytes() == original


def test_verify_reports_pending_counts_without_authenticating_anyone(
    tmp_path, llm_ingestion, checks_path
):
    output = tmp_path / "review.json"
    export_review_packet(llm_ingestion, checks_path, CONFIG, output)

    summary = verify_review_packet(llm_ingestion, checks_path, CONFIG, output)

    assert summary == {
        "approved": 0,
        "revise": 0,
        "pending": 2,
        "cases": 2,
        "all_cases_approved": False,
        "reviewer_identity_authenticated": False,
        "held_out_cases": 0,
    }


def test_verify_counts_recorded_decisions(tmp_path, llm_ingestion, checks_path):
    output = tmp_path / "review.json"
    packet = export_review_packet(llm_ingestion, checks_path, CONFIG, output)
    decided = packet.model_copy(
        update={
            "reviews": (
                CheckReview(
                    case_id="orion-uses-nova",
                    decision="approved",
                    reviewer="second reader",
                    reviewed_on="2026-09-30",
                    independent_of_label_author=True,
                    checks={
                        "source_support": True,
                        "endpoint_direction_and_predicate": True,
                        "ontology_interpretation": True,
                        "negative_label_complete": True,
                    },
                    notes="quote supports the relationship exactly",
                ),
                CheckReview(
                    case_id="nova-on-harbor",
                    decision="revise",
                    reviewer="second reader",
                    reviewed_on="2026-09-30",
                    independent_of_label_author=True,
                    notes="dataset name is ambiguous in this chunk",
                ),
            )
        }
    )
    output.write_text(decided.model_dump_json(), encoding="utf-8")

    summary = verify_review_packet(llm_ingestion, checks_path, CONFIG, output)

    assert summary["approved"] == 1
    assert summary["revise"] == 1
    assert summary["pending"] == 0
    assert summary["all_cases_approved"] is False
    assert summary["reviewer_identity_authenticated"] is False


def test_verify_rejects_edited_checks(tmp_path, llm_ingestion, checks_path):
    output = tmp_path / "review.json"
    export_review_packet(llm_ingestion, checks_path, CONFIG, output)
    payload = json.loads(checks_path.read_text(encoding="utf-8"))
    payload["cases"][0]["reason"] = "rewritten after the packet was exported"
    checks_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(LLMError, match="differs from current source, checks, or config"):
        verify_review_packet(llm_ingestion, checks_path, CONFIG, output)


def test_checks_must_bind_to_the_prepared_source(llm_ingestion, checks_path):
    payload = json.loads(checks_path.read_text(encoding="utf-8"))
    payload["source_hashes"] = {name: "0" * 64 for name in payload["source_hashes"]}
    checks_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(LLMError, match="different source artifacts"):
        build_review_packet(llm_ingestion, checks_path, CONFIG)


def test_checks_need_unique_known_chunk_ids(llm_ingestion, checks_path):
    payload = json.loads(checks_path.read_text(encoding="utf-8"))
    payload["cases"][1]["chunk_id"] = payload["cases"][0]["chunk_id"]
    checks_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(LLMError, match="unique, known source chunk IDs"):
        build_review_packet(llm_ingestion, checks_path, CONFIG)


def test_pending_review_cannot_carry_attestations():
    with pytest.raises(ValidationError, match="must not contain reviewer attestations"):
        CheckReview(case_id="case-1", reviewer="second reader")


def test_approval_requires_independence_and_every_check():
    complete = {
        "source_support": True,
        "endpoint_direction_and_predicate": True,
        "ontology_interpretation": True,
        "negative_label_complete": True,
    }
    with pytest.raises(ValidationError, match="independence attestation"):
        CheckReview(
            case_id="case-1",
            decision="approved",
            reviewer="label author",
            reviewed_on="2026-09-30",
            independent_of_label_author=False,
            checks=complete,
            notes="approved by the person who wrote the label",
        )
    with pytest.raises(ValidationError, match="independence attestation"):
        CheckReview(
            case_id="case-1",
            decision="approved",
            reviewer="second reader",
            reviewed_on="2026-09-30",
            independent_of_label_author=True,
            checks={**complete, "negative_label_complete": None},
            notes="one check left undecided",
        )


def test_reviews_must_cover_exactly_the_packet_cases(tmp_path, llm_ingestion, checks_path):
    packet = build_review_packet(llm_ingestion, checks_path, CONFIG)
    with pytest.raises(ValidationError, match="exactly the packet cases"):
        ExtractionReviewPacket.model_validate(
            packet.model_dump(mode="json") | {"reviews": [{"case_id": "orion-uses-nova"}]}
        )
