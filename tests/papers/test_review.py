import json
from datetime import date

import pytest
from pydantic import ValidationError

from graphrag_bench.cli import main
from graphrag_bench.config import ChunkingConfig
from graphrag_bench.papers import pdf
from graphrag_bench.papers.catalog import PaperCatalog, PaperError
from graphrag_bench.papers.pipeline import prepare_papers, verify_papers
from graphrag_bench.papers.review import (
    QuestionReview,
    ReviewChecks,
    ReviewSheet,
    audit_papers,
    export_review,
    verify_review,
)
from graphrag_bench.serialization import json_bytes


@pytest.fixture
def bundle(paper_inputs, tmp_path):
    output = tmp_path / "bundle"
    prepare_papers(*paper_inputs[:3], output)
    return output


def test_export_only_pending_and_never_overwrites(bundle, tmp_path):
    path = tmp_path / "review.json"
    sheet = export_review(bundle, path)
    assert sheet.questions[0].decision == "pending"
    assert sheet.questions[0].reviewer is None
    result = verify_review(bundle, path)
    assert (result["pending"], result["approved"], result["all_questions_approved"]) == (
        1,
        0,
        False,
    )
    with pytest.raises(PaperError, match="exist"):
        export_review(bundle, path)
    assert path.read_bytes() == json_bytes(sheet)


def approved_review():
    """Simulated reviewer input for tests, never a claim of real human review."""
    return QuestionReview(
        question_id="gate",
        decision="approved",
        reviewer="fictional-test-reviewer",
        reviewed_on=date(2026, 9, 28),
        independent_of_annotation_author=True,
        checks=ReviewChecks(
            source_checked=True,
            answer_supported=True,
            evidence_complete=True,
            alternatives_checked=True,
            document_count_checked=True,
            reasoning_hops_checked=True,
        ),
        notes="Test fixture: inspected both fictional scout reports.",
    )


def test_approval_records_attestation_but_does_not_create_held_out_labels(bundle, tmp_path):
    path = tmp_path / "review.json"
    sheet = export_review(bundle, path).model_copy(update={"questions": (approved_review(),)})
    path.write_bytes(json_bytes(sheet))
    result = verify_review(bundle, path)
    assert result["all_questions_approved"] is True
    assert result["reviewer_identity_authenticated"] is False
    assert result["held_out_questions"] == 0
    assert verify_papers(bundle).review_status == "pending-independent-review"


@pytest.mark.parametrize(
    "change",
    [
        {"reviewer": None},
        {"reviewed_on": None},
        {"notes": None},
        {"independent_of_annotation_author": False},
        {"checks": {"source_checked": True}},
        {"checks": {**ReviewChecks().model_dump(), "source_checked": "yes"}},
    ],
)
def test_incomplete_approvals_fail(change):
    with pytest.raises(ValidationError):
        QuestionReview.model_validate({**approved_review().model_dump(), **change})


def test_revision_can_record_a_specific_disagreement():
    row = QuestionReview(
        question_id="gate",
        decision="revise",
        reviewer="fictional-test-reviewer",
        reviewed_on=date(2026, 9, 28),
        notes="Missing second supporting report.",
        checks=ReviewChecks(evidence_complete=False),
    )
    assert row.decision == "revise"


def test_pending_row_cannot_embed_approval_checks():
    with pytest.raises(ValidationError, match="pending review"):
        QuestionReview(question_id="gate", checks=ReviewChecks(source_checked=True))


def test_duplicate_review_ids_are_rejected():
    with pytest.raises(ValidationError, match="duplicate review question IDs"):
        ReviewSheet(bundle_sha256="0" * 64, questions=(approved_review(), approved_review()))


def test_review_detects_changed_bundle_fingerprint(bundle, tmp_path):
    path = tmp_path / "review.json"
    export_review(bundle, path)
    manifest = bundle / "manifest.json"
    manifest.write_bytes(manifest.read_bytes() + b"\n")
    with pytest.raises(PaperError, match="review is stale"):
        verify_review(bundle, path)


def test_review_requires_exact_question_inventory(bundle, tmp_path):
    path = tmp_path / "review.json"
    sheet = export_review(bundle, path)
    changed = sheet.model_copy(update={"questions": (QuestionReview(question_id="unknown"),)})
    path.write_bytes(json_bytes(changed))
    with pytest.raises(PaperError, match="exactly the frozen"):
        verify_review(bundle, path)


def test_review_verifies_underlying_sources_first(bundle, tmp_path):
    path = tmp_path / "review.json"
    export_review(bundle, path)
    (bundle / "questions.jsonl").write_text("{}\n")
    with pytest.raises(PaperError, match="checksum mismatch"):
        verify_review(bundle, path)


def test_audit_describes_coverage_not_retrieval_scores(bundle):
    report = audit_papers(bundle)
    assert report["all_source_evidence_coverable"] is True
    assert report["required_documents"] == {"2": 1}
    assert report["reasoning_hops"] == {"2": 1}
    assert report["held_out_questions"] == 0
    assert "accuracy" not in report


def test_audit_reports_quote_gaps_from_hard_chunk_cuts(paper_inputs, tmp_path):
    output = tmp_path / "small-chunks"
    prepare_papers(*paper_inputs[:3], output, config=ChunkingConfig(max_chars=10))
    report = audit_papers(output)
    assert report["all_source_evidence_coverable"] is False
    assert report["uncovered_questions"] == ["gate"]


def test_review_cli_reports_pending_status(bundle, tmp_path, capsys):
    path = tmp_path / "review.json"
    assert main(["export-paper-review", str(bundle), "--output", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "pending"
    assert main(["verify-paper-review", str(bundle), "--review", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["all_questions_approved"] is False
    assert main(["audit-papers", str(bundle)]) == 0
    assert json.loads(capsys.readouterr().out)["all_source_evidence_coverable"] is True


def research_catalog(paper_inputs, count):
    data = paper_inputs[3].model_dump()
    template = data["papers"][0]
    data.update(
        purpose="research-corpus",
        papers=[
            {**template, "paper_id": f"paper-{i}", "arxiv_version": f"2401.{i:05}v1"}
            for i in range(count)
        ],
    )
    return data


def test_research_purpose_expands_bounds_without_relaxing_pilot(paper_inputs):
    data = research_catalog(paper_inputs, 30)
    assert len(PaperCatalog.model_validate(data).papers) == 30
    with pytest.raises(ValidationError, match="pilot catalog exceeds 10 papers"):
        PaperCatalog.model_validate({**data, "purpose": "development-pilot"})


def test_research_has_a_hard_acquisition_cap(paper_inputs):
    data = research_catalog(paper_inputs, 30)
    for p in data["papers"]:
        p["pdf_bytes"] = 5_000_000
    with pytest.raises(ValidationError, match="exceeds 128 MB"):
        PaperCatalog.model_validate(data)


@pytest.mark.parametrize("license_id,slug", [("CC-BY-SA-4.0", "by-sa"), ("CC-BY-NC-4.0", "by-nc")])
def test_source_license_urls_are_preserved(paper_inputs, license_id, slug):
    data = paper_inputs[3].model_dump()
    data["papers"][0]["license_id"] = license_id
    catalog = PaperCatalog.model_validate(data)
    assert catalog.papers[0].license_url == f"https://creativecommons.org/licenses/{slug}/4.0/"


def test_optional_font_decoder_cannot_silently_change_coordinates(monkeypatch):
    monkeypatch.setattr(pdf, "find_spec", lambda name: object())
    with pytest.raises(PaperError, match="without optional fontTools"):
        pdf.extract_pages(b"not parsed after preflight")


def test_expanded_bundle_roundtrip_and_purpose_binding(paper_inputs, tmp_path):
    path = paper_inputs[0]
    data = json.loads(path.read_text())
    data["purpose"] = "research-corpus"
    path.write_text(json.dumps(data))
    output = tmp_path / "expanded"
    manifest = prepare_papers(*paper_inputs[:3], output)
    assert manifest.bundle_version == "paper-corpus-v1"
    assert verify_papers(output, raw_directory=paper_inputs[2]) == manifest
    attribution = (output / "attribution.md").read_text()
    assert "their own licenses still apply" in attribution
    value = json.loads((output / "manifest.json").read_text())
    value["bundle_version"] = "paper-pilot-v1"
    (output / "manifest.json").write_text(json.dumps(value))
    with pytest.raises(PaperError, match="purpose and bundle version"):
        verify_papers(output)
