"""Recording decisions must never invent one, and must carry its provenance."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "apply_owner_review", ROOT / "scripts" / "apply_owner_review.py"
)
apply_review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(apply_review)

EMPTY_CHECKS = {
    "schema_version": "1.0",
    "source_checked": None,
    "answer_supported": None,
    "evidence_complete": None,
    "alternatives_checked": None,
    "document_count_checked": None,
    "reasoning_hops_checked": None,
}


def row(question_id, **overrides):
    return {
        "question_id": question_id,
        "decision": "pending",
        "reviewer": None,
        "reviewed_on": None,
        "independent_of_annotation_author": False,
        "checks": dict(EMPTY_CHECKS),
        "notes": None,
    } | overrides


@pytest.fixture
def files(tmp_path):
    review = tmp_path / "review.json"
    review.write_text(json.dumps({"questions": [row("q1"), row("q2")]}), encoding="utf-8")
    decisions = tmp_path / "decisions.json"
    decisions.write_text(
        json.dumps(
            {
                "review_kind": "project-owner-review-not-blind-to-results",
                "reviewer": "A Reviewer",
                "reviewed_on": "2026-10-02",
                "provenance": "recorded by an agent",
                "decisions": {"q1": {"decision": "approved", "notes": "quote proves it"}},
            }
        ),
        encoding="utf-8",
    )
    return review, decisions


def run(review, decisions, monkeypatch):
    monkeypatch.setattr(
        "sys.argv", ["apply_owner_review", str(review), "--decisions", str(decisions)]
    )
    apply_review.main()
    return json.loads(review.read_text(encoding="utf-8"))["questions"]


def test_a_decided_question_is_recorded_with_its_provenance(files, monkeypatch):
    review, decisions = files

    rows = {r["question_id"]: r for r in run(review, decisions, monkeypatch)}

    assert rows["q1"]["decision"] == "approved"
    assert rows["q1"]["reviewer"] == "A Reviewer"
    assert rows["q1"]["independent_of_annotation_author"] is True
    assert all(rows["q1"]["checks"][c] is True for c in apply_review.CHECKS)
    assert "quote proves it" in rows["q1"]["notes"]
    assert "recorded by an agent" in rows["q1"]["notes"]


def test_an_undecided_question_is_left_completely_untouched(files, monkeypatch):
    review, decisions = files

    rows = {r["question_id"]: r for r in run(review, decisions, monkeypatch)}

    assert rows["q2"] == row("q2")


def test_an_existing_reviewer_is_never_overwritten(files, monkeypatch, capsys):
    review, decisions = files
    sheet = json.loads(review.read_text(encoding="utf-8"))
    sheet["questions"][0] = row("q1", decision="revise", reviewer="Someone Else", notes="mine")
    review.write_text(json.dumps(sheet), encoding="utf-8")

    rows = {r["question_id"]: r for r in run(review, decisions, monkeypatch)}

    assert rows["q1"]["reviewer"] == "Someone Else"
    assert rows["q1"]["notes"] == "mine"
    assert json.loads(capsys.readouterr().out)["refused_already_reviewed"] == ["q1"]


def test_decisions_for_unknown_questions_are_rejected(files, monkeypatch):
    review, decisions = files
    source = json.loads(decisions.read_text(encoding="utf-8"))
    source["decisions"]["not-a-question"] = {"decision": "approved", "notes": "x"}
    decisions.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(SystemExit, match="not in this review"):
        run(review, decisions, monkeypatch)


def test_the_real_owner_decisions_file_is_well_formed():
    source = json.loads(
        (ROOT / "datasets/papers/research/owner-review-decisions.json").read_text(encoding="utf-8")
    )

    assert source["review_kind"] == "project-owner-review-not-blind-to-results"
    assert "not a blind review" in source["provenance"]
    assert len(source["decisions"]) == 11
    assert all(entry["notes"].strip() for entry in source["decisions"].values())
    assert set(source["left_pending"]["question_ids"]) == {"paper-b01", "paper-l01"}
