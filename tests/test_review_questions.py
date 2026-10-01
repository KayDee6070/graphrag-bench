"""The guided review tool records a reviewer's answers and invents nothing."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "review_questions", ROOT / "scripts" / "review_questions.py"
)
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)

WORKSHEET = """# Source annotation review worksheet

Status: pending independent review.

## paper-a01: First question?

Reference answer: Yes.

## paper-a02: Second question?

Reference answer: No.
"""


def test_worksheet_splits_into_question_sections(tmp_path):
    path = tmp_path / "sheet.md"
    path.write_text(WORKSHEET, encoding="utf-8")

    parts = review.sections(path)

    assert set(parts) == {"paper-a01", "paper-a02"}
    assert "First question?" in parts["paper-a01"]
    assert "Second question?" in parts["paper-a02"]
    assert "worksheet" not in parts["paper-a01"]  # the preamble is not attached to a question


def test_the_real_worksheet_yields_every_question_with_its_quotes():
    """Skipped unless the local worksheet exists; it is a generated, Git-ignored artifact."""
    worksheet = ROOT / "experiments/runs/m11-question-review-blind-04.md"
    if not worksheet.exists():
        pytest.skip("blind worksheet not generated in this checkout")

    parts = review.sections(worksheet)

    assert len(parts) == 40
    assert all("```text" in body for body in parts.values())


def test_every_schema_check_is_asked():
    assert [field for field, _ in review.CHECKS] == [
        "source_checked",
        "answer_supported",
        "evidence_complete",
        "alternatives_checked",
        "document_count_checked",
        "reasoning_hops_checked",
    ]


@pytest.mark.parametrize(
    ("typed", "expected"),
    [
        ("y", "y"),
        ("Yes", "y"),
        ("n", "n"),
        ("NO", "n"),
        ("s", "skip"),
        ("skip", "skip"),
        ("q", "quit"),
        ("quit", "quit"),
    ],
)
def test_answers_are_parsed_case_insensitively(monkeypatch, typed, expected):
    monkeypatch.setattr("builtins.input", lambda _: typed)

    assert review.ask("anything?") == expected


def test_unrecognised_input_is_re_asked(monkeypatch, capsys):
    replies = iter(["maybe", "", "y"])
    monkeypatch.setattr("builtins.input", lambda _: next(replies))

    assert review.ask("anything?") == "y"
    assert capsys.readouterr().out.count("Please answer") == 2


def test_save_writes_readable_json(tmp_path):
    path = tmp_path / "decisions.json"

    review.save(path, {"questions": [{"question_id": "paper-a01", "decision": "pending"}]})

    assert "paper-a01" in path.read_text(encoding="utf-8")
