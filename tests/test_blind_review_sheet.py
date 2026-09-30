"""Blinding removes the annotation author's reasoning and nothing else."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "blind_review_sheet", ROOT / "scripts" / "blind_review_sheet.py"
)
blind_sheet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(blind_sheet)

SHEET = """# Source annotation review worksheet

Status: pending independent review.

## paper-x01: Which dataset evaluates the model?

Reference answer: Harbor.

Type: relationship; reasoning hops: 2; group: family.

Annotation rationale: Candidate hidden-bridge chain: identify the unnamed method,
then find its property. Two annotated documents and two successive relations.

Sufficient evidence option 1 (all listed facts required):

- fact-one: Alder uses Nova.

  [Alder paper, PDF page 3](https://arxiv.org/pdf/2401.18059v1#page=3) (CC-BY-4.0):

  ```text
  Alder uses Nova.
  ```

Reviewer / date / decision / corrections: PENDING.
"""


def test_rationale_is_replaced_by_a_neutral_reminder():
    result, removed = blind_sheet.blind(SHEET)

    assert removed == 1
    assert "Annotation rationale" not in result
    assert "hidden-bridge chain" not in result
    assert blind_sheet.REMINDER in result


def test_everything_a_reviewer_needs_survives():
    result, _ = blind_sheet.blind(SHEET)

    for required in (
        "## paper-x01: Which dataset evaluates the model?",
        "Reference answer: Harbor.",
        "Type: relationship; reasoning hops: 2; group: family.",
        "- fact-one: Alder uses Nova.",
        "https://arxiv.org/pdf/2401.18059v1#page=3",
        "Alder uses Nova.",
        "Reviewer / date / decision / corrections: PENDING.",
    ):
        assert required in result


def test_declared_labels_are_kept_because_they_are_what_is_reviewed():
    result, _ = blind_sheet.blind(SHEET)

    assert "reasoning hops: 2" in result
    assert "confirm or lower" in blind_sheet.REMINDER or "too high" in blind_sheet.REMINDER


def test_quote_and_link_counts_are_unchanged():
    result, _ = blind_sheet.blind(SHEET)

    assert result.count("```text") == SHEET.count("```text")
    assert result.count("arxiv.org/pdf") == SHEET.count("arxiv.org/pdf")
    assert result.count("PENDING") == SHEET.count("PENDING")


def test_a_sheet_without_rationales_reports_nothing_removed():
    plain = SHEET.replace(
        "Annotation rationale: Candidate hidden-bridge chain: identify the unnamed method,\n"
        "then find its property. Two annotated documents and two successive relations.",
        "Some other paragraph.",
    )

    result, removed = blind_sheet.blind(plain)

    assert removed == 0
    assert result == plain


def test_several_questions_are_each_blinded():
    doubled = SHEET + SHEET.split("# Source annotation review worksheet")[1]

    _, removed = blind_sheet.blind(doubled)

    assert removed == 2
