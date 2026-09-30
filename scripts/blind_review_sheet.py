"""Strip the annotation author's reasoning from a review worksheet.

The generated worksheet explains why each label was chosen, including why a question
was considered two-hop. That reasoning is itself under review, so reading it first
invites agreement rather than judgement. This removes those paragraphs and keeps
everything a reviewer needs: the question, the reference answer, the declared labels,
and every exact quote with its page link.

It removes persuasion, not information. A reviewer still sees each label and must still
confirm or lower it. It cannot make a results-aware reviewer blind to results.
"""

import argparse
from pathlib import Path

RATIONALE = "Annotation rationale:"
REMINDER = (
    "Judge this question on the quotes below alone. For each fact: does the quoted text "
    "prove the statement, on the linked page? Then decide whether any required fact is "
    "missing, whether another passage in the corpus answers the question more directly, "
    "and whether the declared hop and document counts are right or too high."
)


def blind(text: str) -> tuple[str, int]:
    """Drop rationale paragraphs, replacing each with a neutral reminder."""
    blocks = text.split("\n\n")
    kept, removed = [], 0
    for block in blocks:
        if block.lstrip().startswith(RATIONALE):
            removed += 1
            kept.append(REMINDER)
        else:
            kept.append(block)
    return "\n\n".join(kept), removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("worksheet", type=Path, help="generated review.md")
    parser.add_argument("--output", type=Path, required=True, help="new blind worksheet")
    args = parser.parse_args()
    text = args.worksheet.read_text(encoding="utf-8")
    result, removed = blind(text)
    if not removed:
        raise SystemExit(f"no '{RATIONALE}' sections found in {args.worksheet}")
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(result)
    questions = text.count("\n## ")
    print(
        f'{{"questions": {questions}, "rationales_removed": {removed}, "output": "{args.output}"}}'
    )


if __name__ == "__main__":
    main()
