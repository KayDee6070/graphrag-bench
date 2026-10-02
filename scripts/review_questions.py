"""Guided, resumable annotation review. One question at a time, no JSON editing.

Shows each question from the blind worksheet, asks the six checks the review schema
requires, and writes the decision file after every answer so the session can be
interrupted and resumed. It never approves anything on the reviewer's behalf and never
shows a benchmark result.

Verify the finished file with:
    graphrag-bench verify-paper-review <bundle> --review <decisions.json>
"""

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

CHECKS = (
    ("source_checked", "Does each quote appear on the linked page, as shown?"),
    ("answer_supported", "Does the quoted text actually PROVE its statement?"),
    ("evidence_complete", "With these facts and nothing else, is the question answerable?"),
    ("alternatives_checked", "Did you check the corpus for a more direct answer elsewhere?"),
    ("document_count_checked", "Is the declared required-document count right?"),
    ("reasoning_hops_checked", "Is the declared hop count right (not too high or too low)?"),
)
PROMPT = "[y]es / [n]o / [s]kip this question / [q]uit and save: "


def sections(worksheet: Path) -> dict[str, str]:
    """Split the worksheet into per-question text keyed by question ID."""
    text = worksheet.read_text(encoding="utf-8")
    # Any question-id prefix, not just the development set's "paper-".
    parts = re.split(r"\n## ([\w-]+): ", text)
    return {parts[i]: f"## {parts[i]}: {parts[i + 1]}" for i in range(1, len(parts) - 1, 2)}


def ask(question: str) -> str | None:
    while True:
        answer = input(f"  {question}\n  {PROMPT}").strip().lower()
        if answer in {"y", "yes"}:
            return "y"
        if answer in {"n", "no"}:
            return "n"
        if answer in {"s", "skip"}:
            return "skip"
        if answer in {"q", "quit"}:
            return "quit"
        print("  Please answer y, n, s or q.")


def save(path: Path, sheet: dict) -> None:
    path.write_text(json.dumps(sheet, indent=1) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("decisions", type=Path, help="review JSON from export-paper-review")
    parser.add_argument("--worksheet", type=Path, required=True, help="blind review worksheet")
    parser.add_argument("--reviewer", required=True, help="your name, recorded in the file")
    parser.add_argument(
        "--only",
        help="comma-separated question IDs to review, e.g. paper-b01,paper-c01",
    )
    args = parser.parse_args()

    sheet = json.loads(args.decisions.read_text(encoding="utf-8"))
    text = sections(args.worksheet)
    wanted = {q.strip() for q in args.only.split(",")} if args.only else None
    rows = [
        row
        for row in sheet["questions"]
        if row["decision"] == "pending" and (wanted is None or row["question_id"] in wanted)
    ]
    if not rows:
        print("Nothing pending in that selection. Already done, or filtered out by --only.")
        return

    print(f"\n{len(rows)} question(s) to review. Answers save after each one.")
    print("Judge the label, never the outcome. Do not open any results file.\n")
    done = 0
    for row in rows:
        qid = row["question_id"]
        print("=" * 78)
        print(text.get(qid, f"(no worksheet section found for {qid})"))
        print("=" * 78)
        answers, stop = {}, None
        for field, question in CHECKS:
            reply = ask(question)
            if reply in {"skip", "quit"}:
                stop = reply
                break
            answers[field] = reply == "y"
        if stop == "quit":
            break
        if stop == "skip":
            print(f"  -> {qid} left pending.\n")
            continue
        approved = all(answers.values())
        verdict = "approved" if approved else "revise"
        print(f"\n  All six checks pass: {approved}  ->  decision will be '{verdict}'")
        notes = ""
        while not notes.strip():
            notes = input("  Notes (required, say what you saw): ")
        row.update(
            decision=verdict,
            reviewer=args.reviewer,
            reviewed_on=date.today().isoformat(),
            independent_of_annotation_author=True,
            checks=row["checks"] | answers,
            notes=notes.strip(),
        )
        save(args.decisions, sheet)
        done += 1
        print(f"  -> saved as '{verdict}'.\n")

    counts = {
        state: sum(1 for r in sheet["questions"] if r["decision"] == state)
        for state in ("approved", "revise", "pending")
    }
    print(f"\nThis session decided {done}. File now: {counts}")
    print(f"Saved to {args.decisions}")
    if counts["pending"]:
        print("Re-run the same command to continue where you left off.")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nInterrupted. Everything answered so far is already saved.")
        sys.exit(130)
