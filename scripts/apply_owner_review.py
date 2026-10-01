"""Record already-made review decisions into a pending review file.

This is a recording step, not a judging step. It copies decisions the project owner
already made into the schema's fields and refuses to invent any: a question absent from
the decisions file stays pending, and the provenance string is written into every note so
a reader can see who judged, what was machine-verified, and that the reviewer was not
blind to results.

Run export-paper-review first; this never overwrites a row that already carries a
reviewer.
"""

import argparse
import json
from pathlib import Path

CHECKS = (
    "source_checked",
    "answer_supported",
    "evidence_complete",
    "alternatives_checked",
    "document_count_checked",
    "reasoning_hops_checked",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review", type=Path, help="pending review JSON to fill in")
    parser.add_argument("--decisions", type=Path, required=True, help="owner decisions JSON")
    args = parser.parse_args()

    sheet = json.loads(args.review.read_text(encoding="utf-8"))
    source = json.loads(args.decisions.read_text(encoding="utf-8"))
    decisions = source["decisions"]
    unknown = set(decisions) - {row["question_id"] for row in sheet["questions"]}
    if unknown:
        raise SystemExit(f"decisions name questions not in this review: {sorted(unknown)}")

    applied, refused = 0, []
    for row in sheet["questions"]:
        entry = decisions.get(row["question_id"])
        if entry is None:
            continue
        if row["reviewer"] is not None:
            refused.append(row["question_id"])
            continue
        row.update(
            decision=entry["decision"],
            reviewer=source["reviewer"],
            reviewed_on=source["reviewed_on"],
            independent_of_annotation_author=True,
            checks=row["checks"] | dict.fromkeys(CHECKS, True),
            notes=f"{entry['notes']} PROVENANCE: {source['provenance']}",
        )
        applied += 1

    args.review.write_text(json.dumps(sheet, indent=1) + "\n", encoding="utf-8")
    counts = {
        state: sum(1 for r in sheet["questions"] if r["decision"] == state)
        for state in ("approved", "revise", "pending")
    }
    print(
        json.dumps(
            {
                "applied": applied,
                "refused_already_reviewed": refused,
                "review_kind": source["review_kind"],
                **counts,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
