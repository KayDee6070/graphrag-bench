"""Find passages that may answer a question but earn no credit.

The source metrics score annotated spans only. A passage that fully answers a question
without being annotated scores zero, so an unannotated alternative route makes a
retrieval failure indistinguishable from a labelling gap. paper-l01 was found that way
by hand; this sweeps for the same shape across every question.

It is a heuristic triage, not a judgement. A high term overlap is a candidate for a
human to read, nothing more. It reads gold annotations deliberately and writes no
benchmark artifact.
"""

import argparse
import json
import re
from pathlib import Path

STOP = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "these",
    "this",
    "to",
    "use",
    "used",
    "uses",
    "what",
    "which",
    "with",
    "within",
}


def terms(text: str) -> set[str]:
    """Distinctive lowercase tokens: words of 4+ characters, plus any digit run."""
    words = {w for w in re.findall(r"[A-Za-z][\w-]{3,}", text.lower()) if w not in STOP}
    return words | set(re.findall(r"\d+(?:\.\d+)?%?", text))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--min-overlap", type=float, default=0.6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    source = args.bundle / "ingestion"
    chunks = [json.loads(line) for line in (source / "chunks.jsonl").open()]
    questions = [json.loads(line) for line in (args.bundle / "questions.jsonl").open()]

    rows = []
    for question in questions:
        for index, evidence_set in enumerate(question["sufficient_evidence_sets"]):
            facts = evidence_set["facts"]
            # Terms the answer needs: every fact statement, plus the reference answer.
            needed = set()
            for fact in facts:
                needed |= terms(fact["statement"])
            needed |= terms(question["expected_answer"])
            annotated = {
                (span["document_id"], span["start"], span["end"])
                for fact in facts
                for span in fact["spans"]
            }
            candidates = []
            for chunk in chunks:
                covers_annotated = any(
                    chunk["document_id"] == document_id
                    and chunk["start"] <= start
                    and end <= chunk["end"]
                    for document_id, start, end in annotated
                )
                if covers_annotated:
                    continue
                overlap = len(needed & terms(chunk["text"])) / max(1, len(needed))
                if overlap >= args.min_overlap:
                    candidates.append(
                        {
                            "chunk_id": chunk["chunk_id"],
                            "document_id": chunk["document_id"],
                            "start": chunk["start"],
                            "overlap": round(overlap, 3),
                            "excerpt": " ".join(chunk["text"].split())[:300],
                        }
                    )
            candidates.sort(key=lambda c: -c["overlap"])
            if candidates:
                rows.append(
                    {
                        "question_id": question["question_id"],
                        "question": question["question"],
                        "set_index": index,
                        "facts": len(facts),
                        "declared_hops": question["reasoning_hops"],
                        "declared_documents": question["required_document_count"],
                        "candidates": candidates[:3],
                    }
                )

    summary = {
        "kind": "alternative-evidence-sweep-v1",
        "uses_gold_annotations": True,
        "min_overlap": args.min_overlap,
        "questions": len(questions),
        "questions_with_candidates": len({r["question_id"] for r in rows}),
        "rows": rows,
    }
    text = json.dumps(summary, indent=1, sort_keys=True)
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text + "\n")
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "rows"},
            sort_keys=True,
        )
    )
    for row in rows:
        best = row["candidates"][0]
        print(
            f"  {row['question_id']:11s} hops={row['declared_hops']} facts={row['facts']} "
            f"best_overlap={best['overlap']} {best['chunk_id'][:14]}"
        )


if __name__ == "__main__":
    main()
