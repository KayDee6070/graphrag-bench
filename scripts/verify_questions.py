"""Mechanical checks on draft benchmark questions, for use before they are accepted.

What this cannot do is judge whether a quote proves its statement. Two heuristics for
that were tried and rejected: requiring a statement's distinctive terms to appear in its
own quotes fires on 38 of the existing fact/statement pairs, and requiring a question's
terms to appear in its quotes fires on 31 question/set pairs. Both are dominated by a
benign pattern, papers referring to their own method as "the model", and neither
discriminated between the annotations before and after real defects were fixed in them.
So this tool enforces only what is exactly decidable and leaves proof to a reviewer.

`prepare-papers` already requires each quote to occur exactly once on its claimed page,
so that is not rechecked here. These are the checks nothing else performs:

  chunks_required      how many chunks must be retrieved together to satisfy each fact;
                       a span crossing a boundary needs two, which is legal but harder
  hops_consistent      reasoning_hops equals the facts in the smallest sufficient set
  documents_consistent required_document_count equals distinct papers in that set
  identifiers_sane     no fact_id carrying two different statements anywhere in the set
  uncoverable          a fact no run of adjacent chunks can satisfy, which is fatal
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def covering_run(spans, chunks) -> list[str] | None:
    """Smallest greedy run of chunks covering every span, or None if impossible."""
    group: set[str] = set()
    for span in spans:
        overlapping = sorted(
            (
                c
                for c in chunks
                if c["document_id"] == span["document_id"]
                and c["start"] < span["end"]
                and span["start"] < c["end"]
            ),
            key=lambda c: c["start"],
        )
        reached = span["start"]
        for chunk in overlapping:
            if chunk["start"] <= reached < chunk["end"]:
                group.add(chunk["chunk_id"])
                reached = chunk["end"]
                if reached >= span["end"]:
                    break
        if reached < span["end"]:
            return None
    return sorted(group)


def check(question: dict, chunks: list[dict]) -> dict:
    """Exact checks on one question. 'problems' empty means it passed every one."""
    sets = question["sufficient_evidence_sets"]
    sizes = [len(s["facts"]) for s in sets]
    smallest = sets[sizes.index(min(sizes))]
    papers = {span["document_id"] for fact in smallest["facts"] for span in fact["spans"]}
    runs, uncoverable = {}, []
    for evidence_set in sets:
        for fact in evidence_set["facts"]:
            run = covering_run(fact["spans"], chunks)
            if run is None:
                uncoverable.append(fact["fact_id"])
            else:
                runs[fact["fact_id"]] = len(run)
    problems = []
    if uncoverable:
        problems.append(f"uncoverable facts: {sorted(set(uncoverable))}")
    if question["reasoning_hops"] != min(sizes):
        problems.append(
            f"reasoning_hops is {question['reasoning_hops']} "
            f"but the smallest sufficient set holds {min(sizes)} facts"
        )
    if question["required_document_count"] != len(papers):
        problems.append(
            f"required_document_count is {question['required_document_count']} "
            f"but the smallest set spans {len(papers)} documents"
        )
    return {
        "question_id": question["question_id"],
        "problems": problems,
        "sets": sizes,
        "chunks_required": runs,
        "needs_adjacent_chunks": sorted(k for k, v in runs.items() if v > 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="prepared paper bundle")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    source = args.bundle / "ingestion"
    chunks = [json.loads(line) for line in (source / "chunks.jsonl").open()]
    questions = [json.loads(line) for line in (args.bundle / "questions.jsonl").open()]

    statements = defaultdict(set)
    for question in questions:
        for evidence_set in question["sufficient_evidence_sets"]:
            for fact in evidence_set["facts"]:
                statements[fact["fact_id"]].add(fact["statement"])
    collisions = sorted(k for k, v in statements.items() if len(v) > 1)

    rows = [check(question, chunks) for question in questions]
    failed = [r for r in rows if r["problems"]]
    summary = {
        "kind": "question-mechanical-checks-v1",
        "questions": len(rows),
        "failed": len(failed),
        "fact_id_statement_collisions": collisions,
        "facts_needing_adjacent_chunks": sorted(
            fact for row in rows for fact in row["needs_adjacent_chunks"]
        ),
        "rows": rows,
    }
    text = json.dumps(summary, indent=1, sort_keys=True)
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text + "\n")
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "rows"},
            indent=1,
            sort_keys=True,
        )
    )
    for row in failed:
        for problem in row["problems"]:
            print(f"  FAIL {row['question_id']}: {problem}")
    raise SystemExit(1 if failed or collisions else 0)


if __name__ == "__main__":
    main()
