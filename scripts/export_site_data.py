"""Export a compact JSON of the held-out run for the static site.

The site shows real retrieved text, not a mock. This writes only what the page needs:
each question, its required facts, and for each method the chunks it actually returned
with whether the question was completed. No model, no network, read-only.
"""

import argparse
import json
from pathlib import Path

STRATEGIES = ("bm25", "vector", "hybrid", "graph")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="a verified paper comparison directory")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top", type=int, default=5, help="hits to include per method")
    args = parser.parse_args()

    chunks = {
        c["chunk_id"]: c
        for c in (json.loads(line) for line in (args.bundle / "ingestion/chunks.jsonl").open())
    }
    documents = {
        d["document_id"]: d
        for d in (json.loads(line) for line in (args.bundle / "ingestion/documents.jsonl").open())
    }
    questions = {
        q["question_id"]: q
        for q in (json.loads(line) for line in (args.bundle / "questions.jsonl").open())
    }
    rows = [json.loads(line) for line in (args.run / "results.jsonl").open()]
    summary = json.loads((args.run / "summary.json").read_text())

    def title(chunk_id: str) -> str:
        return documents[chunks[chunk_id]["document_id"]]["title"]

    out = []
    for qid, question in questions.items():
        facts = []
        wanted = set()
        for evidence_set in question["sufficient_evidence_sets"]:
            for fact in evidence_set["facts"]:
                facts.append(
                    {
                        "id": fact["fact_id"],
                        "statement": fact["statement"],
                        "paper": documents[fact["spans"][0]["document_id"]]["title"],
                        "quote": fact["spans"][0]["text"],
                    }
                )
                for span in fact["spans"]:
                    for chunk in chunks.values():
                        if (
                            chunk["document_id"] == span["document_id"]
                            and chunk["start"] <= span["start"]
                            and span["end"] <= chunk["end"]
                        ):
                            wanted.add(chunk["chunk_id"])
        methods = {}
        for strategy in STRATEGIES:
            row = next(
                r
                for r in rows
                if r["question_id"] == qid and r["strategy"] == strategy and r["repeat"] == 0
            )
            hits = row["observation"]["result"]["hits"][: args.top]
            methods[strategy] = {
                "complete": next(e for e in row["evaluations"] if e["k"] == 5)["raw"][
                    "complete_evidence"
                ],
                "coverage": next(e for e in row["evaluations"] if e["k"] == 5)["raw"][
                    "evidence_coverage"
                ],
                "hits": [
                    {
                        "paper": title(h["chunk_id"]),
                        "page": chunks[h["chunk_id"]].get("page"),
                        "needed": h["chunk_id"] in wanted,
                        "text": " ".join(chunks[h["chunk_id"]]["text"].split())[:420],
                    }
                    for h in hits
                ],
            }
        out.append(
            {
                "id": qid,
                "question": question["question"],
                "answer": question["expected_answer"],
                "documents": question["required_document_count"],
                "hops": question["reasoning_hops"],
                "type": question["question_type"],
                "facts": facts,
                "methods": methods,
            }
        )

    aggregates = {
        f"{r['strategy']}@{r['k']}@{r['group']}": round(r["complete_evidence_rate"], 4)
        for r in summary["aggregates"]
    }
    payload = {
        "run": args.run.name,
        "split": summary["split"],
        "fingerprint": summary["evaluation_sha256"],
        "question_count": summary["question_count"],
        "aggregates": aggregates,
        "questions": sorted(out, key=lambda q: q["id"]),
    }
    args.output.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(
        json.dumps(
            {
                "questions": len(out),
                "bytes": args.output.stat().st_size,
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
