"""Check draft questions against the corpus before they are accepted into a bundle.

`verify_questions.py` checks questions that are already prepared, where quotes have been
resolved to offsets. This checks the stage before that: raw drafts whose spans are still
(paper_id, page, quote) triples. It enforces the same rules `prepare-papers` would, so a
bad draft fails here in a second rather than after a full bundle build.

Checks, all exact:

  quote_found        the quote occurs in the named paper at all
  quote_unique       it occurs exactly once on that page, which prepare-papers requires
  page_correct       the page named is the page the quote is on
  chunks_required    how many chunks must be retrieved together to satisfy each fact
  hops_consistent    reasoning_hops equals the facts in the smallest sufficient set
  documents_consistent  required_document_count equals the papers that set spans
  ids_unique         no question_id or fact_id repeated, within the draft or against
                     the questions already shipped in the bundle

It does not judge whether a quote proves its statement. Nothing mechanical can.
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def load_corpus(bundle: Path) -> tuple[dict, dict, list]:
    source = bundle / "ingestion"
    documents = [json.loads(line) for line in (source / "documents.jsonl").open()]
    chunks = [json.loads(line) for line in (source / "chunks.jsonl").open()]
    catalog = json.loads((bundle / "catalog.json").read_text(encoding="utf-8"))
    titles = {paper["title"]: paper["paper_id"] for paper in catalog["papers"]}
    by_paper = {}
    for document in documents:
        paper_id = titles.get(document["title"])
        if paper_id is not None:
            by_paper[paper_id] = document
    return by_paper, titles, chunks


def pages_of(document: dict, chunks: list, start: int, end: int) -> set[int]:
    return {
        chunk["page"]
        for chunk in chunks
        if chunk["document_id"] == document["document_id"]
        and chunk["start"] < end
        and start < chunk["end"]
        and chunk.get("page") is not None
    }


def check_span(span: dict, by_paper: dict, chunks: list) -> tuple[list[str], tuple | None]:
    paper_id, quote = span["paper_id"], span["quote"]
    document = by_paper.get(paper_id)
    if document is None:
        return [f"unknown paper_id {paper_id!r}"], None
    occurrences = document["text"].count(quote)
    if occurrences == 0:
        return [f"quote not found in {paper_id}"], None
    start = document["text"].find(quote)
    end = start + len(quote)
    problems = []
    if occurrences > 1:
        problems.append(f"quote occurs {occurrences} times in {paper_id}; must be unique")
    pages = pages_of(document, chunks, start, end)
    if span["page"] not in pages:
        problems.append(
            f"quote in {paper_id} is on page(s) {sorted(pages) or 'unknown'}, "
            f"not the declared {span['page']}"
        )
    return problems, (document["document_id"], start, end)


def covering_run(resolved: list[tuple], chunks: list) -> list[str] | None:
    group: set[str] = set()
    for document_id, start, end in resolved:
        overlapping = sorted(
            (
                c
                for c in chunks
                if c["document_id"] == document_id and c["start"] < end and start < c["end"]
            ),
            key=lambda c: c["start"],
        )
        reached = start
        for chunk in overlapping:
            if chunk["start"] <= reached < chunk["end"]:
                group.add(chunk["chunk_id"])
                reached = chunk["end"]
                if reached >= end:
                    break
        if reached < end:
            return None
    return sorted(group)


def check_question(question: dict, by_paper: dict, chunks: list) -> dict:
    problems, runs = [], {}
    sets = question["sufficient_evidence_sets"]
    sizes = [len(facts) for facts in sets]
    smallest = sets[sizes.index(min(sizes))]
    papers = {span["paper_id"] for fact in smallest for span in fact["spans"]}
    for facts in sets:
        for fact in facts:
            resolved = []
            for span in fact["spans"]:
                span_problems, location = check_span(span, by_paper, chunks)
                problems += [f"{fact['fact_id']}: {p}" for p in span_problems]
                if location is not None:
                    resolved.append(location)
            if len(resolved) == len(fact["spans"]):
                run = covering_run(resolved, chunks)
                if run is None:
                    problems.append(f"{fact['fact_id']}: no run of adjacent chunks covers it")
                else:
                    runs[fact["fact_id"]] = len(run)
    if question["reasoning_hops"] != min(sizes):
        problems.append(
            f"reasoning_hops is {question['reasoning_hops']} but the smallest "
            f"sufficient set holds {min(sizes)} facts"
        )
    # AnnotationBook derives required_document_count and rejects it as an input, so a
    # draft may legitimately omit it. Check it only when a draft states it.
    declared_documents = question.get("required_document_count")
    if declared_documents is not None and declared_documents != len(papers):
        problems.append(
            f"required_document_count is {declared_documents} "
            f"but the smallest set spans {len(papers)} papers"
        )
    return {
        "question_id": question["question_id"],
        "problems": problems,
        "chunks_required": runs,
        "derived_document_count": len(papers),
        "papers": sorted(papers),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("drafts", type=Path, nargs="+", help="draft question JSON files")
    parser.add_argument("--bundle", type=Path, required=True, help="prepared paper bundle")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    by_paper, _, chunks = load_corpus(args.bundle)
    shipped = {json.loads(line)["question_id"] for line in (args.bundle / "questions.jsonl").open()}
    shipped_facts = {
        fact["fact_id"]
        for line in (args.bundle / "questions.jsonl").open()
        for evidence_set in json.loads(line)["sufficient_evidence_sets"]
        for fact in evidence_set["facts"]
    }

    questions = [q for path in args.drafts for q in json.loads(path.read_text())["questions"]]
    rows = [check_question(q, by_paper, chunks) for q in questions]

    ids = Counter(q["question_id"] for q in questions)
    fact_ids = Counter(
        fact["fact_id"]
        for q in questions
        for facts in q["sufficient_evidence_sets"]
        for fact in facts
    )
    global_problems = []
    for name, counts, existing in (
        ("question_id", ids, shipped),
        ("fact_id", fact_ids, shipped_facts),
    ):
        repeated = sorted(k for k, v in counts.items() if v > 1)
        clashing = sorted(set(counts) & existing)
        if repeated:
            global_problems.append(f"{name} repeated within the drafts: {repeated}")
        if clashing:
            global_problems.append(f"{name} already used in the bundle: {clashing}")

    failed = [r for r in rows if r["problems"]]
    summary = {
        "kind": "draft-question-checks-v1",
        "questions": len(rows),
        "failed": len(failed),
        "global_problems": global_problems,
        "facts_needing_adjacent_chunks": sorted(
            fact for row in rows for fact, n in row["chunks_required"].items() if n > 1
        ),
        "rows": rows,
    }
    text = json.dumps(summary, indent=1, sort_keys=True)
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1, sort_keys=True))
    for row in failed:
        for problem in row["problems"]:
            print(f"  FAIL {row['question_id']}: {problem}")
    raise SystemExit(1 if failed or global_problems else 0)


if __name__ == "__main__":
    main()
