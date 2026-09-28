"""Study the expanded development corpus; no downloads, models, or reviewer decisions."""

import argparse
from pathlib import Path

from graphrag_bench.benchmark.dataset import load_benchmark
from graphrag_bench.benchmark.metrics import score_evidence
from graphrag_bench.ingestion.reader import load_ingestion
from graphrag_bench.papers.annotations import load_annotations
from graphrag_bench.papers.catalog import load_catalog
from graphrag_bench.papers.review import audit_papers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, help="Optionally inspect a prepared local bundle")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    directory = root / "datasets/papers/research"
    catalog = load_catalog(directory / "catalog.json")
    book = load_annotations(directory / "annotations.json")
    question = next(q for q in book.questions if q.question_id == "paper-b02")

    print("MISSION 1: The library and the exam")
    print(
        f"Library: {len(catalog.papers)} real papers. Exam: {len(book.questions)} draft questions."
    )
    print("All questions are dev. Independent review is pending. Held-out questions: 0.\n")
    print("MISSION 2: Follow a bridge")
    print("Scouts: Eren -> Lantern Squad -> east gate.")
    print("Papers: REPLUG -> Contriever -> random cropping + 10% token deletion.")
    print(f"Question: {question.question}")
    for fact in question.sufficient_evidence_sets[0]:
        locator = fact.spans[0]
        paper = next(p for p in catalog.papers if p.paper_id == locator.paper_id)
        print(f"  {fact.statement}")
        print(f"  Source: {paper.pdf_url}#page={locator.page} ({paper.license_id})")
    print("The two-hop label is a candidate: another passage could contain the whole answer.\n")
    print("MISSION 3: Challenge the question")
    print("A rejected pooling question had a shortcut: REPLUG already describes mean pooling.")
    print("Two selected documents do not prove that two documents are necessary.")
    print("An exact quotation proves where text came from, not that its interpretation is right.\n")
    print("MISSION 4: Separate the checks")
    print("audit-papers: can all available chunks cover the reference evidence?")
    print("verify-paper-review: are the recorded decisions complete and bound to this bundle?")
    print("Neither authenticates a reviewer or measures retrieval accuracy.")
    if args.bundle:
        audit = audit_papers(args.bundle)
        print(f"\nLocal audit: {audit['chunks']} chunks; uncovered: {audit['uncovered_questions']}")
        batch, _ = load_ingestion(args.bundle / "ingestion")
        questions = load_benchmark(
            args.bundle / "ingestion/documents.jsonl",
            args.bundle / "questions.jsonl",
            split="dev",
        ).questions
        compiled = next(q for q in questions if q.question_id == question.question_id)
        first_document = compiled.sufficient_evidence_sets[0].facts[0].spans[0].document_id
        first_only = tuple(c for c in batch.chunks if c.document_id == first_document)
        for name, chunks in [("REPLUG only", first_only), ("All papers", batch.chunks)]:
            result = score_evidence(compiled, chunks)
            print(
                f"{name}: annotated evidence coverage={result.evidence_coverage:.0%}; "
                f"complete={result.complete_evidence}"
            )
    print("\nYour turn: why can 100% available-source coverage coexist with a bad retriever?")
    print("Answer: the facts exist in the library; the retriever still has to select them.")


if __name__ == "__main__":
    main()
