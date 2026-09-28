"""Study the real-paper catalog and annotation boundary without PDFs or model weights."""

from pathlib import Path

from graphrag_bench.papers.annotations import load_annotations
from graphrag_bench.papers.catalog import load_catalog


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    catalog = load_catalog(root / "datasets/papers/pilot/catalog.json")
    annotations = load_annotations(root / "datasets/papers/pilot/annotations.json")
    print("M10: The Scouts receive real research reports")
    print("Corpus = reports the search team may read. Benchmark = questions and answer receipts.")
    print(
        f"Catalog: {len(catalog.papers)} papers; "
        f"{sum(p.pdf_bytes for p in catalog.papers):,} PDF bytes."
    )
    print(f"Benchmark: {len(annotations.questions)} questions; split={annotations.split}.")
    print(f"Review status: {annotations.review_status}.")
    print("\nEach report has a version, fingerprint, authors, and its own license:")
    for paper in catalog.papers:
        print(f"  {paper.paper_id}: {paper.arxiv_version}; {paper.license_id}")
    question = next(q for q in annotations.questions if q.question_id == "paper-c01")
    print(f"\nExample: {question.question}")
    print(f"Reference answer: {question.expected_answer}")
    for fact in question.sufficient_evidence_sets[0]:
        locator = fact.spans[0]
        print(f"  Required receipt: {locator.paper_id}, PDF page {locator.page}.")
        print(f"    {fact.statement}")
    print("One paper answers only half this comparison. Both receipts are required.")
    print("Two documents do not automatically mean two graph hops: these are two direct facts.")
    print("A bridge question instead follows Eren -> Lantern Squad -> east gate.")
    print("\nThe answer sheet is never used to build the searchable corpus.")
    print("Exact quotes can be checked by software. Their meaning still needs independent review.")
    print(
        "These 12 development questions cannot establish that graph retrieval beats vector search."
    )
    print("No PDFs downloaded, no model loaded, no experiment run by this lesson.")


if __name__ == "__main__":
    main()
