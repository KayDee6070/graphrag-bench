"""Trace a small graph back to source text. Reads source documents, never gold labels."""

from pathlib import Path

from graphrag_bench.config import ChunkingConfig
from graphrag_bench.extraction import extract, load_rules
from graphrag_bench.graph import build_graph
from graphrag_bench.ingestion.chunker import chunk_document
from graphrag_bench.models import Document


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    corpus_path = root / "datasets" / "fixtures" / "tiny" / "corpus" / "documents.jsonl"
    documents = tuple(
        Document.model_validate_json(line)
        for line in corpus_path.read_text(encoding="utf-8").splitlines()
    )
    config = ChunkingConfig(max_units=1, overlap_units=0)
    chunks = tuple(chunk for document in documents for chunk in chunk_document(document, config))
    rules = load_rules(root / "configs" / "extraction.toml")
    result = extract(documents, chunks, rules)
    graph = build_graph(result.entities, result.relations, documents, chunks)
    assert (len(documents), len(chunks), graph.node_count, graph.edge_count) == (8, 25, 15, 23)
    print("M3: turning explicit source statements into a map")
    print(f"Sources: {len(documents)} documents, {len(chunks)} generated chunks")
    print(f"Map: {graph.node_count} entities (nodes), {graph.edge_count} assertions (edges)")

    # This is a deliberately selected teaching example, not a question-answering algorithm.
    alder = graph.lookup("Alder")[0]
    first = graph.outgoing(alder.entity_id, "BASED_ON")[0]
    second = graph.outgoing(first.object_id, "EVALUATED_ON")[0]
    assert graph.entity(second.object_id).name == "Cedar"
    print("\n1. Follow two existing links across two documents:")
    by_document = {document.document_id: document for document in documents}
    by_chunk = {chunk.chunk_id: chunk for chunk in chunks}
    for relation in (first, second):
        print(
            f"  {graph.entity(relation.subject_id).name} --{relation.predicate}--> "
            f"{graph.entity(relation.object_id).name}"
        )
        for evidence in relation.evidence:
            assert (
                evidence.text
                == by_document[evidence.document_id].text[evidence.start : evidence.end]
            )
            chunk = by_chunk[evidence.chunk_id]
            assert chunk.start <= evidence.start < evidence.end <= chunk.end
            print(
                f"    Source: {evidence.document_id}, characters [{evidence.start}, {evidence.end})"
            )
            print(f"    Chunk: {evidence.chunk_id}")
            print(f"    Quote: {evidence.text}")

    print("\n2. Names need care:")
    assert graph.lookup("QZ")[0].name == "Quartz"
    print("  QZ identifies:", graph.lookup("QZ")[0].name)
    base = sorted(entity.name for entity in graph.lookup("Base"))
    assert base == ["Birch", "Elm"]
    print("  Base has two candidates:", ", ".join(base), "(no automatic choice)")

    print("\n3. Preserve separate sources for the same claim:")
    cedar = graph.lookup("Cedar")[0]
    repeated = graph.outgoing(cedar.entity_id, "REPORTS")
    assert len(repeated) == 2
    print("  Cedar REPORTS Accuracy has", len(repeated), "separate assertion edges.")
    print(
        "  Source documents:",
        ", ".join(
            sorted({span.document_id for relation in repeated for span in relation.evidence})
        ),
    )

    print("\n4. Unsupported statements remain visible:")
    assert len(result.issues) == 2
    for issue in result.issues:
        print(f"  {issue.code}: {issue.span.text}")
    print("  Numeric facts remain in the source chunks; M3 creates no numeric relations.")
    print("\nThese checks demonstrate construction and provenance, not retrieval quality.")


if __name__ == "__main__":
    main()
