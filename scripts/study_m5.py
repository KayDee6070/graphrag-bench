"""Demonstrate evidence traversal, hop ablations, ambiguity, and ranking limits offline."""

from pathlib import Path

from graphrag_bench.config import ChunkingConfig
from graphrag_bench.extraction import extract, load_rules
from graphrag_bench.graph import build_graph
from graphrag_bench.ingestion.chunker import chunk_document
from graphrag_bench.ingestion.parser import parse_bytes
from graphrag_bench.models import Document
from graphrag_bench.retrieval.graph import GraphRetriever
from graphrag_bench.retrieval.graph_config import GraphRetrievalConfig


def make_graph(documents, root):
    chunks = tuple(
        chunk
        for document in documents
        for chunk in chunk_document(document, ChunkingConfig(max_units=1, overlap_units=0))
    )
    extracted = extract(documents, chunks, load_rules(root / "configs/extraction.toml"))
    graph = build_graph(extracted.entities, extracted.relations, documents, chunks)
    return graph, chunks


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    texts = (
        "Orion is a retrieval method built on the Nova model.",
        "Nova is a language model evaluated on the Harbor dataset.",
    )
    documents = tuple(
        parse_bytes(text.encode(), relative_path=f"example-{i}.txt").document
        for i, text in enumerate(texts)
    )
    graph, chunks = make_graph(documents, root)
    query = "Which dataset evaluates the model that Orion is based on?"
    print("1. Minimal mechanism example: two statements in two source documents.")
    print("  Orion BASED_ON Nova; Nova EVALUATED_ON Harbor.")
    print("  The retriever sees only the question, source records, and extracted graph.")
    for hops in (0, 1, 2):
        config = GraphRetrievalConfig(max_hops=hops, direction="outgoing", include_mentions=False)
        search = GraphRetriever(graph, documents, chunks, config)
        trace = search.retrieve_with_trace(query, top_k=5)
        assert len(trace.result.hits) == hops
        print(f"  Relation evidence only, max_hops={hops}: {len(trace.result.hits)} chunks")
        if hops == 2:
            for hit in trace.result.hits:
                chunk = search.chunk(hit.chunk_id)
                print(f"    Quote: {chunk.text}")
                print(f"    Source: {chunk.document_id}, [{chunk.start}, {chunk.end})")
            path = next(
                path
                for hit in trace.result.hits
                for path in hit.paths
                if len(path.assertion_ids) == 2
            )
            graph.validate_path(path)
            print(
                "  Verified two-edge path:",
                " / ".join(graph.entity(i).name for i in path.entity_ids),
            )
    print("\n2. Mention expansion changes the meaning of a hop ablation.")
    search = GraphRetriever(graph, documents, chunks, GraphRetrievalConfig(max_hops=1))
    assert len(search.retrieve(query).hits) == 2
    print("  With mentions enabled, one edge reaches Nova and collects its other source mention.")
    print("  Both facts can therefore appear after one traversal edge; log this setting.")

    source = root / "datasets/fixtures/tiny/corpus/documents.jsonl"
    documents = tuple(
        Document.model_validate_json(line)
        for line in source.read_text(encoding="utf-8").splitlines()
    )
    graph, chunks = make_graph(documents, root)
    search = GraphRetriever(graph, documents, chunks)
    print("\n3. Ambiguous alias: keep candidates, never select one silently.")
    trace = search.retrieve_with_trace("Base", top_k=5)
    candidates = sorted(graph.entity(i).name for i in trace.links[0].candidate_entity_ids)
    assert candidates == ["Birch", "Elm"]
    print("  Base candidates:", ", ".join(candidates))

    print("\n4. Reaching evidence and ranking it into top K are different.")
    question = "Which dataset evaluates the model that Alder is based on?"
    trace = search.retrieve_with_trace(question, top_k=5)
    print(f"  Admitted paths: {trace.admitted_paths}; candidate chunks: {trace.candidate_chunks}")
    print(f"  Limits reached: {trace.limits_reached or 'none'}")
    for rank, (hit, detail) in enumerate(
        zip(trace.result.hits, trace.ranking, strict=True), start=1
    ):
        print(
            f"  {rank}. groups={len(detail.seed_groups)}, min_hops={detail.min_hops}: "
            f"{search.chunk(hit.chunk_id).text}"
        )
        for path in hit.paths:
            search.validate_evidence_path(hit.chunk_id, path)
    assert search.retrieve(question, top_k=5).hits == trace.result.hits
    print("  Ranking uses seed coverage, shortest path, then chunk ID; no question-intent model.")
    print(
        "  Inspect missing evidence: a connected graph alone does not guarantee a complete top K."
    )
    print("\nNo embeddings, gold labels, model downloads, or answer generation were used.")


if __name__ == "__main__":
    main()
