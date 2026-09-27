"""Explain rank fusion offline, then optionally inspect real vector + graph retrieval."""

import argparse
from pathlib import Path

from graphrag_bench.config import ChunkingConfig
from graphrag_bench.embeddings.config import load_embedding_config
from graphrag_bench.embeddings.sentence_transformers import SentenceTransformerProvider
from graphrag_bench.extraction import extract, load_rules
from graphrag_bench.graph import build_graph
from graphrag_bench.ingestion.chunker import chunk_document
from graphrag_bench.models import Document, RetrievalHit, RetrievalResult
from graphrag_bench.retrieval.graph import GraphRetriever
from graphrag_bench.retrieval.hybrid import HybridRetriever, fuse_rrf
from graphrag_bench.retrieval.vector import build_vector_index


def ranking(strategy: str, ids: tuple[str, ...]) -> RetrievalResult:
    return RetrievalResult(
        query="hand-written teaching example",
        strategy=strategy,
        hits=tuple(
            RetrievalHit(chunk_id=identifier, score=float(len(ids) - i))
            for i, identifier in enumerate(ids)
        ),
        elapsed_ms=0.0,
    )


def explain_math() -> None:
    print("1. Hand-written rankings, not neural results: vector=[a,b,c], graph=[b,d,a].")
    vector, graph = ranking("vector", ("a", "b", "c")), ranking("graph", ("b", "d", "a"))
    trace = fuse_rrf(vector, graph, rank_constant=60, top_k=2)
    print("  Each list adds 1 / (60 + rank). Ranks start at 1; missing means zero.")
    for detail in trace.ranking:
        print(
            f"  {detail.chunk_id}: vector rank={detail.vector_rank}, "
            f"graph rank={detail.graph_rank}; "
            f"{detail.vector_contribution:.8f} + {detail.graph_contribution:.8f} = "
            f"{detail.score:.8f}"
        )
    assert [hit.chunk_id for hit in trace.result.hits] == ["b", "a"]
    print("  Four unique candidates; only b and a survive final top_k=2.")
    print("  Raw cosine and graph scores are ignored; rank positions supply the votes.")

    print("\n2. Candidate windows differ from the final evidence limit.")
    vector, graph = ranking("vector", ("a", "b")), ranking("graph", ("c", "b"))
    fused = fuse_rrf(vector, graph, top_k=1)
    truncated = fuse_rrf(ranking("vector", ("a",)), ranking("graph", ("c",)), top_k=1)
    assert fused.result.hits[0].chunk_id == "b"
    assert truncated.result.hits[0].chunk_id == "a"
    print("  With two candidates per branch, agreement promotes b into final top 1.")
    print("  With one candidate per branch, b disappears before fusion; a wins the ID tie.")

    print("\n3. The rank constant changes how strongly agreement beats an isolated first place.")
    vector = ranking("vector", tuple(f"a{i:02d}" for i in range(1, 20)) + ("shared",))
    graph = ranking("graph", tuple(f"b{i:02d}" for i in range(1, 20)) + ("shared",))
    for constant in (10, 60, 100):
        hit = fuse_rrf(vector, graph, rank_constant=constant, top_k=1).result.hits[0]
        print(f"  c={constant}: winner={hit.chunk_id}, score={hit.score:.8f}")
        assert hit.chunk_id == ("a01" if constant == 10 else "shared")
    print("  This explains sensitivity; it does not select a winning constant for our benchmark.")

    print("\n4. Fusion can also lose useful evidence.")
    vector = ranking("vector", ("a-bridge", "b-dataset"))
    graph = ranking("graph", ("x-noise", "y-noise"))
    hits = fuse_rrf(vector, graph, top_k=2).result.hits
    assert [hit.chunk_id for hit in hits] == ["a-bridge", "x-noise"]
    print("  Suppose vector's two sources supply both required facts; graph's sources are noise.")
    print("  Fusion top 2: a-bridge, x-noise. The dataset fact drops out.")
    print("  Source IDs break exact ties; no answer labels enter the fusion function.")
    print("  A larger candidate union is not a guarantee of a better final evidence set.")


def semantic_example(root: Path, *, allow_download: bool) -> None:
    source = root / "datasets/fixtures/tiny/corpus/documents.jsonl"
    documents = tuple(
        Document.model_validate_json(line) for line in source.read_text().splitlines()
    )
    chunks = tuple(
        chunk
        for document in documents
        for chunk in chunk_document(document, ChunkingConfig(max_units=1, overlap_units=0))
    )
    extracted = extract(documents, chunks, load_rules(root / "configs/extraction.toml"))
    graph = GraphRetriever(
        build_graph(extracted.entities, extracted.relations, documents, chunks), documents, chunks
    )
    provider = SentenceTransformerProvider(
        load_embedding_config(root / "configs/embedding.toml"), allow_download=allow_download
    )
    vector = build_vector_index(documents, chunks, provider)
    hybrid = HybridRetriever(vector, graph, provider)
    print("\n5. Real pinned local model + extracted graph; 25 source chunks, no gold labels.")
    assert len(chunks) == 25
    print("  Config: c=60; vector_candidates=20; graph_candidates=20; final top_k=5.")
    for query in (
        "Which dataset is Birch evaluated on?",
        "Which dataset evaluates the model that Alder is based on?",
    ):
        trace = hybrid.retrieve_with_trace(query, top_k=5)
        print(f"\n  Query: {query}")
        print(
            f"  Vector candidates={len(trace.vector_result.hits)}, "
            f"graph candidates={len(trace.graph_trace.result.hits)}, "
            f"union={trace.candidate_chunks}"
        )
        for branch in (trace.vector_result, trace.graph_trace.result):
            print(f"  {branch.strategy} top 5:")
            for hit in branch.hits[:5]:
                print(f"    {hybrid.chunk(hit.chunk_id).text}")
        print("  Hybrid top 5:")
        for hit, detail in zip(trace.result.hits, trace.ranking[:5], strict=True):
            chunk = hybrid.chunk(hit.chunk_id)
            print(
                f"    score={hit.score:.8f}, vector rank={detail.vector_rank}, "
                f"graph rank={detail.graph_rank}: {chunk.text}"
            )
            print(f"      Source: {chunk.document_id} [{chunk.start}, {chunk.end})")
            for path in hit.paths:
                graph.validate_evidence_path(hit.chunk_id, path)
        for constant in (10, 60, 100):
            fused = fuse_rrf(
                trace.vector_result, trace.graph_trace.result, rank_constant=constant, top_k=5
            )
            print(
                f"  Same candidates, c={constant}: "
                + ", ".join(
                    f"{hybrid.chunk(hit.chunk_id).document_id}"
                    f"[{hybrid.chunk(hit.chunk_id).start}:{hybrid.chunk(hit.chunk_id).end})"
                    for hit in fused.result.hits
                )
            )
        for _ in range(2):
            assert hybrid.retrieve(query, top_k=5).hits == trace.result.hits
    print("\n  Three identical rankings per question; timings may differ.")
    print(
        "  Inspect successes and misses. M7 will evaluate evidence coverage; no quality claim here."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--semantic", action="store_true", help="use the real local embedding model"
    )
    parser.add_argument("--allow-download", action="store_true", help="fetch the pinned model")
    args = parser.parse_args()
    if args.allow_download and not args.semantic:
        parser.error("--allow-download requires --semantic")
    explain_math()
    if args.semantic:
        semantic_example(Path(__file__).resolve().parents[1], allow_download=args.allow_download)
    else:
        print("\nDefault demonstration is offline. Add --semantic after the M4 model setup.")


if __name__ == "__main__":
    main()
