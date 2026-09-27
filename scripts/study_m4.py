"""Explain cosine search; optionally run the real local embedding model on source text."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from graphrag_bench.config import ChunkingConfig
from graphrag_bench.embeddings.config import load_embedding_config
from graphrag_bench.embeddings.sentence_transformers import SentenceTransformerProvider
from graphrag_bench.ingestion.chunker import chunk_document
from graphrag_bench.models import Document, RetrievalResult
from graphrag_bench.retrieval.bm25 import BM25Retriever
from graphrag_bench.retrieval.vector import build_vector_index, normalize_vectors


def explain_math() -> None:
    print("1. Hand-made vectors explain the arithmetic; these are NOT learned embeddings.")
    raw = np.array([[3, 4], [0, 5], [6, 8]], dtype=np.float32)
    vectors = normalize_vectors(raw, 3, 2)
    query = normalize_vectors(np.array([[1, 0]], dtype=np.float32), 1, 2)[0]
    scores = vectors @ query
    assert np.allclose(scores, [0.6, 0.0, 0.6])
    for original, normalized, score in zip(raw, vectors, scores, strict=True):
        display = "[" + ", ".join(f"{value:.2f}" for value in normalized) + "]"
        print(f"  {original.tolist()} normalized = {display}, cosine with [1, 0] = {score:.2f}")
    print("  [3, 4] and [6, 8] point the same way, so both score 0.60.")
    print("  0.60 is a similarity score, not a 60% probability of being correct.")


def show_result(result: RetrievalResult, chunks: dict) -> None:
    print(f"  Query: {result.query}")
    print(f"  Strategy: {result.strategy}; warm retrieval time: {result.elapsed_ms:.2f} ms")
    for rank, hit in enumerate(result.hits, start=1):
        chunk = chunks[hit.chunk_id]
        print(
            f"  {rank}. score={hit.score:.4f}, source={chunk.document_id} "
            f"[{chunk.start}, {chunk.end})"
        )
        print(f"     {chunk.text}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic", action="store_true", help="use the real local neural model")
    parser.add_argument("--allow-download", action="store_true", help="fetch the pinned model")
    args = parser.parse_args()
    if args.allow_download and not args.semantic:
        parser.error("--allow-download requires --semantic")
    explain_math()
    root = Path(__file__).resolve().parents[1]
    source = root / "datasets/fixtures/tiny/corpus/documents.jsonl"
    documents = tuple(
        Document.model_validate_json(line)
        for line in source.read_text(encoding="utf-8").splitlines()
    )
    config = ChunkingConfig(max_units=1, overlap_units=0)
    chunks = tuple(c for document in documents for c in chunk_document(document, config))
    by_chunk = {c.chunk_id: c for c in chunks}
    question = "Which dataset is Birch evaluated on?"
    print("\n2. BM25 matches words. It is a separate lexical baseline, not neural retrieval.")
    lexical = BM25Retriever(documents, chunks)
    show_result(lexical.retrieve(question, top_k=3), by_chunk)
    if not args.semantic:
        print("\n3. Run with --semantic after optional dependency/model setup for real embeddings.")
        print("  The default demonstration needs no model or network.")
        return
    print("\n3. Encode all 25 source chunks with the real local retrieval model.")
    started = perf_counter()
    provider = SentenceTransformerProvider(
        load_embedding_config(root / "configs/embedding.toml"), allow_download=args.allow_download
    )
    print(f"  Model load: {(perf_counter() - started) * 1000:.2f} ms")
    started = perf_counter()
    index = build_vector_index(documents, chunks, provider)
    print(f"  Chunk embedding/index build: {(perf_counter() - started) * 1000:.2f} ms")
    print(f"  Stored matrix: {index.vectors.shape}; model revision: {index.spec.revision}")
    assert index.vectors.shape == (25, 384)
    for query in (question, "Which dataset evaluates the model that Alder is based on?"):
        result = index.retrieve(query, provider, top_k=3)
        show_result(result, by_chunk)
        for _ in range(2):
            assert index.retrieve(query, provider, top_k=3).hits == result.hits
    print("  Repeated queries returned identical rankings and scores in this process.")
    print("  No graph edges or gold labels were used. Inspect whether all needed evidence appears.")
    print("  This demonstrates retrieval mechanics, not benchmark quality or GraphRAG superiority.")


if __name__ == "__main__":
    main()
