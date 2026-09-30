"""Explain why multi-document questions score zero complete evidence.

Reads draft gold annotations deliberately: this is an oracle diagnostic, not a
retrieval run, and it writes no benchmark artifact. It answers three questions per
failing question: which source chunks would satisfy each annotated fact, how deep each
retriever ranks them, and whether graph query linking produced any seed at all.
"""

import argparse
import json
from pathlib import Path

from graphrag_bench.benchmark.config import load_benchmark_config
from graphrag_bench.benchmark.dataset import load_benchmark
from graphrag_bench.corpus import CorpusIndex
from graphrag_bench.embeddings.config import load_embedding_config
from graphrag_bench.embeddings.sentence_transformers import SentenceTransformerProvider
from graphrag_bench.extraction.llm.pipeline import read_llm_run
from graphrag_bench.ingestion.reader import load_ingestion
from graphrag_bench.retrieval.artifacts import load_vector_index
from graphrag_bench.retrieval.bm25 import BM25Retriever
from graphrag_bench.retrieval.graph import GraphRetriever

DEEP_K = 400


def single_chunks(fact, chunks) -> list[str]:
    """Chunks that on their own fully contain every span of one annotated fact."""
    return [
        chunk.chunk_id
        for chunk in chunks
        if all(
            span.document_id == chunk.document_id
            and chunk.start <= span.start
            and span.end <= chunk.end
            for span in fact.spans
        )
    ]


def covering_group(fact, chunks) -> list[str] | None:
    """Smallest greedy set of chunks whose merged text covers every span of one fact.

    Scoring merges adjacent selected fragments, so a span straddling a chunk boundary is
    satisfiable, but only when every chunk of the group is retrieved together.
    """
    group: set[str] = set()
    for span in fact.spans:
        overlapping = sorted(
            (
                c
                for c in chunks
                if c.document_id == span.document_id and c.start < span.end and span.start < c.end
            ),
            key=lambda c: c.start,
        )
        reached, chosen = span.start, []
        for chunk in overlapping:
            if chunk.start <= reached < chunk.end:
                chosen.append(chunk)
                reached = chunk.end
                if reached >= span.end:
                    break
        if reached < span.end:
            return None
        group.update(chunk.chunk_id for chunk in chosen)
    return sorted(group)


def rank_of(chunk_ids, hits) -> int | None:
    """Rank at which the shallowest listed chunk appears."""
    for position, hit in enumerate(hits, 1):
        if hit.chunk_id in chunk_ids:
            return position
    return None


def group_rank(group, hits) -> int | None:
    """Depth at which *all* chunks of a group have been seen; None if any is absent."""
    remaining = set(group or ())
    if not remaining:
        return None
    for position, hit in enumerate(hits, 1):
        remaining.discard(hit.chunk_id)
        if not remaining:
            return position
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--embedding-config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    source = args.bundle / "ingestion"
    batch, _ = load_ingestion(source)
    corpus = CorpusIndex(batch.documents, batch.chunks)
    chunks = tuple(corpus.chunks.values())
    documents = tuple(corpus.documents.values())
    config = load_benchmark_config(args.config)
    dataset = load_benchmark(
        source / "documents.jsonl", args.bundle / "questions.jsonl", split="dev"
    )
    index = load_vector_index(args.index, source)
    provider = SentenceTransformerProvider(load_embedding_config(args.embedding_config))
    bm25 = BM25Retriever(documents, chunks, config.bm25)
    _, _, graph = read_llm_run(args.graph, source)
    graph_retriever = GraphRetriever(graph, documents, chunks, config.graph)

    report = []
    for question in dataset.questions:
        if question.required_document_count < 2 and question.reasoning_hops < 2:
            continue
        vector_hits = index.retrieve(question.question, provider, top_k=DEEP_K).hits
        bm25_hits = bm25.retrieve(question.question, top_k=DEEP_K).hits
        graph_trace = graph_retriever.retrieve_with_trace(question.question, top_k=DEEP_K)
        row = {
            "question_id": question.question_id,
            "hops": question.reasoning_hops,
            "documents": question.required_document_count,
            "graph_seeds": len(graph_trace.links),
            "graph_hits": len(graph_trace.result.hits),
            "facts": [],
        }
        for evidence_set in question.sufficient_evidence_sets:
            for fact in evidence_set.facts:
                single = single_chunks(fact, chunks)
                group = covering_group(fact, chunks)
                needs_group = not single and group is not None
                row["facts"].append(
                    {
                        "fact_id": fact.fact_id,
                        "single_chunks": len(single),
                        "needs_adjacent_chunks": needs_group,
                        "group_size": len(group) if group else 0,
                        "coverable": bool(single or group),
                        "vector_rank": rank_of(single, vector_hits)
                        if single
                        else group_rank(group, vector_hits),
                        "bm25_rank": rank_of(single, bm25_hits)
                        if single
                        else group_rank(group, bm25_hits),
                        "graph_rank": rank_of(single, graph_trace.result.hits)
                        if single
                        else group_rank(group, graph_trace.result.hits),
                    }
                )
        report.append(row)

    facts = [f for r in report for f in r["facts"]]
    summary = {
        "kind": "multihop-oracle-diagnostic-v1",
        "uses_gold_annotations": True,
        "deep_k": DEEP_K,
        "questions": len(report),
        "facts": len(facts),
        "uncoverable_facts": sum(1 for f in facts if not f["coverable"]),
        "facts_needing_adjacent_chunks": sum(1 for f in facts if f["needs_adjacent_chunks"]),
        "facts_graph_reached": sum(1 for f in facts if f["graph_rank"] is not None),
        "questions_without_graph_seeds": sum(1 for r in report if r["graph_seeds"] == 0),
        "rows": report,
    }
    text = json.dumps(summary, indent=1, sort_keys=True)
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
