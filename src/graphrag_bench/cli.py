"""Ingestion, graph construction, and local evidence retrieval commands."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

from pydantic import ValidationError

from graphrag_bench import __version__
from graphrag_bench.benchmark.config import load_benchmark_config
from graphrag_bench.benchmark.dataset import BenchmarkError, load_benchmark
from graphrag_bench.benchmark.pipeline import (
    run_benchmark,
    validate_run_output,
    verify_benchmark_run,
)
from graphrag_bench.config import ConfigurationError, load_chunking_config
from graphrag_bench.corpus import CorpusError
from graphrag_bench.embeddings.base import EmbeddingError
from graphrag_bench.embeddings.config import EmbeddingConfig, load_embedding_config
from graphrag_bench.embeddings.sentence_transformers import SentenceTransformerProvider
from graphrag_bench.extraction.config import ExtractionError, load_rules
from graphrag_bench.extraction.llm.config import load_llm_config
from graphrag_bench.extraction.llm.pipeline import (
    LLMGraphManifest,
    build_llm_graph_to_directory,
    replay_llm_graph,
    validate_llm_output,
)
from graphrag_bench.extraction.llm.provider import LocalTransformersProvider
from graphrag_bench.fixtures import FixtureError, load_fixture
from graphrag_bench.generation.config import GenerationError, load_generation_config
from graphrag_bench.generation.pipeline import (
    AnswerManifest,
    build_answer_to_directory,
    replay_answer_to_directory,
    validate_answer_output,
)
from graphrag_bench.generation.retrieval import retrieve_for_answer
from graphrag_bench.graph.builder import GraphError, KnowledgeGraph
from graphrag_bench.graph.pipeline import build_graph_to_directory, load_graph
from graphrag_bench.ingestion.pipeline import ingest_to_directory
from graphrag_bench.ingestion.reader import load_ingestion
from graphrag_bench.ingestion.types import IngestionError
from graphrag_bench.models import RetrievalResult
from graphrag_bench.retrieval.artifacts import (
    build_vector_to_directory,
    load_vector_index,
    validate_output,
)
from graphrag_bench.retrieval.bm25 import BM25Config, BM25Retriever
from graphrag_bench.retrieval.graph import GraphRetriever
from graphrag_bench.retrieval.graph_config import load_graph_retrieval_config
from graphrag_bench.retrieval.hybrid import HybridRetriever
from graphrag_bench.retrieval.hybrid_config import load_hybrid_retrieval_config
from graphrag_bench.retrieval.linking import QueryLink
from graphrag_bench.retrieval.vector import RetrievalError, validate_request


def _ingest(args: argparse.Namespace) -> dict[str, str | int]:
    overrides = {
        name: getattr(args, name)
        for name in ("max_chars", "max_units", "overlap_units")
        if getattr(args, name) is not None
    }
    manifest = ingest_to_directory(
        args.source,
        args.output,
        load_chunking_config(args.config, overrides=overrides),
    )
    return {
        "status": "ingested",
        "documents": manifest.document_count,
        "chunks": manifest.chunk_count,
        "ignored_files": len(manifest.ignored_files),
        "output": str(args.output),
    }


def _validate(path: Path) -> dict[str, str | int]:
    fixture = load_fixture(path)
    return {
        "documents": len(fixture.documents),
        "chunks": len(fixture.chunks),
        "entities": len(fixture.entities),
        "relations": len(fixture.relations),
        "questions": len(fixture.questions),
        "status": "valid",
    }


def _build_graph(args: argparse.Namespace) -> dict[str, str | int]:
    manifest = build_graph_to_directory(args.source, args.output, args.rules)
    return {
        "status": "built",
        "entities": manifest.entity_count,
        "assertions": manifest.assertion_count,
        "issues": manifest.issue_count,
        "output": str(args.output),
    }


def _index_vector(args: argparse.Namespace) -> dict:
    validate_output(args.source, args.output)
    # Fail on broken source artifacts before loading or downloading a model.
    load_ingestion(args.source)
    started = perf_counter()
    provider = SentenceTransformerProvider(
        load_embedding_config(args.config),
        allow_download=args.allow_download,
        cache_folder=args.cache_folder,
    )
    manifest = build_vector_to_directory(args.source, args.output, provider)
    return {
        "status": "indexed",
        "chunks": manifest.chunk_count,
        "dimensions": manifest.embedding.dimensions,
        "output": str(args.output),
        "indexing_ms": (perf_counter() - started) * 1000,
    }


def _build_llm_graph(args: argparse.Namespace) -> dict:
    validate_llm_output(args.source, args.output, args.response_cache)
    config = load_llm_config(args.config)
    load_ingestion(args.source)
    provider = LocalTransformersProvider(
        config.model, allow_download=args.allow_download, cache_folder=args.cache_folder
    )
    manifest = build_llm_graph_to_directory(
        args.source, args.output, config, provider, response_cache=args.response_cache
    )
    return _llm_summary(manifest, args.output)


def _llm_summary(manifest: LLMGraphManifest, output: Path) -> dict:
    return {
        "status": "built_with_issues" if manifest.issue_count else "built",
        "mode": manifest.execution_mode,
        "entities": manifest.entity_count,
        "assertions": manifest.assertion_count,
        "issues": manifest.issue_count,
        "rejected_chunks": len(manifest.rejected_chunks),
        "review_status": "unreviewed",
        "output": str(output),
    }


def _query_vector(args: argparse.Namespace) -> dict:
    validate_request(args.query, args.top_k)
    index = load_vector_index(args.index, args.source)
    provider = SentenceTransformerProvider(
        EmbeddingConfig.model_validate(index.spec.settings),
        allow_download=args.allow_download,
        cache_folder=args.cache_folder,
    )
    result = index.retrieve(args.query, provider, top_k=args.top_k)
    return {
        "result": result.model_dump(mode="json"),
        "evidence": [index.chunk(hit.chunk_id).model_dump(mode="json") for hit in result.hits],
        "embedding": index.spec.model_dump(mode="json"),
    }


def _answer(args: argparse.Namespace) -> dict:
    validate_answer_output(args.source, args.output, args.graph, args.index)
    config = load_generation_config(
        args.config,
        overrides={
            name: getattr(args, name)
            for name in ("strategy", "top_k")
            if getattr(args, name) is not None
        },
    )
    retrieval = retrieve_for_answer(
        args.source,
        args.query,
        config.retrieval,
        graph_directory=args.graph,
        index_directory=args.index,
        allow_download=args.allow_download,
        cache_folder=args.cache_folder,
    )
    provider = LocalTransformersProvider(
        config.model, allow_download=args.allow_download, cache_folder=args.cache_folder
    )
    return _answer_summary(
        build_answer_to_directory(args.source, args.output, retrieval, config, provider),
        args.output,
    )


def _answer_summary(manifest: AnswerManifest, output: Path) -> dict:
    return {
        "status": manifest.answer_status,
        "mode": manifest.execution_mode,
        "claims": manifest.claim_count,
        "citations": manifest.citation_count,
        "review_status": "unreviewed",
        "output": str(output),
        "answer_file": str(output / "answer.md"),
    }


def _query_bm25(args: argparse.Namespace) -> dict:
    validate_request(args.query, args.top_k)
    batch, _ = load_ingestion(args.source)
    retriever = BM25Retriever(batch.documents, batch.chunks, BM25Config(k1=args.k1, b=args.b))
    result = retriever.retrieve(args.query, top_k=args.top_k)
    chunks = {chunk.chunk_id: chunk for chunk in batch.chunks}
    return {
        "result": result.model_dump(mode="json"),
        "evidence": [chunks[hit.chunk_id].model_dump(mode="json") for hit in result.hits],
        "config": retriever.config.model_dump(mode="json"),
    }


def _query_graph(args: argparse.Namespace) -> dict:
    validate_request(args.query, args.top_k)
    overrides = {
        name: getattr(args, name)
        for name in ("max_hops", "direction")
        if getattr(args, name) is not None
    }
    config = load_graph_retrieval_config(args.config, overrides=overrides)
    batch, _ = load_ingestion(args.source)
    graph = load_graph(args.graph, args.source)
    retriever = GraphRetriever(graph, batch.documents, batch.chunks, config)
    trace = retriever.retrieve_with_trace(args.query, top_k=args.top_k)
    return (
        trace.model_dump(mode="json")
        | {
            "evidence": [
                retriever.chunk(hit.chunk_id).model_dump(mode="json") for hit in trace.result.hits
            ],
        }
        | _graph_audit(graph, trace.result, trace.links)
    )


def _graph_audit(
    graph: KnowledgeGraph, result: RetrievalResult, links: tuple[QueryLink, ...]
) -> dict:
    assertion_ids = {
        identifier for hit in result.hits for path in hit.paths for identifier in path.assertion_ids
    }
    entity_ids = {identifier for link in links for identifier in link.candidate_entity_ids}
    entity_ids.update(
        identifier for hit in result.hits for path in hit.paths for identifier in path.entity_ids
    )
    return {
        "assertions": [
            graph.assertion(identifier).model_dump(mode="json")
            for identifier in sorted(assertion_ids)
        ],
        "entities": [
            graph.entity(identifier).model_dump(mode="json") for identifier in sorted(entity_ids)
        ],
    }


def _query_hybrid(args: argparse.Namespace) -> dict:
    validate_request(args.query, args.top_k)
    overrides = {
        name: getattr(args, name)
        for name in ("rank_constant", "vector_candidates", "graph_candidates")
        if getattr(args, name) is not None
    }
    config = load_hybrid_retrieval_config(args.config, overrides=overrides)
    graph_config = load_graph_retrieval_config(args.graph_config)
    # Verify both indexes against the same ingestion artifacts before loading the model.
    batch, input_hashes = load_ingestion(args.source)
    graph = load_graph(args.graph, args.source)
    index = load_vector_index(args.index, args.source)
    graph_retriever = GraphRetriever(graph, batch.documents, batch.chunks, graph_config)
    provider = SentenceTransformerProvider(
        EmbeddingConfig.model_validate(index.spec.settings),
        allow_download=args.allow_download,
        cache_folder=args.cache_folder,
    )
    retriever = HybridRetriever(index, graph_retriever, provider, config)
    trace = retriever.retrieve_with_trace(args.query, top_k=args.top_k)
    return (
        trace.model_dump(mode="json")
        | {
            "evidence": [
                retriever.chunk(hit.chunk_id).model_dump(mode="json") for hit in trace.result.hits
            ],
            "embedding": index.spec.model_dump(mode="json"),
            "input_hashes": input_hashes,
            "package_version": __version__,
        }
        | _graph_audit(graph, trace.result, trace.graph_trace.links)
    )


def _benchmark(args: argparse.Namespace) -> dict:
    validate_run_output(args.output)
    config = load_benchmark_config(args.config)
    dataset = load_benchmark(args.documents, args.questions, split=args.split)
    rules = load_rules(args.rules)
    embedding = load_embedding_config(args.embedding_config)
    started = perf_counter()
    provider = SentenceTransformerProvider(
        embedding,
        allow_download=args.allow_download,
        cache_folder=args.cache_folder,
    )
    model_load_ms = (perf_counter() - started) * 1000
    summary = run_benchmark(
        dataset, args.output, config, rules, provider, model_load_ms=model_load_ms
    )
    return {
        "status": "benchmarked",
        "output": str(args.output),
        "questions": summary.question_count,
        "records": summary.record_count,
        "split": summary.split,
        "repeatable": summary.repeatable,
        "evaluation_sha256": summary.evaluation_sha256,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GraphRAG Bench development tools")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-fixtures", help="audit a fixture directory")
    validate.add_argument("path", type=Path)
    ingest = commands.add_parser(
        "ingest", help="parse and chunk a directory of UTF-8 text/Markdown"
    )
    ingest.add_argument("source", type=Path)
    ingest.add_argument("--output", type=Path, required=True, help="new output directory")
    ingest.add_argument("--config", type=Path, help="TOML file with a [chunking] table")
    ingest.add_argument("--max-chars", type=int, help="hard Unicode character limit per chunk")
    ingest.add_argument("--max-units", type=int, help="maximum sentence/fragment units per chunk")
    ingest.add_argument(
        "--overlap-units", type=int, help="requested overlap between adjacent windows"
    )
    graph = commands.add_parser("build-graph", help="extract a graph from M2 ingestion artifacts")
    graph.add_argument("source", type=Path, help="ingestion artifact directory")
    graph.add_argument("--output", type=Path, required=True, help="new output directory")
    graph.add_argument("--rules", type=Path, required=True, help="TOML extraction grammar")
    llm = commands.add_parser(
        "build-llm-graph", help="extract source-checked graph proposals with a local language model"
    )
    llm.add_argument("source", type=Path, help="verified ingestion directory")
    llm.add_argument("--config", type=Path, required=True, help="TOML model and ontology")
    llm.add_argument("--output", type=Path, required=True, help="new graph directory")
    llm.add_argument("--response-cache", type=Path, help="optional saved inference receipts")
    llm.add_argument("--allow-download", action="store_true", help="fetch the pinned local model")
    llm.add_argument("--cache-folder", type=Path, help="model cache directory")
    replay = commands.add_parser(
        "replay-llm-graph", help="rebuild an LLM graph from saved responses without inference"
    )
    replay.add_argument("graph", type=Path)
    replay.add_argument("--source", type=Path, required=True, help="original ingestion directory")
    replay.add_argument("--output", type=Path, required=True, help="new replay directory")
    vector = commands.add_parser("index-vector", help="embed verified ingestion chunks locally")
    vector.add_argument("source", type=Path, help="ingestion artifact directory")
    vector.add_argument("--output", type=Path, required=True, help="new index directory")
    vector.add_argument("--config", type=Path, required=True, help="TOML embedding configuration")
    query = commands.add_parser("query-vector", help="rank indexed chunks by cosine similarity")
    query.add_argument("index", type=Path)
    query.add_argument("--source", type=Path, required=True, help="original ingestion directory")
    query.add_argument("--query", required=True)
    query.add_argument("--top-k", type=int, default=5)
    hybrid = commands.add_parser("query-hybrid", help="fuse vector and graph ranks with RRF")
    hybrid.add_argument("index", type=Path, help="M4 vector index directory")
    hybrid.add_argument("--graph", type=Path, required=True, help="M3 graph artifact directory")
    hybrid.add_argument("--source", type=Path, required=True, help="original ingestion directory")
    hybrid.add_argument("--query", required=True)
    hybrid.add_argument("--top-k", type=int, default=5)
    hybrid.add_argument("--config", type=Path, help="TOML hybrid fusion settings")
    hybrid.add_argument("--graph-config", type=Path, help="TOML graph traversal settings")
    hybrid.add_argument("--rank-constant", type=int, help="override nonnegative RRF constant")
    hybrid.add_argument("--vector-candidates", type=int, help="override vector candidate count")
    hybrid.add_argument("--graph-candidates", type=int, help="override graph candidate count")
    for command in (vector, query, hybrid):
        command.add_argument(
            "--allow-download",
            action="store_true",
            help="allow fetching the pinned model; cached-only by default",
        )
        command.add_argument("--cache-folder", type=Path, help="optional local model cache")
    bm25 = commands.add_parser("query-bm25", help="lexical sanity baseline; no model required")
    bm25.add_argument("source", type=Path, help="ingestion artifact directory")
    bm25.add_argument("--query", required=True)
    bm25.add_argument("--top-k", type=int, default=5)
    bm25.add_argument("--k1", type=float, default=1.5)
    bm25.add_argument("--b", type=float, default=0.75)
    query_graph = commands.add_parser(
        "query-graph", help="link names and traverse source-backed edges"
    )
    query_graph.add_argument("graph", type=Path, help="M3 graph artifact directory")
    query_graph.add_argument(
        "--source", type=Path, required=True, help="original ingestion directory"
    )
    query_graph.add_argument("--query", required=True)
    query_graph.add_argument("--top-k", type=int, default=5)
    query_graph.add_argument("--config", type=Path, help="TOML graph traversal settings")
    query_graph.add_argument("--max-hops", type=int, help="override hop limit: 0, 1, or 2")
    query_graph.add_argument("--direction", choices=("outgoing", "incoming", "both"))
    benchmark = commands.add_parser(
        "benchmark", help="compare retrieval against annotated source evidence"
    )
    benchmark.add_argument("documents", type=Path, help="source Document JSONL, with original IDs")
    benchmark.add_argument("--questions", type=Path, required=True, help="BenchmarkQuestion JSONL")
    benchmark.add_argument("--split", choices=("fixture", "dev", "test"), required=True)
    benchmark.add_argument("--output", type=Path, required=True, help="new run directory")
    benchmark.add_argument("--config", type=Path, help="TOML comparison settings")
    benchmark.add_argument("--rules", type=Path, required=True, help="extraction grammar")
    benchmark.add_argument(
        "--embedding-config", type=Path, required=True, help="pinned model/tokenizer"
    )
    benchmark.add_argument(
        "--allow-download", action="store_true", help="fetch pinned model if needed"
    )
    benchmark.add_argument("--cache-folder", type=Path)
    verify_run = commands.add_parser(
        "verify-benchmark", help="verify saved run checksums without a model"
    )
    verify_run.add_argument("directory", type=Path)
    answer = commands.add_parser(
        "answer", help="generate an answer proposal with exact source citations"
    )
    answer.add_argument("source", type=Path, help="verified ingestion directory")
    answer.add_argument("--query", required=True)
    answer.add_argument(
        "--config", type=Path, required=True, help="TOML generation and retrieval settings"
    )
    answer.add_argument("--output", type=Path, required=True, help="new answer directory")
    answer.add_argument("--strategy", choices=("bm25", "vector", "graph", "hybrid"))
    answer.add_argument("--top-k", type=int)
    answer.add_argument("--graph", type=Path)
    answer.add_argument("--index", type=Path)
    answer.add_argument(
        "--allow-download", action="store_true", help="fetch pinned local models if needed"
    )
    answer.add_argument("--cache-folder", type=Path, help="local model cache")
    replay_answer = commands.add_parser(
        "replay-answer", help="replay selection and citation checks without models"
    )
    replay_answer.add_argument("directory", type=Path)
    replay_answer.add_argument("--source", type=Path, required=True)
    replay_answer.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "ingest":
            summary = _ingest(args)
        elif args.command == "build-graph":
            summary = _build_graph(args)
        elif args.command == "build-llm-graph":
            summary = _build_llm_graph(args)
        elif args.command == "replay-llm-graph":
            summary = _llm_summary(
                replay_llm_graph(args.graph, args.source, args.output), args.output
            )
        elif args.command == "index-vector":
            summary = _index_vector(args)
        elif args.command == "query-vector":
            summary = _query_vector(args)
        elif args.command == "query-bm25":
            summary = _query_bm25(args)
        elif args.command == "query-graph":
            summary = _query_graph(args)
        elif args.command == "query-hybrid":
            summary = _query_hybrid(args)
        elif args.command == "benchmark":
            summary = _benchmark(args)
        elif args.command == "verify-benchmark":
            checked = verify_benchmark_run(args.directory)
            summary = {
                "status": "valid",
                "questions": checked.question_count,
                "records": checked.record_count,
                "evaluation_sha256": checked.evaluation_sha256,
            }
        elif args.command == "answer":
            summary = _answer(args)
        elif args.command == "replay-answer":
            summary = _answer_summary(
                replay_answer_to_directory(args.directory, args.source, args.output), args.output
            )
        else:
            summary = _validate(args.path)
    except (
        FixtureError,
        ConfigurationError,
        IngestionError,
        ValidationError,
        CorpusError,
        ExtractionError,
        GraphError,
        EmbeddingError,
        RetrievalError,
        BenchmarkError,
        GenerationError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 0
