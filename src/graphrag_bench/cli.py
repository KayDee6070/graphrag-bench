"""Ingestion, graph construction, and local evidence retrieval commands."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

from pydantic import ValidationError

from graphrag_bench import __version__
from graphrag_bench.config import ConfigurationError, load_chunking_config
from graphrag_bench.corpus import CorpusError
from graphrag_bench.embeddings.base import EmbeddingError
from graphrag_bench.embeddings.config import EmbeddingConfig, load_embedding_config
from graphrag_bench.embeddings.sentence_transformers import SentenceTransformerProvider
from graphrag_bench.extraction.config import ExtractionError
from graphrag_bench.fixtures import FixtureError, load_fixture
from graphrag_bench.graph.builder import GraphError
from graphrag_bench.graph.pipeline import build_graph_to_directory, load_graph
from graphrag_bench.ingestion.pipeline import ingest_to_directory
from graphrag_bench.ingestion.reader import load_ingestion
from graphrag_bench.ingestion.types import IngestionError
from graphrag_bench.retrieval.artifacts import (
    build_vector_to_directory,
    load_vector_index,
    validate_output,
)
from graphrag_bench.retrieval.bm25 import BM25Config, BM25Retriever
from graphrag_bench.retrieval.graph import GraphRetriever
from graphrag_bench.retrieval.graph_config import load_graph_retrieval_config
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
    assertion_ids = {
        identifier
        for hit in trace.result.hits
        for path in hit.paths
        for identifier in path.assertion_ids
    }
    entity_ids = {identifier for link in trace.links for identifier in link.candidate_entity_ids}
    entity_ids.update(
        identifier
        for hit in trace.result.hits
        for path in hit.paths
        for identifier in path.entity_ids
    )
    return trace.model_dump(mode="json") | {
        "evidence": [
            retriever.chunk(hit.chunk_id).model_dump(mode="json") for hit in trace.result.hits
        ],
        "assertions": [
            graph.assertion(identifier).model_dump(mode="json")
            for identifier in sorted(assertion_ids)
        ],
        "entities": [
            graph.entity(identifier).model_dump(mode="json") for identifier in sorted(entity_ids)
        ],
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
    vector = commands.add_parser("index-vector", help="embed verified ingestion chunks locally")
    vector.add_argument("source", type=Path, help="ingestion artifact directory")
    vector.add_argument("--output", type=Path, required=True, help="new index directory")
    vector.add_argument("--config", type=Path, required=True, help="TOML embedding configuration")
    query = commands.add_parser("query-vector", help="rank indexed chunks by cosine similarity")
    query.add_argument("index", type=Path)
    query.add_argument("--source", type=Path, required=True, help="original ingestion directory")
    query.add_argument("--query", required=True)
    query.add_argument("--top-k", type=int, default=5)
    for command in (vector, query):
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
    args = parser.parse_args(argv)
    try:
        if args.command == "ingest":
            summary = _ingest(args)
        elif args.command == "build-graph":
            summary = _build_graph(args)
        elif args.command == "index-vector":
            summary = _index_vector(args)
        elif args.command == "query-vector":
            summary = _query_vector(args)
        elif args.command == "query-bm25":
            summary = _query_bm25(args)
        elif args.command == "query-graph":
            summary = _query_graph(args)
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
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 0
