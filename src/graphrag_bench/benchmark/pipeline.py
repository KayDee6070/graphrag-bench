"""Build from sources, evaluate separately, and save a complete inspectable run."""

import io
import platform
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Protocol
from uuid import uuid4

import numpy as np

from graphrag_bench import __version__
from graphrag_bench.benchmark.config import BenchmarkConfig
from graphrag_bench.benchmark.context import CONTEXT_VERSION, TokenCounter
from graphrag_bench.benchmark.dataset import BenchmarkDataset, BenchmarkError
from graphrag_bench.benchmark.metrics import METRICS_VERSION, score_evidence
from graphrag_bench.benchmark.report import BenchmarkSummary, render_report, summarize
from graphrag_bench.benchmark.runner import (
    BENCHMARK_VERSION,
    QuestionRun,
    Search,
    SearchObservation,
    evaluation_fingerprint,
    run_queries,
)
from graphrag_bench.corpus import CorpusIndex
from graphrag_bench.embeddings.base import EmbeddingProvider
from graphrag_bench.extraction import extract
from graphrag_bench.extraction.config import ExtractionRules
from graphrag_bench.extraction.extractor import EXTRACTOR_VERSION
from graphrag_bench.graph import build_graph
from graphrag_bench.graph.builder import GRAPH_VERSION
from graphrag_bench.ingestion.chunker import CHUNKER_VERSION, chunk_document
from graphrag_bench.models import Document, RunManifest, require_unique
from graphrag_bench.retrieval.bm25 import BM25Retriever
from graphrag_bench.retrieval.graph import GRAPH_RETRIEVAL_VERSION, GraphRetriever
from graphrag_bench.retrieval.hybrid import HYBRID_RETRIEVAL_VERSION, HybridRetriever
from graphrag_bench.retrieval.vector import VECTOR_VERSION, build_vector_index
from graphrag_bench.serialization import json_bytes


class BenchmarkProvider(EmbeddingProvider, TokenCounter, Protocol):
    """An encoder with a tokenizer sharing the recorded model specification."""


@dataclass(frozen=True)
class PreparedRetrieval:
    corpus: CorpusIndex
    searches: dict[str, Search]
    artifacts: dict[str, bytes]
    setup_ms: dict[str, float]
    issue_count: int


def prepare_retrieval(
    documents: tuple[Document, ...],
    config: BenchmarkConfig,
    rules: ExtractionRules,
    provider: BenchmarkProvider,
) -> PreparedRetrieval:
    """No benchmark question, answer, or gold graph enters construction."""
    started = perf_counter()
    chunks = tuple(
        chunk for document in documents for chunk in chunk_document(document, config.chunking)
    )
    corpus = CorpusIndex(documents, chunks)
    times = {"chunking_ms": (perf_counter() - started) * 1000}
    artifacts = {"chunks.jsonl": b"".join(json_bytes(chunk) for chunk in corpus.chunks.values())}
    searches: dict[str, Search] = {}
    graph, vector, issue_count = None, None, 0
    if {"graph", "hybrid"} & set(config.strategies):
        started = perf_counter()
        extraction = extract(documents, chunks, rules)
        built = build_graph(extraction.entities, extraction.relations, documents, chunks)
        graph = GraphRetriever(built, documents, chunks, config.graph)
        times["extraction_graph_ms"] = (perf_counter() - started) * 1000
        issue_count = len(extraction.issues)
        artifacts["graph.json"] = json_bytes(built.snapshot())
        artifacts["extraction_issues.jsonl"] = b"".join(json_bytes(i) for i in extraction.issues)

        def search_graph(query: str, k: int) -> SearchObservation:
            trace = graph.retrieve_with_trace(query, top_k=k)
            return SearchObservation(
                result=trace.result, diagnostics=trace.model_dump(mode="json", exclude={"result"})
            )

        if "graph" in config.strategies:
            searches["graph"] = search_graph
    if {"vector", "hybrid"} & set(config.strategies):
        started = perf_counter()
        vector = build_vector_index(documents, chunks, provider)
        times["embedding_index_ms"] = (perf_counter() - started) * 1000
        buffer = io.BytesIO()
        np.save(buffer, vector.vectors, allow_pickle=False)
        artifacts["vectors.npy"] = buffer.getvalue()

        def search_vector(query: str, k: int) -> SearchObservation:
            return SearchObservation(result=vector.retrieve(query, provider, top_k=k))

        if "vector" in config.strategies:
            searches["vector"] = search_vector
    if "hybrid" in config.strategies:
        hybrid = HybridRetriever(vector, graph, provider, config.hybrid)

        def search_hybrid(query: str, k: int) -> SearchObservation:
            trace = hybrid.retrieve_with_trace(query, top_k=k)
            return SearchObservation(
                result=trace.result, diagnostics=trace.model_dump(mode="json", exclude={"result"})
            )

        searches["hybrid"] = search_hybrid
    if "bm25" in config.strategies:
        started = perf_counter()
        lexical = BM25Retriever(documents, chunks, config.bm25)
        times["bm25_index_ms"] = (perf_counter() - started) * 1000
        searches["bm25"] = lambda query, k: SearchObservation(
            result=lexical.retrieve(query, top_k=k)
        )
    return PreparedRetrieval(corpus, searches, artifacts, times, issue_count)


def validate_run_output(output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise BenchmarkError(f"output already exists; choose a new directory: {output}")


def _code_provenance() -> dict:
    package = Path(__file__).resolve().parents[1]
    code = b"".join(
        path.relative_to(package).as_posix().encode() + b"\0" + path.read_bytes() + b"\0"
        for path in sorted(package.rglob("*.py"))
    )
    result = {
        "package_sources_sha256": sha256(code).hexdigest(),
        "git_commit": None,
        "git_dirty": None,
    }
    try:
        root = package.parents[1]
        if (root / ".git").exists():
            result["git_commit"] = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
            ).strip()
            result["git_dirty"] = bool(
                subprocess.check_output(
                    ["git", "-C", str(root), "status", "--porcelain"], text=True
                )
            )
    except (OSError, subprocess.CalledProcessError):
        pass
    return result


def run_benchmark(
    dataset: BenchmarkDataset,
    output: Path,
    config: BenchmarkConfig,
    rules: ExtractionRules,
    provider: BenchmarkProvider,
    *,
    model_load_ms: float = 0.0,
) -> BenchmarkSummary:
    validate_run_output(output)
    started = perf_counter()
    prepared = prepare_retrieval(dataset.documents, config, rules, provider)
    unreachable = tuple(
        q.question_id
        for q in dataset.questions
        if not score_evidence(q, prepared.corpus.chunks.values()).complete_evidence
    )
    records = run_queries(dataset, prepared.corpus, prepared.searches, provider, config)
    summary = summarize(
        dataset, records, unreachable=unreachable, extraction_issue_count=prepared.issue_count
    )
    artifacts = prepared.artifacts | {
        "documents.jsonl": b"".join(json_bytes(d) for d in dataset.documents),
        "questions.jsonl": b"".join(json_bytes(q) for q in dataset.questions),
        "results.jsonl": b"".join(json_bytes(record) for record in records),
        "summary.json": json_bytes(summary),
        "report.md": render_report(summary).encode("utf-8"),
    }
    provenance = _code_provenance()
    manifest = RunManifest(
        run_id=f"run-{uuid4().hex}",
        created_at=datetime.now(UTC),
        seed=config.seed,
        python_version=platform.python_version(),
        package_version=__version__,
        git_commit=provenance["git_commit"],
        corpus_sha256=dataset.corpus_sha256,
        benchmark_sha256=dataset.benchmark_sha256,
        config={
            "benchmark_version": BENCHMARK_VERSION,
            "metrics_version": METRICS_VERSION,
            "context_version": CONTEXT_VERSION,
            "chunker_version": CHUNKER_VERSION,
            "extractor_version": EXTRACTOR_VERSION,
            "graph_version": GRAPH_VERSION,
            "graph_retrieval_version": GRAPH_RETRIEVAL_VERSION,
            "vector_version": VECTOR_VERSION,
            "hybrid_version": HYBRID_RETRIEVAL_VERSION,
            "bm25_version": "unicode-word-bm25-v1",
            "benchmark": config.model_dump(mode="json"),
            "split": summary.split,
            "rules": rules.model_dump(mode="json"),
            "rules_sha256": rules.fingerprint,
            "embedding": provider.spec.model_dump(mode="json"),
            "tokenizer": {
                "model_id": provider.spec.model_id,
                "revision": provider.spec.revision,
                "add_special_tokens": False,
                "truncation": False,
                "prefix": "",
                "scope": "entire rendered evidence including source headers and separators",
            },
            "code": provenance,
            "machine": {
                "system": platform.system(),
                "machine": platform.machine(),
                "processor": platform.processor(),
            },
            "setup_ms": prepared.setup_ms | {"model_load_ms": model_load_ms},
            "build_and_evaluate_ms": (perf_counter() - started) * 1000,
            "warmup_queries_per_strategy": 1,
            "order": "seeded shuffle of strategies per question and repeat",
        },
        artifact_hashes={name: sha256(content).hexdigest() for name, content in artifacts.items()},
        provider_versions=provider.spec.versions
        | {name: version(name) for name in ("numpy", "networkx", "pydantic")},
    )
    try:
        output.mkdir(parents=True, exist_ok=False)
        for name, content in artifacts.items():
            with (output / name).open("xb") as stream:
                stream.write(content)
        with (output / "manifest.json").open("xb") as stream:
            stream.write(json_bytes(manifest))
    except OSError as error:
        raise BenchmarkError(f"cannot write benchmark run: {error}") from error
    return summary


def verify_benchmark_run(directory: Path) -> BenchmarkSummary:
    """Verify checksums and evaluation fingerprint without loading a neural model.

    This detects accidental corruption, not malicious rehashing or invalid scientific claims.
    It does not rerun retrieval or retokenize context.
    """
    required = {
        "documents.jsonl",
        "questions.jsonl",
        "chunks.jsonl",
        "results.jsonl",
        "summary.json",
        "report.md",
    }
    optional = {"graph.json", "extraction_issues.jsonl", "vectors.npy"}
    try:
        manifest = RunManifest.model_validate_json((directory / "manifest.json").read_bytes())
        if manifest.config.get("benchmark_version") != BENCHMARK_VERSION:
            raise BenchmarkError("unsupported benchmark version")
        names = set(manifest.artifact_hashes)
        if not required <= names or names - required - optional:
            raise BenchmarkError("unexpected benchmark artifact file list")
        raw = {name: (directory / name).read_bytes() for name in names}
        if {
            name: sha256(content).hexdigest() for name, content in raw.items()
        } != manifest.artifact_hashes:
            raise BenchmarkError("benchmark artifact checksum mismatch")
        summary = BenchmarkSummary.model_validate_json(raw["summary.json"])
        records = tuple(
            QuestionRun.model_validate_json(line) for line in raw["results.jsonl"].splitlines()
        )
        require_unique(
            tuple(f"{r.question_id}/{r.strategy}/{r.repeat}" for r in records), "question runs"
        )
        if (
            len(records) != summary.record_count
            or evaluation_fingerprint(records) != summary.evaluation_sha256
        ):
            raise BenchmarkError("benchmark result count or evaluation fingerprint mismatch")
        return summary
    except (OSError, ValueError) as error:
        raise BenchmarkError(f"cannot verify benchmark run: {error}") from error
