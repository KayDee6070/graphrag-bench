"""Compare frozen paper artifacts without rechunking, extraction, or index rebuilding."""

from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Literal

from pydantic import Field

from graphrag_bench import __version__
from graphrag_bench.benchmark.config import BenchmarkConfig
from graphrag_bench.benchmark.context import (
    CONTEXT_VERSION,
    ContextPiece,
    _uncovered,
    render_context,
)
from graphrag_bench.benchmark.dataset import BenchmarkDataset, BenchmarkError, load_benchmark
from graphrag_bench.benchmark.metrics import METRICS_VERSION, score_evidence
from graphrag_bench.benchmark.pipeline import (
    BenchmarkProvider,
    _code_provenance,
    validate_run_output,
)
from graphrag_bench.benchmark.report import BenchmarkSummary, render_report, summarize
from graphrag_bench.benchmark.runner import QuestionRun, Search, SearchObservation, run_queries
from graphrag_bench.corpus import CorpusIndex
from graphrag_bench.embeddings.base import EmbeddingSpec
from graphrag_bench.extraction.llm.pipeline import read_llm_run
from graphrag_bench.graph.builder import KnowledgeGraph
from graphrag_bench.ingestion.reader import load_ingestion
from graphrag_bench.models import JsonValue, NonNegativeInt, Record, Sha256, Text, require_unique
from graphrag_bench.papers.pipeline import BUNDLE_FILES, verify_papers
from graphrag_bench.retrieval.artifacts import load_vector_index
from graphrag_bench.retrieval.bm25 import BM25Retriever
from graphrag_bench.retrieval.graph import GraphRetriever
from graphrag_bench.retrieval.hybrid import HybridRetriever
from graphrag_bench.retrieval.vector import VectorIndex
from graphrag_bench.serialization import json_bytes

EXPERIMENT_VERSION = "frozen-paper-comparison-v1"
PAPER_FILES = BUNDLE_FILES | {"manifest.json"}
VECTOR_FILES = {"vectors.npy", "rows.json", "manifest.json"}
GRAPH_FILES = {"graph.json", "issues.jsonl", "responses.jsonl", "manifest.json"}
RESULT_FILES = {"results.jsonl", "summary.json", "report.md"}


class PaperExperimentManifest(Record):
    experiment_version: Literal["frozen-paper-comparison-v1"] = EXPERIMENT_VERSION
    package_version: Text
    created_at: datetime
    comparison: BenchmarkConfig
    embedding: EmbeddingSpec
    split: Literal["dev"] = "dev"
    annotation_status: Literal["pending-independent-review"] = "pending-independent-review"
    graph_status: Literal["unreviewed", "not-used"]
    metrics_version: Literal["complete-facts-source-coverage-v1"] = METRICS_VERSION
    context_version: Literal["source-union-greedy-v1"] = CONTEXT_VERSION
    source_hashes: dict[str, Sha256]
    extraction_issue_count: NonNegativeInt
    setup_ms: dict[str, float] = Field(default_factory=dict)
    code: dict[str, JsonValue]
    artifact_hashes: dict[str, Sha256]


@dataclass(frozen=True)
class PaperInputs:
    dataset: BenchmarkDataset
    corpus: CorpusIndex
    index: VectorIndex
    graph: KnowledgeGraph | None
    issue_count: int
    artifacts: dict[str, bytes]
    source_hashes: dict[str, str]


def _needs_graph(config: BenchmarkConfig) -> bool:
    return bool({"graph", "hybrid"} & set(config.strategies))


def _input_names(config: BenchmarkConfig) -> set[str]:
    return (
        {f"papers/{name}" for name in PAPER_FILES}
        | {f"vector/{name}" for name in VECTOR_FILES}
        | ({f"graph/{name}" for name in GRAPH_FILES} if _needs_graph(config) else set())
    )


def load_paper_inputs(
    bundle: Path,
    index_directory: Path,
    config: BenchmarkConfig,
    graph_directory: Path | None = None,
) -> PaperInputs:
    """Verify artifacts before loading weights; all input labels remain development data."""
    if _needs_graph(config) != (graph_directory is not None):
        raise BenchmarkError("--graph is required exactly when graph or hybrid is selected")
    verify_papers(bundle)
    source = bundle / "ingestion"
    batch, _ = load_ingestion(source)
    if config.chunking != batch.config:
        raise BenchmarkError("comparison chunking must match frozen paper chunks; no rechunking")
    index = load_vector_index(index_directory, source)
    graph, issue_count = None, 0
    directories = [("papers", bundle, PAPER_FILES), ("vector", index_directory, VECTOR_FILES)]
    if graph_directory is not None:
        manifest, _, graph = read_llm_run(graph_directory, source)
        issue_count = manifest.issue_count
        directories.append(("graph", graph_directory, GRAPH_FILES))
    artifacts = {
        f"{prefix}/{name}": (directory / name).read_bytes()
        for prefix, directory, names in directories
        for name in sorted(names)
    }
    hashes = {
        prefix: sha256(artifacts[f"{prefix}/manifest.json"]).hexdigest()
        for prefix, _, _ in directories
    }
    dataset = load_benchmark(source / "documents.jsonl", bundle / "questions.jsonl", split="dev")
    return PaperInputs(
        dataset,
        CorpusIndex(batch.documents, batch.chunks),
        index,
        graph,
        issue_count,
        artifacts,
        hashes,
    )


def paper_searches(
    inputs: PaperInputs, provider: BenchmarkProvider, config: BenchmarkConfig
) -> dict[str, Search]:
    """Bind source artifacts only. Reference answers never enter retriever constructors."""
    if provider.spec != inputs.index.spec:
        raise BenchmarkError("provider must match frozen index and comparison tokenizer")
    documents = tuple(inputs.corpus.documents.values())
    chunks = tuple(inputs.corpus.chunks.values())
    searches: dict[str, Search] = {}
    if "vector" in config.strategies:
        searches["vector"] = lambda query, k: SearchObservation(
            result=inputs.index.retrieve(query, provider, top_k=k)
        )
    graph = (
        GraphRetriever(inputs.graph, documents, chunks, config.graph)
        if inputs.graph is not None
        else None
    )

    def traced(retriever):
        def search(query: str, k: int) -> SearchObservation:
            result = retriever.retrieve_with_trace(query, top_k=k)
            return SearchObservation(
                result=result.result,
                diagnostics=result.model_dump(mode="json", exclude={"result"}),
            )

        return search

    if "graph" in config.strategies:
        searches["graph"] = traced(graph)
    if "hybrid" in config.strategies:
        searches["hybrid"] = traced(HybridRetriever(inputs.index, graph, provider, config.hybrid))
    if "bm25" in config.strategies:
        lexical = BM25Retriever(documents, chunks, config.bm25)
        searches["bm25"] = lambda query, k: SearchObservation(
            result=lexical.retrieve(query, top_k=k)
        )
    return searches


def _summary(inputs: PaperInputs, records: tuple[QuestionRun, ...]) -> BenchmarkSummary:
    unreachable = tuple(
        q.question_id
        for q in inputs.dataset.questions
        if not score_evidence(q, inputs.corpus.chunks.values()).complete_evidence
    )
    return summarize(
        inputs.dataset, records, unreachable=unreachable, extraction_issue_count=inputs.issue_count
    )


def _report(summary: BenchmarkSummary, config: BenchmarkConfig) -> str:
    return (
        "# Frozen real-paper development comparison\n\n"
        "Annotations: pending independent review. Split: dev. Held-out questions: 0.\n"
        f"Strategies: {', '.join(config.strategies)}. Graph assertions, if used, are unreviewed.\n"
        "Existing PDF chunks, vector rows, and model graph are reused without rebuilding.\n"
        "This is a development diagnostic; no general retrieval advantage is established.\n"
        "Loading and artifact validation are excluded from warm retrieval timings.\n\n"
        + render_report(summary)
    )


def run_paper_benchmark(
    bundle: Path,
    index_directory: Path,
    output: Path,
    config: BenchmarkConfig,
    provider: BenchmarkProvider,
    *,
    graph_directory: Path | None = None,
) -> BenchmarkSummary:
    validate_run_output(output)
    for directory in (bundle, index_directory, graph_directory):
        if directory is not None and output.resolve().is_relative_to(directory.resolve()):
            raise BenchmarkError("experiment output must be outside all input directories")
    started = perf_counter()
    inputs = load_paper_inputs(bundle, index_directory, config, graph_directory)
    searches = paper_searches(inputs, provider, config)
    setup_ms = {"validate_artifacts_and_bind_retrievers": (perf_counter() - started) * 1000}
    records = run_queries(inputs.dataset, inputs.corpus, searches, provider, config)
    summary = _summary(inputs, records)
    artifacts = inputs.artifacts | {
        "results.jsonl": b"".join(json_bytes(r) for r in records),
        "summary.json": json_bytes(summary),
        "report.md": _report(summary, config).encode(),
    }
    manifest = PaperExperimentManifest(
        package_version=__version__,
        created_at=datetime.now(UTC),
        comparison=config,
        embedding=provider.spec,
        graph_status="unreviewed" if inputs.graph is not None else "not-used",
        source_hashes=inputs.source_hashes,
        extraction_issue_count=inputs.issue_count,
        setup_ms=setup_ms,
        code=_code_provenance(),
        artifact_hashes={name: sha256(raw).hexdigest() for name, raw in artifacts.items()},
    )
    try:
        output.mkdir(parents=True, exist_ok=False)
        for name, raw in artifacts.items():
            path = output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(raw)
        with (output / "manifest.json").open("xb") as stream:
            stream.write(json_bytes(manifest))
    except OSError as error:
        raise BenchmarkError(f"cannot write paper experiment: {error}") from error
    return summary


def _validate_records(
    inputs: PaperInputs, records: tuple[QuestionRun, ...], config: BenchmarkConfig
) -> None:
    expected = {
        (q.question_id, strategy, repeat)
        for q in inputs.dataset.questions
        for strategy in config.strategies
        for repeat in range(config.repeats)
    }
    keys = tuple((r.question_id, r.strategy, r.repeat) for r in records)
    require_unique(keys, "question runs")
    if set(keys) != expected:
        raise BenchmarkError("paper results must cover every question, strategy, and repeat")
    questions = {q.question_id: q for q in inputs.dataset.questions}
    for row in records:
        question = questions[row.question_id]
        result = row.observation.result
        if (result.query, result.strategy) != (question.question, row.strategy):
            raise BenchmarkError("result query or strategy mismatch")
        if len(result.hits) > max(config.cutoffs):
            raise BenchmarkError("result exceeds configured cutoff")
        if tuple(e.k for e in row.evaluations) != config.cutoffs:
            raise BenchmarkError("result cutoffs differ from comparison")
        for hit in result.hits:
            if hit.chunk_id not in inputs.corpus.chunks:
                raise BenchmarkError("result contains unknown source chunk")
        for evaluation in row.evaluations:
            chunks = tuple(inputs.corpus.chunks[h.chunk_id] for h in result.hits[: evaluation.k])
            context = evaluation.context
            if (
                context.max_tokens != config.max_context_tokens
                or context.token_count > context.max_tokens
            ):
                raise BenchmarkError("context violates recorded token budget")
            # Replay source-union selection using recorded budget decisions. The model's
            # tokenizer is deliberately not required by offline verification.
            pieces: list[ContextPiece] = []
            selected, skipped_budget, skipped_overlap = [], [], []
            for chunk in chunks:
                uncovered = _uncovered(
                    chunk.start,
                    chunk.end,
                    ((p.start, p.end) for p in pieces if p.document_id == chunk.document_id),
                )
                if not uncovered:
                    skipped_overlap.append(chunk.chunk_id)
                elif chunk.chunk_id in context.skipped_budget:
                    skipped_budget.append(chunk.chunk_id)
                else:
                    selected.append(chunk.chunk_id)
                    pieces.extend(
                        ContextPiece(
                            chunk_id=chunk.chunk_id,
                            document_id=chunk.document_id,
                            start=start,
                            end=end,
                            text=chunk.text[start - chunk.start : end - chunk.start],
                        )
                        for start, end in uncovered
                    )
            if (
                tuple(pieces) != context.pieces
                or tuple(selected) != context.selected_chunk_ids
                or tuple(skipped_budget) != context.skipped_budget
                or tuple(skipped_overlap) != context.skipped_overlap
                or render_context(pieces) != context.text
            ):
                raise BenchmarkError("context selection differs from ranked source pieces")
            if evaluation.raw != score_evidence(
                question, chunks
            ) or evaluation.budgeted != score_evidence(question, context.pieces):
                raise BenchmarkError("saved evidence scores differ from source coverage")


def verify_paper_benchmark(directory: Path) -> BenchmarkSummary:
    """Rebuild graph and source metrics offline; do not rerun neural ranking or tokenization."""
    try:
        manifest = PaperExperimentManifest.model_validate_json(
            (directory / "manifest.json").read_bytes()
        )
        config = manifest.comparison
        names = _input_names(config) | RESULT_FILES
        if set(manifest.artifact_hashes) != names:
            raise BenchmarkError("unexpected paper experiment artifact file list")
        raw = {name: (directory / name).read_bytes() for name in names}
        if {
            name: sha256(value).hexdigest() for name, value in raw.items()
        } != manifest.artifact_hashes:
            raise BenchmarkError("paper experiment artifact checksum mismatch")
        inputs = load_paper_inputs(
            directory / "papers",
            directory / "vector",
            config,
            directory / "graph" if _needs_graph(config) else None,
        )
        if (
            manifest.embedding != inputs.index.spec
            or manifest.source_hashes != inputs.source_hashes
            or manifest.extraction_issue_count != inputs.issue_count
            or manifest.graph_status != ("unreviewed" if _needs_graph(config) else "not-used")
        ):
            raise BenchmarkError("paper experiment input provenance mismatch")
        records = tuple(
            QuestionRun.model_validate_json(line) for line in raw["results.jsonl"].splitlines()
        )
        _validate_records(inputs, records, config)
        summary = _summary(inputs, records)
        if (
            json_bytes(summary) != raw["summary.json"]
            or _report(summary, config).encode() != raw["report.md"]
        ):
            raise BenchmarkError("paper summary or report differs from recomputed results")
        return summary
    except (OSError, ValueError) as error:
        raise BenchmarkError(f"cannot verify paper experiment: {error}") from error
