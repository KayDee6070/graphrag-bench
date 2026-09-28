"""Export LLM graphs with inference receipts; verify by replaying source validation."""

from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from platform import python_version
from typing import Literal

import networkx as nx

from graphrag_bench import __version__
from graphrag_bench.corpus import CorpusIndex
from graphrag_bench.extraction.llm.config import LLMError, LLMExtractionConfig
from graphrag_bench.extraction.llm.contracts import (
    EXTRACTOR_VERSION,
    ExtractionProvider,
    LLMExtractionResult,
    LLMIssue,
    ModelSpec,
    ResponseRecord,
)
from graphrag_bench.extraction.llm.extractor import extract_with_llm, process_responses
from graphrag_bench.extraction.llm.prompt import extraction_fingerprint
from graphrag_bench.graph.builder import GRAPH_VERSION, GraphSnapshot, KnowledgeGraph, build_graph
from graphrag_bench.ingestion.reader import load_ingestion
from graphrag_bench.models import Identifier, NonNegativeInt, Record, Sha256, Text
from graphrag_bench.serialization import json_bytes


class LLMGraphManifest(Record):
    package_version: Text
    python_version: Text
    pydantic_version: Text
    package_sources_sha256: Sha256
    networkx_version: Text
    extractor_version: Literal["llm-source-proposals-v1"] = EXTRACTOR_VERSION
    graph_version: Text
    execution_mode: Literal["inference", "replay"]
    config: LLMExtractionConfig
    provider: ModelSpec
    extraction_sha256: Sha256
    input_hashes: dict[str, Sha256]
    chunk_count: NonNegativeInt
    entity_count: NonNegativeInt
    assertion_count: NonNegativeInt
    issue_count: NonNegativeInt
    rejected_chunks: tuple[Identifier, ...]
    artifact_hashes: dict[str, Sha256]


def validate_llm_output(source: Path, output: Path, cache: Path | None = None) -> None:
    if output.exists() or output.is_symlink():
        raise LLMError(f"output already exists; choose a new directory: {output}")
    if output.resolve().is_relative_to(source.resolve()):
        raise LLMError("output directory must be outside the ingestion directory")
    if cache is not None and (
        cache.resolve().is_relative_to(source.resolve())
        or cache.resolve().is_relative_to(output.resolve())
        or (cache.exists() and not cache.is_dir())
    ):
        raise LLMError("response cache must be a directory outside ingestion and output")


def _write_run(
    output: Path,
    result: LLMExtractionResult,
    graph: KnowledgeGraph,
    config: LLMExtractionConfig,
    spec: ModelSpec,
    input_hashes: dict[str, str],
    *,
    mode: Literal["inference", "replay"],
) -> LLMGraphManifest:
    artifacts = {
        "graph.json": json_bytes(graph.snapshot()),
        "issues.jsonl": b"".join(json_bytes(issue) for issue in result.issues),
        "responses.jsonl": b"".join(json_bytes(record) for record in result.records),
    }
    manifest = LLMGraphManifest(
        package_version=__version__,
        python_version=python_version(),
        pydantic_version=version("pydantic"),
        package_sources_sha256=_package_fingerprint(),
        networkx_version=nx.__version__,
        graph_version=GRAPH_VERSION,
        execution_mode=mode,
        config=config,
        provider=spec,
        extraction_sha256=extraction_fingerprint(config, spec),
        input_hashes=input_hashes,
        chunk_count=len(result.records),
        entity_count=graph.node_count,
        assertion_count=graph.edge_count,
        issue_count=len(result.issues),
        rejected_chunks=result.rejected_chunks,
        artifact_hashes={name: sha256(raw).hexdigest() for name, raw in artifacts.items()},
    )
    try:
        output.mkdir(parents=True, exist_ok=False)
        for name, content in artifacts.items():
            with (output / name).open("xb") as stream:
                stream.write(content)
        with (output / "manifest.json").open("xb") as stream:
            stream.write(json_bytes(manifest))
    except OSError as error:
        raise LLMError(f"cannot write LLM graph artifacts: {error}") from error
    return manifest


def _package_fingerprint() -> str:
    package = Path(__file__).resolve().parents[2]
    return sha256(
        b"".join(
            path.relative_to(package).as_posix().encode("utf-8") + b"\0" + path.read_bytes() + b"\0"
            for path in sorted(package.rglob("*.py"))
        )
    ).hexdigest()


def build_llm_graph_to_directory(
    source: Path,
    output: Path,
    config: LLMExtractionConfig,
    provider: ExtractionProvider,
    *,
    response_cache: Path | None = None,
) -> LLMGraphManifest:
    validate_llm_output(source, output, response_cache)
    batch, input_hashes = load_ingestion(source)
    result = extract_with_llm(
        batch.documents, batch.chunks, config, provider, response_cache=response_cache
    )
    graph = build_graph(result.entities, result.relations, batch.documents, batch.chunks)
    return _write_run(output, result, graph, config, provider.spec, input_hashes, mode="inference")


def read_llm_run(
    directory: Path, source: Path
) -> tuple[LLMGraphManifest, LLMExtractionResult, KnowledgeGraph]:
    """Verify receipts and recompute the graph; no inference or semantic truth judgment."""
    try:
        manifest = LLMGraphManifest.model_validate_json((directory / "manifest.json").read_bytes())
        if manifest.graph_version != GRAPH_VERSION:
            raise LLMError("unsupported graph version")
        if manifest.extraction_sha256 != extraction_fingerprint(manifest.config, manifest.provider):
            raise LLMError("extraction recipe fingerprint mismatch")
        names = {"graph.json", "issues.jsonl", "responses.jsonl"}
        if set(manifest.artifact_hashes) != names:
            raise LLMError("unexpected LLM graph artifact file list")
        raw = {name: (directory / name).read_bytes() for name in names}
        if {
            name: sha256(value).hexdigest() for name, value in raw.items()
        } != manifest.artifact_hashes:
            raise LLMError("LLM graph artifact checksum mismatch")
        batch, input_hashes = load_ingestion(source)
        if input_hashes != manifest.input_hashes:
            raise LLMError("LLM graph was built from different ingestion artifacts")
        records = tuple(
            ResponseRecord.model_validate_json(line) for line in raw["responses.jsonl"].splitlines()
        )
        result = process_responses(
            CorpusIndex(batch.documents, batch.chunks), records, manifest.config, manifest.provider
        )
        snapshot = GraphSnapshot.model_validate_json(raw["graph.json"])
        issues = tuple(
            LLMIssue.model_validate_json(line) for line in raw["issues.jsonl"].splitlines()
        )
        if (
            snapshot != GraphSnapshot(entities=result.entities, relations=result.relations)
            or issues != result.issues
        ):
            raise LLMError("saved graph or issues differ from replayed model proposals")
        if (
            manifest.chunk_count,
            manifest.entity_count,
            manifest.assertion_count,
            manifest.issue_count,
            manifest.rejected_chunks,
        ) != (
            len(records),
            len(result.entities),
            len(result.relations),
            len(result.issues),
            result.rejected_chunks,
        ):
            raise LLMError("LLM graph manifest count mismatch")
        graph = build_graph(result.entities, result.relations, batch.documents, batch.chunks)
        return manifest, result, graph
    except (OSError, UnicodeError, ValueError, RecursionError) as error:
        raise LLMError(f"cannot load LLM graph artifacts: {error}") from error


def load_llm_graph(directory: Path, source: Path) -> KnowledgeGraph:
    return read_llm_run(directory, source)[2]


def replay_llm_graph(directory: Path, source: Path, output: Path) -> LLMGraphManifest:
    validate_llm_output(source, output)
    if output.resolve().is_relative_to(directory.resolve()):
        raise LLMError("replay output must be outside the original graph directory")
    manifest, result, graph = read_llm_run(directory, source)
    return _write_run(
        output,
        result,
        graph,
        manifest.config,
        manifest.provider,
        manifest.input_hashes,
        mode="replay",
    )
