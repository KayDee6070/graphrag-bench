"""Persist and verify portable graphs without pickle or gold benchmark inputs."""

import json
from hashlib import sha256
from pathlib import Path

import networkx as nx

from graphrag_bench import __version__
from graphrag_bench.corpus import CorpusIndex
from graphrag_bench.extraction.config import ExtractionRules, load_rules
from graphrag_bench.extraction.extractor import EXTRACTOR_VERSION, ExtractionIssue, extract
from graphrag_bench.graph.builder import (
    GRAPH_VERSION,
    GraphError,
    GraphSnapshot,
    KnowledgeGraph,
    build_graph,
)
from graphrag_bench.ingestion.reader import load_ingestion
from graphrag_bench.models import NonNegativeInt, Record, Sha256, Text
from graphrag_bench.serialization import json_bytes


class GraphManifest(Record):
    package_version: Text
    networkx_version: Text
    extractor_version: Text
    graph_version: Text
    rules: ExtractionRules
    rules_sha256: Sha256
    input_hashes: dict[str, Sha256]
    entity_count: NonNegativeInt
    assertion_count: NonNegativeInt
    issue_count: NonNegativeInt
    artifact_hashes: dict[str, Sha256]


def build_graph_to_directory(source: Path, output: Path, rules_path: Path) -> GraphManifest:
    if output.exists() or output.is_symlink():
        raise GraphError(f"output already exists; choose a new directory: {output}")
    if output.resolve().is_relative_to(source.resolve()):
        raise GraphError("output directory must be outside the ingestion directory")
    batch, input_hashes = load_ingestion(source)
    rules = load_rules(rules_path)
    result = extract(batch.documents, batch.chunks, rules)
    graph = build_graph(result.entities, result.relations, batch.documents, batch.chunks)
    artifacts = {
        "graph.json": json_bytes(graph.snapshot()),
        "issues.jsonl": b"".join(json_bytes(issue) for issue in result.issues),
    }
    manifest = GraphManifest(
        package_version=__version__,
        networkx_version=nx.__version__,
        extractor_version=EXTRACTOR_VERSION,
        graph_version=GRAPH_VERSION,
        rules=rules,
        rules_sha256=rules.fingerprint,
        input_hashes=input_hashes,
        entity_count=graph.node_count,
        assertion_count=graph.edge_count,
        issue_count=len(result.issues),
        artifact_hashes={name: sha256(content).hexdigest() for name, content in artifacts.items()},
    )
    try:
        output.mkdir(parents=True, exist_ok=False)
        for name, content in artifacts.items():
            with (output / name).open("xb") as stream:
                stream.write(content)
        with (output / "manifest.json").open("xb") as stream:
            stream.write(json_bytes(manifest))
    except OSError as error:
        raise GraphError(f"cannot write graph artifacts to {output}: {error}") from error
    return manifest


def load_graph(directory: Path, ingestion_directory: Path) -> KnowledgeGraph:
    """Verify artifact integrity and the original corpus before reconstructing the backend."""
    try:
        manifest = GraphManifest.model_validate_json((directory / "manifest.json").read_bytes())
        if manifest.graph_version != GRAPH_VERSION:
            raise GraphError(f"unsupported graph version: {manifest.graph_version}")
        if manifest.rules_sha256 != manifest.rules.fingerprint:
            raise GraphError("extraction rules checksum mismatch")
        expected = {"graph.json", "issues.jsonl"}
        if set(manifest.artifact_hashes) != expected:
            raise GraphError("graph manifest must hash graph.json and issues.jsonl")
        raw = {name: (directory / name).read_bytes() for name in sorted(expected)}
        if {n: sha256(b).hexdigest() for n, b in raw.items()} != manifest.artifact_hashes:
            raise GraphError("graph artifact checksum mismatch")
        batch, input_hashes = load_ingestion(ingestion_directory)
        if input_hashes != manifest.input_hashes:
            raise GraphError("graph was built from different ingestion artifacts")
        snapshot = GraphSnapshot.model_validate_json(raw["graph.json"])
        issues = tuple(
            ExtractionIssue.model_validate(json.loads(line))
            for line in raw["issues.jsonl"].decode("utf-8").splitlines()
        )
        if (len(snapshot.entities), len(snapshot.relations), len(issues)) != (
            manifest.entity_count,
            manifest.assertion_count,
            manifest.issue_count,
        ):
            raise GraphError("graph manifest count mismatch")
        corpus = CorpusIndex(batch.documents, batch.chunks)
        for issue in issues:
            corpus.validate_span(issue.span)
        expected_version = f"{manifest.extractor_version}:{manifest.rules_sha256}"
        if any(r.extraction_version != expected_version for r in snapshot.relations):
            raise GraphError("assertion extraction version differs from manifest")
        return build_graph(snapshot.entities, snapshot.relations, batch.documents, batch.chunks)
    except (OSError, UnicodeError, ValueError) as error:
        raise GraphError(f"cannot load graph artifacts from {directory}: {error}") from error
