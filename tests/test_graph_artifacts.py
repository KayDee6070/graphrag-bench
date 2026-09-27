import json
from hashlib import sha256
from pathlib import Path

import pytest

from graphrag_bench.cli import main
from graphrag_bench.graph import GraphError
from graphrag_bench.graph.pipeline import build_graph_to_directory, load_graph
from graphrag_bench.ingestion.pipeline import ingest_to_directory
from graphrag_bench.ingestion.reader import load_ingestion
from graphrag_bench.ingestion.types import IngestionError

ROOT = Path(__file__).resolve().parents[1]
RULES = ROOT / "configs" / "extraction.toml"


@pytest.fixture
def ingested(tmp_path):
    output = tmp_path / "ingestion"
    ingest_to_directory(ROOT / "datasets" / "examples" / "ingestion", output)
    return output


def replace_manifest_field(directory, field, value):
    path = directory / "manifest.json"
    data = json.loads(path.read_text())
    data[field] = value
    path.write_text(json.dumps(data), encoding="utf-8")


def rehash(directory, filename):
    path = directory / "manifest.json"
    data = json.loads(path.read_text())
    data["artifact_hashes"][filename] = sha256((directory / filename).read_bytes()).hexdigest()
    path.write_text(json.dumps(data), encoding="utf-8")


def test_artifacts_are_reproducible_and_roundtrip_without_gold(ingested, tmp_path, monkeypatch):
    def reject_gold(*args, **kwargs):
        raise AssertionError("graph construction must not read gold")

    monkeypatch.setattr("graphrag_bench.cli.load_fixture", reject_gold)
    monkeypatch.setattr("graphrag_bench.fixtures.load_fixture", reject_gold)
    left, right = tmp_path / "left", tmp_path / "right"
    manifest = build_graph_to_directory(ingested, left, RULES)
    build_graph_to_directory(ingested, right, RULES)
    assert (manifest.entity_count, manifest.assertion_count, manifest.issue_count) == (7, 6, 0)
    for name in ("graph.json", "issues.jsonl", "manifest.json"):
        assert (left / name).read_bytes() == (right / name).read_bytes()
    graph = load_graph(left, ingested)
    assert graph.node_count == 7
    assert graph.edge_count == 6
    assert graph.lookup("Alder")[0].entity_type == "Method"


def test_build_graph_cli(ingested, tmp_path, capsys):
    output = tmp_path / "graph"
    args = ["build-graph", str(ingested), "--rules", str(RULES), "--output", str(output)]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out) == {
        "status": "built",
        "entities": 7,
        "assertions": 6,
        "issues": 0,
        "output": str(output),
    }
    previous = (output / "graph.json").read_bytes()
    assert main(args) == 1
    captured = capsys.readouterr()
    assert "output already exists" in captured.err
    assert "Traceback" not in captured.err
    assert (output / "graph.json").read_bytes() == previous


@pytest.mark.parametrize("filename", ["graph.json", "issues.jsonl"])
def test_graph_corruption_rejected(ingested, tmp_path, filename):
    output = tmp_path / "graph"
    build_graph_to_directory(ingested, output, RULES)
    with (output / filename).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(GraphError, match="checksum mismatch"):
        load_graph(output, ingested)


@pytest.mark.parametrize("filename", ["documents.jsonl", "chunks.jsonl"])
def test_ingestion_corruption_rejected_before_creating_graph(ingested, tmp_path, filename):
    with (ingested / filename).open("ab") as stream:
        stream.write(b" ")
    output = tmp_path / "graph"
    with pytest.raises(IngestionError, match="checksum mismatch"):
        build_graph_to_directory(ingested, output, RULES)
    assert not output.exists()


def test_missing_manifest_rejected_without_traceback(tmp_path, capsys):
    assert (
        main(
            [
                "build-graph",
                str(tmp_path),
                "--rules",
                str(RULES),
                "--output",
                str(tmp_path.parent / "unused-graph-output"),
            ]
        )
        == 1
    )
    assert "manifest.json" in capsys.readouterr().err


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("entity_count", 99, "count mismatch"),
        ("rules_sha256", "0" * 64, "rules checksum mismatch"),
        ("input_hashes", {}, "different ingestion artifacts"),
        ("artifact_hashes", {"../outside": "0" * 64}, "must hash"),
        ("graph_version", "unknown", "unsupported graph version"),
    ],
)
def test_inconsistent_graph_manifest_rejected(ingested, tmp_path, field, value, match):
    output = tmp_path / "graph"
    build_graph_to_directory(ingested, output, RULES)
    replace_manifest_field(output, field, value)
    with pytest.raises(GraphError, match=match):
        load_graph(output, ingested)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("document_count", 99, "count mismatch"),
        ("artifact_hashes", {}, "must hash"),
        ("sources", [], "at least 1"),
    ],
)
def test_inconsistent_ingestion_manifest_rejected(ingested, field, value, match):
    replace_manifest_field(ingested, field, value)
    with pytest.raises(IngestionError, match=match):
        load_ingestion(ingested)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("quote", "source text mismatch"),
        ("duplicate", "duplicate chunk IDs"),
        ("ordinal", "ordinals"),
    ],
)
def test_rehashed_but_invalid_chunks_rejected(ingested, mutation, match):
    path = ingested / "chunks.jsonl"
    chunks = [json.loads(line) for line in path.read_text().splitlines()]
    if mutation == "quote":
        chunks[0]["text"] = "X" * len(chunks[0]["text"])
    elif mutation == "duplicate":
        chunks[1]["chunk_id"] = chunks[0]["chunk_id"]
    else:
        chunks[0]["ordinal"] = 9
    path.write_text("".join(json.dumps(c) + "\n" for c in chunks), encoding="utf-8")
    rehash(ingested, "chunks.jsonl")
    with pytest.raises(IngestionError, match=match):
        load_ingestion(ingested)


def test_rehashed_graph_still_checks_provenance(ingested, tmp_path):
    output = tmp_path / "graph"
    build_graph_to_directory(ingested, output, RULES)
    path = output / "graph.json"
    graph = json.loads(path.read_text())
    evidence = graph["relations"][0]["evidence"][0]
    evidence["text"] = "x" * len(evidence["text"])
    path.write_text(json.dumps(graph), encoding="utf-8")
    rehash(output, "graph.json")
    with pytest.raises(GraphError, match="source text mismatch"):
        load_graph(output, ingested)


def test_output_cannot_be_inside_ingestion_directory(ingested):
    with pytest.raises(GraphError, match="outside the ingestion directory"):
        build_graph_to_directory(ingested, ingested / "graph", RULES)


def test_unsupported_corpus_exports_empty_graph_with_diagnostics(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "text.txt").write_text("This sentence has no supported relation.", encoding="utf-8")
    ingested = tmp_path / "ingestion"
    output = tmp_path / "graph"
    ingest_to_directory(source, ingested)
    manifest = build_graph_to_directory(ingested, output, RULES)
    assert (manifest.entity_count, manifest.assertion_count, manifest.issue_count) == (0, 0, 1)
    graph = load_graph(output, ingested)
    assert graph.node_count == graph.edge_count == 0
