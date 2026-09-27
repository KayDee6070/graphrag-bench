import json
from pathlib import Path

import pytest

from graphrag_bench.cli import main
from graphrag_bench.graph.pipeline import build_graph_to_directory
from graphrag_bench.ingestion.pipeline import ingest_to_directory
from graphrag_bench.retrieval.artifacts import build_vector_to_directory

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def artifacts(tmp_path, embedding_provider):
    source = tmp_path / "source"
    source.mkdir()
    (source / "one.txt").write_text("Orion is a retrieval method built on the Nova model.")
    (source / "two.txt").write_text("Nova is a language model evaluated on the Harbor dataset.")
    ingestion, graph, vector = (tmp_path / name for name in ("ingestion", "graph", "vector"))
    ingest_to_directory(source, ingestion)
    build_graph_to_directory(ingestion, graph, ROOT / "configs/extraction.toml")
    build_vector_to_directory(ingestion, vector, embedding_provider)
    return ingestion, graph, vector


def arguments(artifacts):
    ingestion, graph, vector = artifacts
    return [
        "query-hybrid",
        str(vector),
        "--source",
        str(ingestion),
        "--graph",
        str(graph),
        "--query",
        "Orion",
        "--top-k",
        "1",
    ]


def test_cli_fuses_verified_artifacts_and_limits_evidence_to_final_hits(
    artifacts, embedding_provider, monkeypatch, capsys
):
    def no_gold(*args, **kwargs):
        raise AssertionError("hybrid retrieval must not read gold labels")

    options = []

    def provider(config, **kwargs):
        assert config.model_dump(mode="json") == embedding_provider.spec.settings
        options.append(kwargs)
        return embedding_provider

    monkeypatch.setattr("graphrag_bench.cli.load_fixture", no_gold)
    monkeypatch.setattr("graphrag_bench.cli.SentenceTransformerProvider", provider)
    args = arguments(artifacts) + [
        "--config",
        str(ROOT / "configs/hybrid-retrieval.toml"),
        "--graph-config",
        str(ROOT / "configs/graph-retrieval.toml"),
        "--rank-constant",
        "10",
        "--vector-candidates",
        "1",
        "--graph-candidates",
        "2",
    ]
    assert main(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["result"]["strategy"] == "hybrid"
    assert len(payload["result"]["hits"]) == len(payload["evidence"]) == 1
    assert [h["chunk_id"] for h in payload["result"]["hits"]] == [
        c["chunk_id"] for c in payload["evidence"]
    ]
    assert len(payload["vector_result"]["hits"]) == 1
    assert len(payload["graph_trace"]["result"]["hits"]) == 2
    assert payload["config"]["rank_constant"] == 10
    assert payload["top_k"] == 1
    assert payload["candidate_chunks"] == 2
    assert payload["input_hashes"].keys() == {"manifest.json", "documents.jsonl", "chunks.jsonl"}
    used_assertions = {
        identifier
        for hit in payload["result"]["hits"]
        for path in hit["paths"]
        for identifier in path["assertion_ids"]
    }
    assert {r["assertion_id"] for r in payload["assertions"]} == used_assertions
    assert payload["embedding"] == embedding_provider.spec.model_dump(mode="json")
    assert options == [{"allow_download": False, "cache_folder": None}]
    assert len(embedding_provider.document_calls) == 1
    assert embedding_provider.query_calls == ["Orion"]


@pytest.mark.parametrize("broken", ["graph", "vector", "source", "config", "graph_config"])
def test_invalid_inputs_fail_before_loading_model(artifacts, broken, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid inputs must fail before model loading")

    monkeypatch.setattr("graphrag_bench.cli.SentenceTransformerProvider", forbidden)
    ingestion, graph, vector = artifacts
    args = arguments(artifacts)
    if broken == "graph":
        (graph / "graph.json").write_text("corrupt")
    elif broken == "vector":
        (vector / "vectors.npy").write_bytes(b"corrupt")
    elif broken == "source":
        (ingestion / "chunks.jsonl").write_text("corrupt")
    elif broken == "config":
        args += ["--vector-candidates", "0"]
    else:
        args += ["--graph-config", str(ingestion / "missing.toml")]
    assert main(args) == 1
    captured = capsys.readouterr()
    assert "error:" in captured.err and "Traceback" not in captured.err
    assert not captured.out


def test_different_artifact_sources_cannot_be_fused(artifacts, tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("mismatched artifacts must fail before model loading")

    monkeypatch.setattr("graphrag_bench.cli.SentenceTransformerProvider", forbidden)
    other_source = tmp_path / "other-source"
    other_source.mkdir()
    (other_source / "other.txt").write_text("Another unrelated source.")
    other_ingestion, other_graph = tmp_path / "other-ingestion", tmp_path / "other-graph"
    ingest_to_directory(other_source, other_ingestion)
    build_graph_to_directory(other_ingestion, other_graph, ROOT / "configs/extraction.toml")
    args = arguments(artifacts) + ["--graph", str(other_graph)]
    assert main(args) == 1
    captured = capsys.readouterr()
    assert "different ingestion" in captured.err and "Traceback" not in captured.err
