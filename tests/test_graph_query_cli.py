import json
from pathlib import Path

from graphrag_bench.cli import main
from graphrag_bench.graph.pipeline import build_graph_to_directory
from graphrag_bench.ingestion.pipeline import ingest_to_directory

ROOT = Path(__file__).resolve().parents[1]


def test_graph_cli_returns_ranked_sources_links_and_valid_paths_without_model_or_gold(
    tmp_path, monkeypatch, capsys
):
    def forbidden(*args, **kwargs):
        raise AssertionError("graph retrieval must not use gold labels or embeddings")

    monkeypatch.setattr("graphrag_bench.cli.load_fixture", forbidden)
    monkeypatch.setattr("graphrag_bench.cli.SentenceTransformerProvider", forbidden)
    source = tmp_path / "source"
    source.mkdir()
    (source / "one.txt").write_text("Orion is a retrieval method built on the Nova model.")
    (source / "two.txt").write_text("Nova is a language model evaluated on the Harbor dataset.")
    ingested, graph = tmp_path / "ingested", tmp_path / "graph"
    ingest_to_directory(source, ingested)
    build_graph_to_directory(ingested, graph, ROOT / "configs/extraction.toml")
    arguments = [
        "query-graph",
        str(graph),
        "--source",
        str(ingested),
        "--query",
        "Orion",
        "--top-k",
        "5",
        "--config",
        str(ROOT / "configs/graph-retrieval.toml"),
    ]
    assert main(arguments) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["result"]["strategy"] == "graph"
    assert len(payload["result"]["hits"]) == len(payload["evidence"]) == 2
    assert [h["chunk_id"] for h in payload["result"]["hits"]] == [
        c["chunk_id"] for c in payload["evidence"]
    ]
    assert len(payload["assertions"]) == 2
    assert payload["links"][0]["surfaces"] == ["orion"]
    assert payload["limits_reached"] == []
    assert main([*arguments, "--max-hops", "0"]) == 0
    assert len(json.loads(capsys.readouterr().out)["result"]["hits"]) == 1
    assert main([*arguments, "--max-hops", "3"]) == 1
    error = capsys.readouterr()
    assert "max_hops" in error.err and "Traceback" not in error.err


def test_graph_cli_rejects_missing_artifacts_without_traceback(tmp_path, capsys):
    assert main(["query-graph", str(tmp_path), "--source", str(tmp_path), "--query", "Orion"]) == 1
    error = capsys.readouterr()
    assert "manifest.json" in error.err and "Traceback" not in error.err
