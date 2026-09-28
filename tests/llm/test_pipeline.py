import json
from hashlib import sha256
from pathlib import Path

import pytest
from pydantic import ValidationError

from graphrag_bench.cli import main
from graphrag_bench.extraction.llm.config import LLMError, LLMExtractionConfig, load_llm_config
from graphrag_bench.extraction.llm.extractor import extract_with_llm
from graphrag_bench.extraction.llm.pipeline import (
    build_llm_graph_to_directory,
    read_llm_run,
    replay_llm_graph,
)
from graphrag_bench.graph.pipeline import load_graph
from graphrag_bench.retrieval.artifacts import build_vector_to_directory

ROOT = Path(__file__).resolve().parents[2]


def test_cache_reuses_exact_requests_and_retains_inference_receipts(
    tmp_path, source_pair, llm_config, make_provider
):
    cache = tmp_path / "cache"
    first, second = make_provider(), make_provider(AssertionError("inference should not run"))
    left = extract_with_llm(*source_pair, llm_config, first, response_cache=cache)
    right = extract_with_llm(*source_pair, llm_config, second, response_cache=cache)
    assert left == right
    assert len(first.requests) == 2 and not second.requests


def test_cache_key_changes_with_prompt_ontology_and_model(
    tmp_path, source_pair, llm_config, make_provider
):
    cache = tmp_path / "cache"
    extract_with_llm(*source_pair, llm_config, make_provider(), response_cache=cache)
    changed = llm_config.model_copy(deep=True)
    changed.entity_types["Model"] = "A different description of a model."
    provider = make_provider(config=changed)
    extract_with_llm(*source_pair, changed, provider, response_cache=cache)
    assert len(provider.requests) == 2
    changed = changed.model_copy(
        update={"model": changed.model.model_copy(update={"revision": "f" * 40})}
    )
    provider = make_provider(config=changed)
    extract_with_llm(*source_pair, changed, provider, response_cache=cache)
    assert len(provider.requests) == 2
    assert len(list(cache.glob("*.json"))) == 6


def test_corrupt_cache_fails_without_regenerating(tmp_path, source_pair, llm_config, make_provider):
    cache = tmp_path / "cache"
    extract_with_llm(*source_pair, llm_config, make_provider(), response_cache=cache)
    for path in cache.glob("*.json"):
        data = json.loads(path.read_text())
        data["record"]["completion"]["text"] = "different response"
        path.write_text(json.dumps(data), encoding="utf-8")
    provider = make_provider()
    with pytest.raises(LLMError, match="invalid response cache"):
        extract_with_llm(*source_pair, llm_config, provider, response_cache=cache)
    assert not provider.requests


def test_run_replay_and_existing_graph_loader(llm_ingestion, tmp_path, llm_config, make_provider):
    output, replay = tmp_path / "graph", tmp_path / "replay"
    manifest = build_llm_graph_to_directory(llm_ingestion, output, llm_config, make_provider())
    assert (manifest.chunk_count, manifest.entity_count, manifest.assertion_count) == (2, 3, 2)
    graph = load_graph(output, llm_ingestion)
    assert graph.node_count == 3 and graph.edge_count == 2
    replay_manifest = replay_llm_graph(output, llm_ingestion, replay)
    assert replay_manifest.execution_mode == "replay"
    assert manifest.artifact_hashes == replay_manifest.artifact_hashes
    for name in manifest.artifact_hashes:
        assert (output / name).read_bytes() == (replay / name).read_bytes()


@pytest.mark.parametrize("filename", ["graph.json", "responses.jsonl", "issues.jsonl"])
def test_corruption_is_rejected(filename, llm_ingestion, tmp_path, llm_config, make_provider):
    output = tmp_path / "graph"
    build_llm_graph_to_directory(llm_ingestion, output, llm_config, make_provider())
    with (output / filename).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(LLMError, match="checksum mismatch"):
        read_llm_run(output, llm_ingestion)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("chunk_count", 7, "count mismatch"),
        ("rejected_chunks", ["nonexistent"], "count mismatch"),
        ("artifact_hashes", {"../file": "0" * 64}, "file list"),
        ("input_hashes", {}, "different ingestion"),
        ("graph_version", "unknown", "unsupported graph"),
        ("extraction_sha256", "0" * 64, "fingerprint mismatch"),
    ],
)
def test_manifest_consistency(
    field, value, match, llm_ingestion, tmp_path, llm_config, make_provider
):
    output = tmp_path / "graph"
    build_llm_graph_to_directory(llm_ingestion, output, llm_config, make_provider())
    path = output / "manifest.json"
    data = json.loads(path.read_text())
    data[field] = value
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(LLMError, match=match):
        read_llm_run(output, llm_ingestion)


def test_rehashed_graph_must_still_match_recorded_model_response(
    llm_ingestion, tmp_path, llm_config, make_provider
):
    output = tmp_path / "graph"
    build_llm_graph_to_directory(llm_ingestion, output, llm_config, make_provider())
    path = output / "graph.json"
    data = json.loads(path.read_text())
    data["relations"][0]["predicate"] = "DIFFERENT"
    path.write_text(json.dumps(data), encoding="utf-8")
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["artifact_hashes"]["graph.json"] = sha256(
        (output / "graph.json").read_bytes()
    ).hexdigest()
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(LLMError, match="differ from replayed"):
        read_llm_run(output, llm_ingestion)


def test_no_overwrite_or_outputs_in_ingestion(llm_ingestion, tmp_path, llm_config, make_provider):
    output = tmp_path / "graph"
    build_llm_graph_to_directory(llm_ingestion, output, llm_config, make_provider())
    provider = make_provider()
    with pytest.raises(LLMError, match="already exists"):
        build_llm_graph_to_directory(llm_ingestion, output, llm_config, provider)
    with pytest.raises(LLMError, match="outside the ingestion"):
        build_llm_graph_to_directory(llm_ingestion, llm_ingestion / "graph", llm_config, provider)
    with pytest.raises(LLMError, match="response cache"):
        build_llm_graph_to_directory(
            llm_ingestion,
            tmp_path / "other",
            llm_config,
            provider,
            response_cache=llm_ingestion / "cache",
        )
    assert not provider.requests


def test_cli_build_query_hybrid_and_model_free_replay(
    llm_ingestion, tmp_path, llm_config, make_provider, embedding_provider, monkeypatch, capsys
):
    output = tmp_path / "graph"
    monkeypatch.setattr(
        "graphrag_bench.cli.LocalTransformersProvider", lambda *a, **k: make_provider()
    )
    assert (
        main(
            [
                "build-llm-graph",
                str(llm_ingestion),
                "--config",
                str(ROOT / "configs/llm-extraction.toml"),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["assertions"] == 2 and result["review_status"] == "unreviewed"
    assert (
        main(["query-graph", str(output), "--source", str(llm_ingestion), "--query", "Orion"]) == 0
    )
    assert len(json.loads(capsys.readouterr().out)["evidence"]) == 2
    index = tmp_path / "vectors"
    build_vector_to_directory(llm_ingestion, index, embedding_provider)
    monkeypatch.setattr(
        "graphrag_bench.cli.SentenceTransformerProvider", lambda *a, **k: embedding_provider
    )
    assert (
        main(
            [
                "query-hybrid",
                str(index),
                "--graph",
                str(output),
                "--source",
                str(llm_ingestion),
                "--query",
                "Orion",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["result"]["strategy"] == "hybrid"

    def no_model(*args, **kwargs):
        raise AssertionError("model loading must not happen")

    monkeypatch.setattr("graphrag_bench.cli.LocalTransformersProvider", no_model)
    assert (
        main(
            [
                "replay-llm-graph",
                str(output),
                "--source",
                str(llm_ingestion),
                "--output",
                str(tmp_path / "replay"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["mode"] == "replay"
    assert (
        main(
            [
                "build-llm-graph",
                str(llm_ingestion),
                "--config",
                str(ROOT / "configs/llm-extraction.toml"),
                "--output",
                str(output),
            ]
        )
        == 1
    )
    assert "already exists" in capsys.readouterr().err


@pytest.mark.parametrize(
    "change", ["revision", "unknown_type", "duplicate_predicate", "zero_limit"]
)
def test_invalid_configuration(change, llm_config):
    data = llm_config.model_dump(mode="json")
    if change == "revision":
        data["model"]["revision"] = "main"
    elif change == "unknown_type":
        data["relations"][0]["subject_types"] = ["undeclared"]
    elif change == "duplicate_predicate":
        data["relations"].append(data["relations"][0])
    else:
        data["model"]["max_new_tokens"] = 0
    with pytest.raises(ValidationError):
        LLMExtractionConfig.model_validate(data)


def test_wrong_toml_table(tmp_path):
    path = tmp_path / "wrong.toml"
    path.write_text("[other]\nvalue = 1\n", encoding="utf-8")
    with pytest.raises(LLMError, match="llm_extraction"):
        load_llm_config(path)
