import json
from hashlib import sha256
from pathlib import Path

import pytest
from pydantic import ValidationError

from graphrag_bench.cli import main
from graphrag_bench.generation.config import (
    GenerationConfig,
    GenerationError,
    load_generation_config,
)
from graphrag_bench.generation.pipeline import (
    build_answer_to_directory,
    read_answer_run,
    replay_answer_to_directory,
)
from graphrag_bench.generation.retrieval import retrieve_for_answer
from graphrag_bench.graph.pipeline import build_graph_to_directory
from graphrag_bench.retrieval.artifacts import build_vector_to_directory

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def saved_answer(answer_source, tmp_path, retrieval_receipt, generation_config, answer_provider):
    output = tmp_path / "answer"
    build_answer_to_directory(
        answer_source, output, retrieval_receipt, generation_config, answer_provider()
    )
    return output


def test_model_free_replay_preserves_all_artifact_bytes(saved_answer, answer_source, tmp_path):
    replay = tmp_path / "replay"
    before = read_answer_run(saved_answer, answer_source)[0]
    after = replay_answer_to_directory(saved_answer, answer_source, replay)
    assert after.execution_mode == "replay" and after.artifact_hashes == before.artifact_hashes
    for name in before.artifact_hashes:
        assert (saved_answer / name).read_bytes() == (replay / name).read_bytes()


@pytest.mark.parametrize(
    "filename", ["retrieval.json", "request.json", "receipt.json", "answer.json", "answer.md"]
)
def test_artifact_corruption_rejected(filename, saved_answer, answer_source):
    with (saved_answer / filename).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(GenerationError, match="checksum"):
        read_answer_run(saved_answer, answer_source)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("claim_count", 100, "count mismatch"),
        ("answer_status", "invalid_output", "status or count"),
        ("artifact_hashes", {"../bad": "0" * 64}, "file list"),
        ("input_hashes", {}, "different ingestion"),
        ("generation_version", "unknown", "generation_version"),
    ],
)
def test_manifest_consistency(field, value, match, saved_answer, answer_source):
    path = saved_answer / "manifest.json"
    data = json.loads(path.read_text())
    data[field] = value
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(GenerationError, match=match):
        read_answer_run(saved_answer, answer_source)


def test_rehashed_answer_must_match_raw_response(saved_answer, answer_source):
    path = saved_answer / "answer.json"
    data = json.loads(path.read_text())
    data["claims"][0]["text"] = "An edited conclusion."
    path.write_text(json.dumps(data), encoding="utf-8")
    manifest_path = saved_answer / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact_hashes"]["answer.json"] = sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(GenerationError, match="differs from replayed"):
        read_answer_run(saved_answer, answer_source)


def test_no_overwrite_or_writes_inside_sources(
    saved_answer, answer_source, retrieval_receipt, generation_config, answer_provider
):
    provider = answer_provider(AssertionError("no call expected"))
    for output in (saved_answer, answer_source / "answer"):
        with pytest.raises(GenerationError):
            build_answer_to_directory(
                answer_source, output, retrieval_receipt, generation_config, provider
            )
    assert not provider.requests
    with pytest.raises(GenerationError):
        replay_answer_to_directory(saved_answer, answer_source, saved_answer / "nested")


@pytest.mark.parametrize("strategy", ["bm25", "graph", "vector", "hybrid"])
def test_cli_all_retrievers_share_answer_pipeline_and_replay_without_models(
    strategy,
    answer_source,
    tmp_path,
    generation_config,
    answer_provider,
    embedding_provider,
    monkeypatch,
    capsys,
):
    graph, index = tmp_path / "graph", tmp_path / "index"
    if strategy in {"graph", "hybrid"}:
        build_graph_to_directory(answer_source, graph, ROOT / "configs/extraction.toml")
    if strategy in {"vector", "hybrid"}:
        build_vector_to_directory(answer_source, index, embedding_provider)
        monkeypatch.setattr(
            "graphrag_bench.generation.retrieval.SentenceTransformerProvider",
            lambda *a, **k: embedding_provider,
        )
    monkeypatch.setattr(
        "graphrag_bench.cli.LocalTransformersProvider", lambda *a, **k: answer_provider()
    )
    output = tmp_path / "generated"
    args = [
        "answer",
        str(answer_source),
        "--config",
        str(ROOT / "configs/generation.toml"),
        "--strategy",
        strategy,
        "--query",
        "Which dataset evaluates the model used by Orion?",
        "--output",
        str(output),
    ]
    if strategy in {"graph", "hybrid"}:
        args.extend(("--graph", str(graph)))
    if strategy in {"vector", "hybrid"}:
        args.extend(("--index", str(index)))
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "answered"
    manifest, retrieval, _, _, _ = read_answer_run(output, answer_source)
    assert retrieval.result.strategy == strategy
    assert len(retrieval.result.hits) == 2
    assert set(retrieval.index_hashes) == (
        {"graph", "vector"} if strategy == "hybrid" else {strategy} if strategy != "bm25" else set()
    )
    assert manifest.citation_count == 2

    def no_model(*a, **k):
        raise AssertionError("replay must not load a model")

    monkeypatch.setattr("graphrag_bench.cli.LocalTransformersProvider", no_model)
    assert (
        main(
            [
                "replay-answer",
                str(output),
                "--source",
                str(answer_source),
                "--output",
                str(tmp_path / "replay"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["mode"] == "replay"
    assert main(args) == 1
    assert "already exists" in capsys.readouterr().err


def test_missing_index_and_bad_source_fail_before_loading_models(
    answer_source, generation_config, monkeypatch, tmp_path
):
    def no_model(*a, **k):
        raise AssertionError("invalid inputs must fail before loading a model")

    monkeypatch.setattr("graphrag_bench.generation.retrieval.SentenceTransformerProvider", no_model)
    with pytest.raises(GenerationError, match="--graph"):
        retrieve_for_answer(answer_source, "Orion?", generation_config.retrieval)
    with pytest.raises(GenerationError, match="--index"):
        retrieve_for_answer(
            answer_source,
            "Orion?",
            generation_config.retrieval.model_copy(update={"strategy": "vector"}),
        )


@pytest.mark.parametrize("change", ["revision", "context", "strategy", "top_k"])
def test_invalid_configuration(change, generation_config):
    data = generation_config.model_dump(mode="json")
    if change == "revision":
        data["model"]["revision"] = "main"
    elif change == "context":
        data["context_tokens"] = data["model"]["max_input_tokens"]
    elif change == "strategy":
        data["retrieval"]["strategy"] = "unknown"
    else:
        data["retrieval"]["top_k"] = 0
    with pytest.raises(ValidationError):
        GenerationConfig.model_validate(data)


def test_wrong_config_table(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("[wrong]\nvalue = 1\n", encoding="utf-8")
    with pytest.raises(GenerationError, match="generation"):
        load_generation_config(path)
