import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from graphrag_bench.cli import main
from graphrag_bench.ingestion.pipeline import ingest_to_directory
from graphrag_bench.retrieval.artifacts import build_vector_to_directory, load_vector_index
from graphrag_bench.retrieval.vector import RetrievalError

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def ingested_vectors(tmp_path):
    path = tmp_path / "input"
    ingest_to_directory(ROOT / "datasets/examples/ingestion", path)
    return path


def modify_manifest(directory, field, value):
    path = directory / "manifest.json"
    data = json.loads(path.read_text())
    data[field] = value
    path.write_text(json.dumps(data), encoding="utf-8")


def rehash(directory, filename):
    path = directory / "manifest.json"
    data = json.loads(path.read_text())
    data["artifact_hashes"][filename] = sha256((directory / filename).read_bytes()).hexdigest()
    path.write_text(json.dumps(data), encoding="utf-8")


def test_reproducible_artifacts_and_roundtrip_without_reembedding(
    ingested_vectors, tmp_path, embedding_provider
):
    first, second = tmp_path / "first", tmp_path / "second"
    build_vector_to_directory(ingested_vectors, first, embedding_provider)
    build_vector_to_directory(ingested_vectors, second, embedding_provider)
    for name in ("vectors.npy", "rows.json", "manifest.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    index = load_vector_index(first, ingested_vectors)
    again = load_vector_index(second, ingested_vectors)
    assert np.array_equal(index.vectors, again.vectors)
    assert (
        index.retrieve("question", embedding_provider).hits
        == again.retrieve("question", embedding_provider).hits
    )
    assert len(embedding_provider.document_calls) == 2


@pytest.mark.parametrize("filename", ["vectors.npy", "rows.json"])
def test_corruption_rejected(ingested_vectors, tmp_path, embedding_provider, filename):
    output = tmp_path / "index"
    build_vector_to_directory(ingested_vectors, output, embedding_provider)
    with (output / filename).open("ab") as stream:
        stream.write(b"modified")
    with pytest.raises(RetrievalError, match="checksum mismatch"):
        load_vector_index(output, ingested_vectors)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("nan", "finite"),
        ("zero", "unit length"),
        ("float64", "float32"),
        ("shape", "dimensions"),
        ("object", "Object arrays"),
        ("trailing", "exactly one"),
    ],
)
def test_rehashed_invalid_matrices_are_rejected(
    ingested_vectors, tmp_path, embedding_provider, mutation, match
):
    output = tmp_path / "index"
    build_vector_to_directory(ingested_vectors, output, embedding_provider)
    path = output / "vectors.npy"
    matrix = np.load(path, allow_pickle=False)
    if mutation == "nan":
        matrix[0, 0] = np.nan
    elif mutation == "zero":
        matrix[0] = 0
    elif mutation == "float64":
        matrix = matrix.astype(np.float64)
    elif mutation == "shape":
        matrix = matrix[:, :2]
    elif mutation == "object":
        matrix = matrix.astype(object)
    np.save(path, matrix)
    if mutation == "trailing":
        with path.open("ab") as stream:
            stream.write(b"extra")
    rehash(output, "vectors.npy")
    with pytest.raises(RetrievalError, match=match):
        load_vector_index(output, ingested_vectors)


def test_reordered_rows_cannot_reassign_vector_provenance(
    ingested_vectors, tmp_path, embedding_provider
):
    output = tmp_path / "index"
    build_vector_to_directory(ingested_vectors, output, embedding_provider)
    path = output / "rows.json"
    rows = json.loads(path.read_text())
    rows["chunk_ids"].reverse()
    path.write_text(json.dumps(rows), encoding="utf-8")
    rehash(output, "rows.json")
    with pytest.raises(RetrievalError, match="sorted order"):
        load_vector_index(output, ingested_vectors)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("chunk_count", 99, "count mismatch"),
        ("index_version", "unknown", "unsupported vector"),
        ("input_hashes", {}, "different ingestion"),
        ("artifact_hashes", {"../outside": "0" * 64}, "must hash"),
    ],
)
def test_inconsistent_manifests_rejected(
    ingested_vectors, tmp_path, embedding_provider, field, value, match
):
    output = tmp_path / "index"
    build_vector_to_directory(ingested_vectors, output, embedding_provider)
    modify_manifest(output, field, value)
    with pytest.raises(RetrievalError, match=match):
        load_vector_index(output, ingested_vectors)


def test_existing_output_is_not_modified_or_reembedded(
    ingested_vectors, tmp_path, embedding_provider
):
    output = tmp_path / "index"
    build_vector_to_directory(ingested_vectors, output, embedding_provider)
    before = (output / "vectors.npy").read_bytes()
    with pytest.raises(RetrievalError, match="output already exists"):
        build_vector_to_directory(ingested_vectors, output, embedding_provider)
    assert (output / "vectors.npy").read_bytes() == before
    assert len(embedding_provider.document_calls) == 1


def test_cli_indexes_and_returns_source_evidence_without_gold(
    ingested_vectors, tmp_path, embedding_provider, monkeypatch, capsys
):
    def no_gold(*args, **kwargs):
        raise AssertionError("retrieval must not read gold")

    monkeypatch.setattr("graphrag_bench.cli.load_fixture", no_gold)
    monkeypatch.setattr(
        "graphrag_bench.cli.SentenceTransformerProvider", lambda *args, **kwargs: embedding_provider
    )
    output = tmp_path / "index"
    assert (
        main(
            [
                "index-vector",
                str(ingested_vectors),
                "--output",
                str(output),
                "--config",
                str(ROOT / "configs/embedding.toml"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["chunks"] == 4
    query_args = [
        "query-vector",
        str(output),
        "--source",
        str(ingested_vectors),
        "--query",
        "Which model does Alder use?",
        "--top-k",
        "2",
    ]
    assert main(query_args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["result"]["hits"]) == len(payload["evidence"]) == 2
    assert [h["chunk_id"] for h in payload["result"]["hits"]] == [
        c["chunk_id"] for c in payload["evidence"]
    ]
    assert len(embedding_provider.document_calls) == 1
    assert main([*query_args[:-1], "0"]) == 1
    error = capsys.readouterr()
    assert "positive integer" in error.err and "Traceback" not in error.err


def test_cli_rejects_bad_source_before_loading_model(tmp_path, monkeypatch, capsys):
    def no_model(*args, **kwargs):
        raise AssertionError("invalid artifacts must be rejected before model loading")

    monkeypatch.setattr("graphrag_bench.cli.SentenceTransformerProvider", no_model)
    source = tmp_path / "missing"
    assert (
        main(
            [
                "index-vector",
                str(source),
                "--config",
                str(ROOT / "configs/embedding.toml"),
                "--output",
                str(tmp_path / "output"),
            ]
        )
        == 1
    )
    assert "manifest.json" in capsys.readouterr().err
    assert not (tmp_path / "output").exists()
