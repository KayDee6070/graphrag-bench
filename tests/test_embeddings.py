import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from graphrag_bench.embeddings.base import EmbeddingError
from graphrag_bench.embeddings.config import EmbeddingConfig, load_embedding_config
from graphrag_bench.embeddings.sentence_transformers import SentenceTransformerProvider


@pytest.fixture
def model_stub(monkeypatch):
    calls = []

    class Model:
        max_seq_length = 8

        def __init__(self, model_id, **kwargs):
            calls.append(("load", model_id, kwargs))

        def eval(self):
            calls.append(("eval",))

        def get_embedding_dimension(self):
            return 2

        def tokenizer(self, texts, **kwargs):
            calls.append(("tokenize", texts, kwargs))
            return {"input_ids": [[0] * (len(t.split()) + 2) for t in texts]}

        def encode_query(self, texts, **kwargs):
            calls.append(("query", texts, kwargs))
            return np.tile([1.0, 0.0], (len(texts), 1))

        def encode_document(self, texts, **kwargs):
            calls.append(("document", texts, kwargs))
            return np.tile([0.0, 1.0], (len(texts), 1))

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=lambda n: None))
    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Model)
    )
    monkeypatch.setattr("graphrag_bench.embeddings.sentence_transformers.version", lambda _: "test")
    return calls


def config(**updates):
    return EmbeddingConfig(model_id="test/model", revision="a" * 40, max_seq_length=8, **updates)


def test_pinned_offline_loading_and_asymmetric_encode_methods(model_stub):
    provider = SentenceTransformerProvider(
        config(query_prefix="question: ", document_prefix="text: ")
    )
    assert provider.embed_query("hello").tolist() == [1.0, 0.0]
    assert provider.embed_documents(("hello",)).tolist() == [[0.0, 1.0]]
    loading = model_stub[0][2]
    assert loading["local_files_only"] is True
    assert loading["revision"] == "a" * 40
    assert loading["trust_remote_code"] is False
    assert loading["model_kwargs"] == {"use_safetensors": True}
    assert loading["device"] == "cpu"
    query = next(call for call in model_stub if call[0] == "query")
    document = next(call for call in model_stub if call[0] == "document")
    assert query[2]["prompt"] == "question: "
    assert query[2]["processing_kwargs"] == {"text": {"truncation": False}}
    assert document[2]["prompt"] == "text: "
    assert provider.spec.dimensions == 2


def test_download_flag_is_explicit(model_stub, tmp_path):
    SentenceTransformerProvider(config(), allow_download=True, cache_folder=tmp_path)
    assert model_stub[0][2]["local_files_only"] is False
    assert model_stub[0][2]["cache_folder"] == str(tmp_path)


def test_provider_settings_cannot_change_after_specification_is_recorded(model_stub):
    provider = SentenceTransformerProvider(config())
    with pytest.raises(AttributeError):
        provider.config = config(query_prefix="changed: ")


def test_prefix_and_special_tokens_count_toward_limit(model_stub):
    provider = SentenceTransformerProvider(config(query_prefix="query prefix: "))
    provider.embed_query("one two three four")
    with pytest.raises(EmbeddingError, match="9 tokens"):
        provider.embed_query("one two three four five")
    assert sum(call[0] == "query" for call in model_stub) == 1
    assert all(call[2]["truncation"] is False for call in model_stub if call[0] == "tokenize")


def test_model_capacity_cannot_be_silently_increased(model_stub):
    values = config().model_dump() | {"max_seq_length": 9}
    with pytest.raises(EmbeddingError, match="exceeds"):
        SentenceTransformerProvider(EmbeddingConfig.model_validate(values))


def test_empty_documents_and_blank_inputs(model_stub):
    provider = SentenceTransformerProvider(config())
    assert provider.embed_documents(()).shape == (0, 2)
    with pytest.raises(EmbeddingError, match="nonblank"):
        provider.embed_query(" ")


def test_missing_optional_dependencies_explain_installation(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=lambda n: None))
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(EmbeddingError, match="optional embeddings dependencies"):
        SentenceTransformerProvider(config())


def test_mutable_model_revision_is_rejected():
    with pytest.raises(ValidationError):
        EmbeddingConfig(model_id="test/model", revision="main")


def test_example_config_and_invalid_toml(tmp_path):
    path = Path(__file__).resolve().parents[1] / "configs" / "embedding.toml"
    assert load_embedding_config(path).max_seq_length == 384
    invalid = tmp_path / "bad.toml"
    invalid.write_text('[wrong]\nmodel_id="example"\n', encoding="utf-8")
    with pytest.raises(EmbeddingError, match="embedding.*table"):
        load_embedding_config(invalid)
