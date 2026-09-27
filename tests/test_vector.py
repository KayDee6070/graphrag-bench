import numpy as np
import pytest

from graphrag_bench.corpus import CorpusError
from graphrag_bench.ingestion.chunker import chunk_document
from graphrag_bench.models import Chunk, Document, text_sha256
from graphrag_bench.retrieval.vector import (
    RetrievalError,
    VectorIndex,
    build_vector_index,
    normalize_vectors,
)


def records(texts):
    documents = tuple(
        Document(
            document_id=f"doc-{i}",
            title=f"Document {i}",
            text=text,
            content_sha256=text_sha256(text),
            source_uri=f"{i}.txt",
        )
        for i, text in enumerate(texts)
    )
    chunks = tuple(
        Chunk(
            document_id=d.document_id,
            chunk_id=f"chunk-{i}",
            ordinal=0,
            text=d.text,
            start=0,
            end=len(d.text),
        )
        for i, d in enumerate(documents)
    )
    return documents, chunks


def test_exact_cosine_ranking_has_hand_calculated_scores_and_stable_ties(embedding_provider):
    _, chunks = records(("First source.", "Second source.", "Third source.", "Fourth source."))
    raw = np.array([[3, 4, 0], [0, 5, 0], [6, 8, 0], [-2, 0, 0]], dtype=np.float32)
    index = VectorIndex(chunks, normalize_vectors(raw, 4, 3), embedding_provider.spec)
    result = index.retrieve("test query", embedding_provider, top_k=20)
    assert [h.chunk_id for h in result.hits] == ["chunk-0", "chunk-2", "chunk-1", "chunk-3"]
    assert [h.score for h in result.hits] == pytest.approx([0.6, 0.6, 0, -1])
    assert all(hit.paths == () for hit in result.hits)
    assert result.elapsed_ms >= 0
    assert embedding_provider.query_calls == ["test query"]


def test_build_uses_only_chunk_text_in_canonical_order_and_reuses_index(embedding_provider):
    documents, chunks = records(("One.", "Another source."))
    index = build_vector_index(reversed(documents), reversed(chunks), embedding_provider)
    assert embedding_provider.document_calls == [("One.", "Another source.")]
    first = index.retrieve("same question", embedding_provider, top_k=1)
    second = index.retrieve("same question", embedding_provider, top_k=1)
    assert first.hits == second.hits
    assert len(embedding_provider.document_calls) == 1
    assert index.chunk(first.hits[0].chunk_id) in chunks
    assert index.chunks == chunks


def test_same_text_from_different_sources_keeps_both_ids(embedding_provider):
    docs, chunks = records(("Same quote.", "Same quote."))
    index = build_vector_index(docs, chunks, embedding_provider)
    result = index.retrieve("same", embedding_provider, top_k=5)
    assert [h.chunk_id for h in result.hits] == ["chunk-0", "chunk-1"]


@pytest.mark.parametrize(
    "values",
    [
        [[0, 0, 0]],
        [[np.nan, 1, 0]],
        [[np.inf, 0, 0]],
        [[1e100, 0, 0]],
        [[1, 2]],
        [1, 2, 3],
        [["1", "2", "3"]],
        [[1j, 0, 0]],
        [[True, False, True]],
    ],
)
def test_invalid_embeddings_rejected(values):
    with pytest.raises(RetrievalError):
        normalize_vectors(np.asarray(values), 1, 3)


@pytest.mark.parametrize(
    "query,top_k", [("", 5), (" \n", 5), (None, 5), ("q", 0), ("q", -1), ("q", True), ("q", 1.5)]
)
def test_invalid_requests_fail_before_encoding(query, top_k, embedding_provider):
    docs, chunks = records(("Source.",))
    index = build_vector_index(docs, chunks, embedding_provider)
    with pytest.raises(RetrievalError):
        index.retrieve(query, embedding_provider, top_k=top_k)
    assert embedding_provider.query_calls == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("revision", "1" * 40),
        ("model_id", "test/different"),
        ("dimensions", 4),
        ("versions", {"test": "2"}),
        ("settings", {"query_prefix": "different"}),
    ],
)
def test_incompatible_query_space_rejected_even_when_dimensions_match(
    field, value, embedding_provider
):
    docs, chunks = records(("Source.",))
    index = build_vector_index(docs, chunks, embedding_provider)
    embedding_provider.spec = embedding_provider.spec.model_copy(update={field: value})
    with pytest.raises(RetrievalError, match="does not match"):
        index.retrieve("question", embedding_provider)
    assert embedding_provider.query_calls == []


def test_wrong_query_shape_rejected(embedding_provider, monkeypatch):
    docs, chunks = records(("Source.",))
    index = build_vector_index(docs, chunks, embedding_provider)
    monkeypatch.setattr(embedding_provider, "embed_query", lambda _: np.ones((1, 3)))
    with pytest.raises(RetrievalError, match="wrong shape"):
        index.retrieve("question", embedding_provider)


def test_empty_index_returns_no_hits_without_query_embedding(embedding_provider):
    index = build_vector_index((), (), embedding_provider)
    assert index.retrieve("question", embedding_provider).hits == ()
    assert embedding_provider.query_calls == []


def test_invalid_chunk_provenance_rejected_before_encoding(embedding_provider):
    docs, chunks = records(("Source.",))
    broken = chunks[0].model_copy(update={"text": "Wrong!!"})
    with pytest.raises(CorpusError, match="source text mismatch"):
        build_vector_index(docs, (broken,), embedding_provider)
    assert embedding_provider.document_calls == []


def test_returned_vectors_and_specs_cannot_mutate_index(embedding_provider):
    docs, chunks = records(("Source.",))
    index = build_vector_index(docs, chunks, embedding_provider)
    original = index.vectors
    index.vectors[:] = 0
    index.spec.settings["query_prefix"] = "changed"
    assert np.array_equal(original, index.vectors)
    assert index.spec == embedding_provider.spec


def test_all_generated_fixture_chunks_are_indexed_without_graph_or_gold(corpus, embedding_provider):
    chunks = tuple(c for document in corpus.documents for c in chunk_document(document))
    index = build_vector_index(corpus.documents, chunks, embedding_provider)
    assert {c.chunk_id for c in index.chunks} == {c.chunk_id for c in chunks}
    assert {h.chunk_id for h in index.retrieve("query", embedding_provider, top_k=999).hits} == {
        c.chunk_id for c in chunks
    }
