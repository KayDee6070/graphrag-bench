from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from graphrag_bench.embeddings.base import EmbeddingError
from graphrag_bench.extraction import extract, load_rules
from graphrag_bench.graph import build_graph
from graphrag_bench.ingestion.chunker import chunk_document
from graphrag_bench.ingestion.parser import parse_bytes
from graphrag_bench.models import RetrievalHit, RetrievalResult, TraversalPath
from graphrag_bench.retrieval.graph import GraphRetriever
from graphrag_bench.retrieval.graph_config import GraphRetrievalConfig
from graphrag_bench.retrieval.hybrid import HybridRetriever, HybridSearchTrace, fuse_rrf
from graphrag_bench.retrieval.hybrid_config import (
    HybridRetrievalConfig,
    load_hybrid_retrieval_config,
)
from graphrag_bench.retrieval.vector import RetrievalError, VectorIndex, build_vector_index

ROOT = Path(__file__).resolve().parents[1]


def result(strategy, ids, *, query="question", scores=None, paths=None):
    return RetrievalResult(
        query=query,
        strategy=strategy,
        hits=tuple(
            RetrievalHit(
                chunk_id=identifier,
                score=float(scores[i] if scores is not None else len(ids) - i),
                paths=(paths or {}).get(identifier, ()),
            )
            for i, identifier in enumerate(ids)
        ),
        elapsed_ms=0.0,
    )


@pytest.mark.parametrize("constant", [0, 10, 60, 100])
def test_rrf_matches_hand_arithmetic_and_exposes_every_candidate(constant):
    trace = fuse_rrf(
        result("vector", ("a", "b", "c")),
        result("graph", ("b", "d", "a")),
        rank_constant=constant,
        top_k=2,
    )
    expected = {
        "a": 1 / (constant + 1) + 1 / (constant + 3),
        "b": 1 / (constant + 2) + 1 / (constant + 1),
        "c": 1 / (constant + 3),
        "d": 1 / (constant + 2),
    }
    assert [h.chunk_id for h in trace.result.hits] == ["b", "a"]
    assert [d.chunk_id for d in trace.ranking] == ["b", "a", "d", "c"]
    assert {d.chunk_id: d.score for d in trace.ranking} == pytest.approx(expected)
    detail = trace.ranking[2]
    assert (detail.vector_rank, detail.graph_rank) == (None, 2)
    assert detail.vector_contribution == 0
    assert detail.graph_contribution == pytest.approx(1 / (constant + 2))
    assert trace.result.elapsed_ms >= 0


def test_raw_scores_have_no_effect_and_exact_ties_use_chunk_id():
    first = fuse_rrf(result("vector", ("b", "a")), result("graph", ("a", "b")))
    second = fuse_rrf(
        result("vector", ("b", "a"), scores=(-100, 1e100)),
        result("graph", ("a", "b"), scores=(0, -1e100)),
    )
    assert first.result.hits == second.result.hits
    assert [h.chunk_id for h in first.result.hits] == ["a", "b"]
    assert first.result.hits[0].score == first.result.hits[1].score


def test_final_cutoff_is_after_fusion_and_paths_are_preserved_and_deduplicated():
    short = TraversalPath(entity_ids=("orion", "nova"), assertion_ids=("r1",))
    long = TraversalPath(entity_ids=("orion", "nova", "harbor"), assertion_ids=("r1", "r2"))
    trace = fuse_rrf(
        result("vector", ("a", "b")),
        result("graph", ("c", "b"), paths={"b": (long, short, long)}),
        top_k=1,
    )
    assert len(trace.ranking) == 3
    assert trace.result.hits[0].chunk_id == "b"
    assert trace.result.hits[0].paths == (short, long)


@pytest.mark.parametrize("strategy", ["vector", "graph"])
def test_one_empty_branch_keeps_the_other_order_with_rrf_scores(strategy):
    vector = result("vector", ("z", "a") if strategy == "vector" else ())
    graph = result("graph", ("z", "a") if strategy == "graph" else ())
    trace = fuse_rrf(vector, graph, top_k=100)
    assert [h.chunk_id for h in trace.result.hits] == ["z", "a"]
    assert [h.score for h in trace.result.hits] == pytest.approx([1 / 61, 1 / 62])
    assert trace.result.strategy == "hybrid"


def test_empty_union_and_duplicate_input_contract():
    trace = fuse_rrf(result("vector", ()), result("graph", ()))
    assert trace.result.hits == trace.ranking == ()
    with pytest.raises(ValidationError, match="retrieved chunk IDs"):
        result("vector", ("a", "a"))


@pytest.mark.parametrize("constant", [-1, True, 1.5, "60"])
def test_invalid_constant_rejected(constant):
    with pytest.raises(RetrievalError, match="nonnegative integer"):
        fuse_rrf(result("vector", ()), result("graph", ()), rank_constant=constant)


def test_queries_and_branch_labels_must_match():
    with pytest.raises(RetrievalError, match="same query"):
        fuse_rrf(result("vector", ()), result("graph", (), query="different"))
    with pytest.raises(RetrievalError, match="one vector result and one graph"):
        fuse_rrf(result("vector", ()), result("vector", ()))


@pytest.fixture
def components(embedding_provider):
    texts = (
        "Orion is a retrieval method built on the Nova model.",
        "Nova is a language model evaluated on the Harbor dataset.",
        "This unrelated source has no extracted entity.",
    )
    documents = tuple(
        parse_bytes(text.encode(), relative_path=f"{i}.txt").document
        for i, text in enumerate(texts)
    )
    chunks = tuple(c for document in documents for c in chunk_document(document))
    extracted = extract(documents, chunks, load_rules(ROOT / "configs/extraction.toml"))
    graph = build_graph(extracted.entities, extracted.relations, documents, chunks)
    retriever = GraphRetriever(graph, documents, chunks)
    index = build_vector_index(documents, chunks, embedding_provider)
    return index, retriever, documents


def test_hybrid_uses_independent_windows_then_final_k_and_validates_sources(
    components, embedding_provider
):
    index, graph, _ = components
    config = HybridRetrievalConfig(vector_candidates=3, graph_candidates=2)
    retriever = HybridRetriever(index, graph, embedding_provider, config)
    trace = retriever.retrieve_with_trace("Orion", top_k=1)
    assert len(trace.result.hits) == 1
    assert len(trace.vector_result.hits) == 3
    assert len(trace.graph_trace.result.hits) == 2
    assert trace.candidate_chunks == len(trace.ranking) == 3
    assert trace.config == config
    assert trace.top_k == 1
    assert trace.result.hits[0].paths
    for hit in trace.result.hits:
        assert retriever.chunk(hit.chunk_id) == graph.chunk(hit.chunk_id)
        for path in hit.paths:
            graph.validate_evidence_path(hit.chunk_id, path)
    assert HybridSearchTrace.model_validate_json(trace.model_dump_json()) == trace
    assert embedding_provider.query_calls == ["Orion"]
    assert len(embedding_provider.document_calls) == 1
    assert trace.result.elapsed_ms >= (
        trace.vector_result.elapsed_ms + trace.graph_trace.result.elapsed_ms + trace.fusion_ms
    )
    for _ in range(2):
        assert retriever.retrieve("Orion", top_k=1).hits == trace.result.hits


def test_no_linked_name_is_a_valid_empty_graph_branch(components, embedding_provider):
    index, graph, _ = components
    trace = HybridRetriever(index, graph, embedding_provider).retrieve_with_trace("unlinked name")
    assert trace.graph_trace.links == trace.graph_trace.result.hits == ()
    assert [h.chunk_id for h in trace.result.hits] == [h.chunk_id for h in trace.vector_result.hits]
    assert all(h.paths == () for h in trace.result.hits)
    assert all(d.graph_rank is None for d in trace.ranking)


def test_candidate_window_excludes_paths_and_votes_outside_that_window(
    components, embedding_provider
):
    index, graph, _ = components
    trace = HybridRetriever(
        index, graph, embedding_provider, HybridRetrievalConfig(graph_candidates=1)
    ).retrieve_with_trace("Orion", top_k=100)
    graph_id = trace.graph_trace.result.hits[0].chunk_id
    assert len(trace.result.hits) == 3
    assert trace.graph_trace.candidate_chunks == 2
    assert all(bool(h.paths) == (h.chunk_id == graph_id) for h in trace.result.hits)
    assert sum(d.graph_rank is not None for d in trace.ranking) == 1


@pytest.mark.parametrize("mutation", ["subset", "same_id_changed_text", "metadata"])
def test_different_chunk_records_rejected_before_encoding(components, embedding_provider, mutation):
    index, graph, _ = components
    if mutation == "subset":
        chunks, vectors = index.chunks[:-1], index.vectors[:-1]
    else:
        update = (
            {"text": "X" * len(index.chunks[0].text)}
            if mutation == "same_id_changed_text"
            else {"section": "different"}
        )
        chunks = (index.chunks[0].model_copy(update=update), *index.chunks[1:])
        vectors = index.vectors
    other = VectorIndex(chunks, vectors, index.spec)
    with pytest.raises(RetrievalError, match="exactly the same source chunks"):
        HybridRetriever(other, graph, embedding_provider)
    assert embedding_provider.query_calls == []


def test_provider_mismatch_is_not_silent_fallback(components, embedding_provider):
    index, graph, _ = components
    embedding_provider.spec = embedding_provider.spec.model_copy(update={"revision": "1" * 40})
    with pytest.raises(RetrievalError, match="does not match"):
        HybridRetriever(index, graph, embedding_provider)
    assert embedding_provider.query_calls == []


def test_branch_errors_propagate_and_graph_seed_limits_fail_before_encoding(
    components, embedding_provider, monkeypatch
):
    index, graph, documents = components
    limited = GraphRetriever(
        graph.graph, documents, graph.chunks, GraphRetrievalConfig(max_seeds=1)
    )
    with pytest.raises(RetrievalError, match="query links 2 candidates"):
        HybridRetriever(index, limited, embedding_provider).retrieve("Orion and Nova")
    assert embedding_provider.query_calls == []

    def fail(_):
        raise EmbeddingError("encoder unavailable")

    monkeypatch.setattr(embedding_provider, "embed_query", fail)
    with pytest.raises(EmbeddingError, match="encoder unavailable"):
        HybridRetriever(index, graph, embedding_provider).retrieve("Orion")


@pytest.mark.parametrize("query,top_k", [(" ", 5), ("Orion", 0), ("Orion", True)])
def test_invalid_request_fails_before_encoding(components, embedding_provider, query, top_k):
    index, graph, _ = components
    with pytest.raises(RetrievalError):
        HybridRetriever(index, graph, embedding_provider).retrieve(query, top_k=top_k)
    assert embedding_provider.query_calls == []


def test_empty_corpus_requires_no_query_embedding_and_config_is_read_only(embedding_provider):
    vector = VectorIndex((), np.empty((0, 3), dtype=np.float32), embedding_provider.spec)
    graph = GraphRetriever(build_graph((), (), (), ()), (), ())
    retriever = HybridRetriever(vector, graph, embedding_provider)
    assert retriever.retrieve("anything").hits == ()
    assert embedding_provider.query_calls == []
    with pytest.raises(AttributeError):
        retriever.config = HybridRetrievalConfig(rank_constant=10)
    with pytest.raises(ValidationError):
        retriever.config.rank_constant = 10


@pytest.mark.parametrize(
    "settings",
    [
        {"rank_constant": -1},
        {"rank_constant": True},
        {"rank_constant": "60"},
        {"vector_candidates": 0},
        {"graph_candidates": 1.5},
        {"method": "weighted"},
        {"unknown": 1},
    ],
)
def test_invalid_configuration_rejected(settings):
    with pytest.raises(ValidationError):
        HybridRetrievalConfig.model_validate(settings)


def test_config_file_and_explicit_overrides(tmp_path):
    config = load_hybrid_retrieval_config(ROOT / "configs/hybrid-retrieval.toml")
    assert config == HybridRetrievalConfig()
    assert load_hybrid_retrieval_config(overrides={"rank_constant": 10}).rank_constant == 10
    path = tmp_path / "bad.toml"
    for contents in ("[wrong]\nrank_constant=10", "hybrid_retrieval=1", "[hybrid_retrieval"):
        path.write_text(contents)
        with pytest.raises(RetrievalError, match="cannot load hybrid retrieval configuration"):
            load_hybrid_retrieval_config(path)
    with pytest.raises(RetrievalError, match="cannot load"):
        load_hybrid_retrieval_config(tmp_path / "missing")


def test_default_weights_are_exactly_unweighted_fusion():
    vector, graph = result("vector", ["a", "b", "c"]), result("graph", ["c", "d"])

    plain = fuse_rrf(vector, graph, rank_constant=60, top_k=4)
    weighted = fuse_rrf(
        vector, graph, rank_constant=60, top_k=4, vector_weight=1.0, graph_weight=1.0
    )

    assert plain.ranking == weighted.ranking
    assert plain.result.model_dump(exclude={"elapsed_ms"}) == weighted.result.model_dump(
        exclude={"elapsed_ms"}
    )


def test_weights_scale_each_method_contribution():
    vector, graph = result("vector", ["a"]), result("graph", ["b"])

    trace = fuse_rrf(vector, graph, rank_constant=60, top_k=2, vector_weight=1.0, graph_weight=0.25)

    scores = {d.chunk_id: d for d in trace.ranking}
    assert scores["a"].vector_contribution == pytest.approx(1.0 / 61)
    assert scores["b"].graph_contribution == pytest.approx(0.25 / 61)
    assert [hit.chunk_id for hit in trace.result.hits] == ["a", "b"]


def test_down_weighting_graph_stops_it_displacing_a_deeper_vector_hit():
    """Unweighted, graph rank 1 outranks vector rank 6; this is what weights fix."""
    vector = result("vector", [f"v{i}" for i in range(1, 7)])
    graph = result("graph", ["g1"])

    unweighted = fuse_rrf(vector, graph, rank_constant=60, top_k=6)
    weighted = fuse_rrf(vector, graph, rank_constant=60, top_k=6, graph_weight=0.5)

    assert "g1" in [hit.chunk_id for hit in unweighted.result.hits]
    assert "v6" not in [hit.chunk_id for hit in unweighted.result.hits]
    assert [hit.chunk_id for hit in weighted.result.hits] == [f"v{i}" for i in range(1, 7)]


def test_zero_graph_weight_reproduces_the_vector_order():
    vector, graph = result("vector", ["a", "b", "c"]), result("graph", ["z", "y"])

    trace = fuse_rrf(vector, graph, rank_constant=60, top_k=3, graph_weight=0.0)

    assert [hit.chunk_id for hit in trace.result.hits] == ["a", "b", "c"]
    assert {d.chunk_id for d in trace.ranking} == {"a", "b", "c"}


def test_zero_weight_keeps_a_shared_candidate_through_the_other_method():
    vector, graph = result("vector", ["a", "shared"]), result("graph", ["shared"])

    trace = fuse_rrf(vector, graph, rank_constant=60, top_k=2, graph_weight=0.0)

    shared = next(d for d in trace.ranking if d.chunk_id == "shared")
    assert shared.graph_rank == 1
    assert shared.graph_contribution == 0.0
    assert shared.score == pytest.approx(1.0 / 62)


@pytest.mark.parametrize("weights", [{"vector_weight": -0.1}, {"graph_weight": -1}])
def test_negative_weights_rejected(weights):
    vector, graph = result("vector", ["a"]), result("graph", ["b"])

    with pytest.raises(RetrievalError, match="must be a nonnegative number"):
        fuse_rrf(vector, graph, rank_constant=60, top_k=1, **weights)


def test_both_weights_zero_rejected():
    vector, graph = result("vector", ["a"]), result("graph", ["b"])

    with pytest.raises(RetrievalError, match="at least one fusion weight"):
        fuse_rrf(vector, graph, rank_constant=60, top_k=1, vector_weight=0, graph_weight=0)


def test_config_defaults_and_validation_of_weights():
    assert HybridRetrievalConfig().vector_weight == 1.0
    assert HybridRetrievalConfig().graph_weight == 1.0
    assert HybridRetrievalConfig(graph_weight=0.25).graph_weight == 0.25
    with pytest.raises(ValidationError):
        HybridRetrievalConfig(graph_weight=-1)
    with pytest.raises(ValidationError, match="at least one fusion weight"):
        HybridRetrievalConfig(vector_weight=0, graph_weight=0)
