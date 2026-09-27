from pathlib import Path

import pytest
from pydantic import ValidationError

from graphrag_bench.config import ChunkingConfig
from graphrag_bench.extraction import extract, load_rules
from graphrag_bench.graph import GraphError, build_graph
from graphrag_bench.ingestion.chunker import chunk_document
from graphrag_bench.ingestion.parser import parse_bytes
from graphrag_bench.models import Entity, EvidenceSpan, RelationAssertion
from graphrag_bench.retrieval.graph import GraphRetriever
from graphrag_bench.retrieval.graph_config import GraphRetrievalConfig, load_graph_retrieval_config
from graphrag_bench.retrieval.linking import link_query
from graphrag_bench.retrieval.vector import RetrievalError

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def chain():
    texts = (
        "Orion is a retrieval method built on the Nova model.",
        "Nova is a language model evaluated on the Harbor dataset.",
    )
    docs = tuple(
        parse_bytes(text.encode(), relative_path=f"{i}.txt").document
        for i, text in enumerate(texts)
    )
    chunks = tuple(c for doc in docs for c in chunk_document(doc))
    result = extract(docs, chunks, load_rules(ROOT / "configs/extraction.toml"))
    return build_graph(result.entities, result.relations, docs, chunks), docs, chunks


def custom_graph(edges):
    """Controlled graph records with independently validated source quotes."""
    documents, chunks, relations = [], [], []
    entities = {
        identifier for _, subject, object_id in edges for identifier in (subject, object_id)
    }
    for i, (text, subject, object_id) in enumerate(edges):
        document = parse_bytes(text.encode(), relative_path=f"source-{i}.txt").document
        chunk = chunk_document(document)[0]
        documents.append(document)
        chunks.append(chunk)
        relations.append(
            RelationAssertion(
                assertion_id=f"edge-{i}",
                subject_id=subject,
                object_id=object_id,
                predicate="LINKS",
                evidence=(
                    EvidenceSpan(
                        document_id=document.document_id,
                        chunk_id=chunk.chunk_id,
                        start=0,
                        end=len(text),
                        text=text,
                    ),
                ),
                extraction_method="test",
                extraction_version="1",
            )
        )
    records = tuple(
        Entity(entity_id=name, name=name, entity_type="Concept") for name in sorted(entities)
    )
    graph = build_graph(records, relations, documents, chunks)
    return graph, tuple(documents), tuple(chunks)


@pytest.mark.parametrize("hops,expected", [(0, 0), (1, 1), (2, 2)])
def test_relation_only_hop_ablation_collects_bridge_evidence(chain, hops, expected):
    graph, docs, chunks = chain
    config = GraphRetrievalConfig(max_hops=hops, direction="outgoing", include_mentions=False)
    search = GraphRetriever(graph, docs, chunks, config)
    trace = search.retrieve_with_trace("Which dataset evaluates Orion's underlying model?", top_k=5)
    assert len(trace.result.hits) == expected
    if hops == 2:
        assert {h.chunk_id for h in trace.result.hits} == {c.chunk_id for c in chunks}
        second = next(h for h in trace.result.hits if h.chunk_id == chunks[1].chunk_id)
        path = next(p for p in second.paths if len(p.assertion_ids) == 2)
        assert [graph.entity(identifier).name for identifier in path.entity_ids] == [
            "Orion",
            "Nova",
            "Harbor",
        ]
        assert len({graph.assertion(i).evidence[0].document_id for i in path.assertion_ids}) == 2
        search.validate_evidence_path(second.chunk_id, path)


def test_neighbor_mentions_can_reveal_second_fact_after_only_one_walk_edge(chain):
    graph, docs, chunks = chain
    one = GraphRetriever(graph, docs, chunks, GraphRetrievalConfig(max_hops=1))
    trace = one.retrieve_with_trace("Orion", top_k=5)
    assert {h.chunk_id for h in trace.result.hits} == {c.chunk_id for c in chunks}
    zero = GraphRetriever(graph, docs, chunks, GraphRetrievalConfig(max_hops=0))
    assert [h.chunk_id for h in zero.retrieve("Orion").hits] == [chunks[0].chunk_id]


def test_incoming_walk_keeps_original_assertion_direction(chain):
    graph, docs, chunks = chain
    outgoing = GraphRetriever(
        graph, docs, chunks, GraphRetrievalConfig(direction="outgoing", include_mentions=False)
    )
    incoming = GraphRetriever(
        graph, docs, chunks, GraphRetrievalConfig(direction="incoming", include_mentions=False)
    )
    assert outgoing.retrieve("Harbor").hits == ()
    trace = incoming.retrieve_with_trace("Harbor")
    assert len(trace.result.hits) == 2
    path = next(
        path for hit in trace.result.hits for path in hit.paths if len(path.assertion_ids) == 2
    )
    graph.validate_path(path, direction="incoming")
    graph.validate_path(path, direction="both")
    with pytest.raises(GraphError, match="direction"):
        graph.validate_path(path, direction="outgoing")
    first = graph.assertion(path.assertion_ids[0])
    assert graph.entity(first.subject_id).name == "Nova"
    assert graph.entity(first.object_id).name == "Harbor"


def test_unknown_query_names_return_empty_without_fallback(chain):
    graph, docs, chunks = chain
    trace = GraphRetriever(graph, docs, chunks).retrieve_with_trace("What about an unknown topic?")
    assert trace.links == trace.ranking == trace.result.hits == ()
    assert trace.admitted_paths == trace.candidate_chunks == 0


def test_linking_aliases_boundaries_longest_names_and_unicode():
    entities = (
        Entity(entity_id="quartz", name="Quartz", aliases=("QZ",), entity_type="Encoder"),
        Entity(entity_id="birch", name="Birch", aliases=("Base",), entity_type="Model"),
        Entity(entity_id="elm", name="Elm", aliases=("Base",), entity_type="Model"),
        Entity(entity_id="new-york", name="New York", entity_type="Place"),
        Entity(entity_id="york", name="York", entity_type="Place"),
        Entity(entity_id="street", name="Straße", entity_type="Concept"),
    )
    links = link_query("QUARTZ and QZ in New   York, STRASSE, and Base?", entities)
    assert any(
        link.surfaces == ("quartz", "qz") and link.candidate_entity_ids == ("quartz",)
        for link in links
    )
    assert any(link.candidate_entity_ids == ("birch", "elm") for link in links)
    assert {identifier for link in links for identifier in link.candidate_entity_ids} == {
        "quartz",
        "birch",
        "elm",
        "new-york",
        "street",
    }
    assert link_query("Quartzite Quartz-v2", entities) == ()
    assert link_query("New York and York", entities)[-1].candidate_entity_ids == ("york",)


def test_ambiguous_alias_traverses_both_candidates_as_one_query_group(corpus):
    chunks = tuple(c for d in corpus.documents for c in chunk_document(d))
    extracted = extract(corpus.documents, chunks, load_rules(ROOT / "configs/extraction.toml"))
    graph = build_graph(extracted.entities, extracted.relations, corpus.documents, chunks)
    search = GraphRetriever(graph, corpus.documents, chunks, GraphRetrievalConfig(max_hops=0))
    trace = search.retrieve_with_trace("Base", top_k=999)
    assert len(trace.links) == 1
    assert {graph.entity(i).name for i in trace.links[0].candidate_entity_ids} == {"Birch", "Elm"}
    assert {search.chunk(h.chunk_id).document_id for h in trace.result.hits} >= {
        "doc-birch",
        "doc-elm",
    }
    assert all(rank.seed_groups == (0,) for rank in trace.ranking)
    limited = GraphRetriever(graph, corpus.documents, chunks, GraphRetrievalConfig(max_seeds=1))
    with pytest.raises(RetrievalError, match="ambiguous candidates are not discarded"):
        limited.retrieve("Base")


def test_numeric_fact_remains_retrievable_through_seed_mentions(corpus):
    chunks = tuple(
        c
        for d in corpus.documents
        for c in chunk_document(d, ChunkingConfig(max_units=1, overlap_units=0))
    )
    extracted = extract(corpus.documents, chunks, load_rules(ROOT / "configs/extraction.toml"))
    graph = build_graph(extracted.entities, extracted.relations, corpus.documents, chunks)
    search = GraphRetriever(graph, corpus.documents, chunks, GraphRetrievalConfig(max_hops=0))
    hits = search.retrieve("How many examples does Cedar contain?", top_k=99).hits
    assert "Cedar contains 1200 examples." in {search.chunk(h.chunk_id).text for h in hits}


def test_parallel_sources_survive_until_explicit_assertion_limit(corpus):
    chunks = tuple(c for d in corpus.documents for c in chunk_document(d))
    extracted = extract(corpus.documents, chunks, load_rules(ROOT / "configs/extraction.toml"))
    graph = build_graph(extracted.entities, extracted.relations, corpus.documents, chunks)
    settings = dict(
        max_hops=1, direction="outgoing", include_mentions=False, predicates=("REPORTS",)
    )
    full = GraphRetriever(graph, corpus.documents, chunks, GraphRetrievalConfig(**settings))
    assert len(full.retrieve("Cedar").hits) == 2
    limited = GraphRetriever(
        graph,
        corpus.documents,
        chunks,
        GraphRetrievalConfig(**settings, max_assertions_per_neighbor=1),
    )
    trace = limited.retrieve_with_trace("Cedar")
    assert len(trace.result.hits) == 1
    assert trace.limits_reached == ("assertions",)


def test_hub_and_total_path_limits_are_explicit_and_deterministic():
    graph, docs, chunks = custom_graph(
        tuple((f"Hub links N{i}.", "Hub", f"N{i}") for i in range(5))
    )
    config = GraphRetrievalConfig(max_hops=1, max_neighbors=2, include_mentions=False)
    limited = GraphRetriever(graph, docs, chunks, config)
    trace = limited.retrieve_with_trace("Hub", top_k=99)
    assert trace.limits_reached == ("neighbors",)
    assert trace.admitted_paths == 3 and len(trace.result.hits) == 2
    again = GraphRetriever(graph, reversed(docs), reversed(chunks), config)
    assert again.retrieve("Hub", top_k=99).hits == trace.result.hits
    config = GraphRetrievalConfig(max_hops=2, max_seeds=1, max_paths=2, include_mentions=False)
    trace = GraphRetriever(graph, docs, chunks, config).retrieve_with_trace("Hub")
    assert trace.admitted_paths == 2
    assert trace.limits_reached == ("paths",)


def test_cycles_do_not_revisit_entities_and_paths_deduplicate():
    graph, docs, chunks = custom_graph((("A links B.", "A", "B"), ("B links A.", "B", "A")))
    trace = GraphRetriever(
        graph, docs, chunks, GraphRetrievalConfig(include_mentions=False)
    ).retrieve_with_trace("A")
    assert trace.admitted_paths == 3
    assert len(trace.result.hits) == 2
    for hit in trace.result.hits:
        assert len(hit.paths) == len(set(hit.paths))
        assert all(len(path.entity_ids) == len(set(path.entity_ids)) for path in hit.paths)


def test_seed_coverage_precedes_distance_and_scores_are_ordinal():
    graph, docs, chunks = custom_graph(
        (
            ("A links C.", "A", "C"),
            ("B links C.", "B", "C"),
            ("A links D.", "A", "D"),
        )
    )
    search = GraphRetriever(
        graph, docs, chunks, GraphRetrievalConfig(max_hops=2, include_mentions=False)
    )
    trace = search.retrieve_with_trace("A and B", top_k=99)
    assert [len(rank.seed_groups) for rank in trace.ranking] == [2, 2, 1]
    assert [hit.score for hit in trace.result.hits] == [3.0, 2.0, 1.0]
    assert trace.ranking[-1].chunk_id == chunks[2].chunk_id
    assert [hit.chunk_id for hit in trace.result.hits[:2]] == sorted(c.chunk_id for c in chunks[:2])


def test_every_returned_path_supports_its_exact_chunk(chain):
    graph, docs, chunks = chain
    search = GraphRetriever(graph, docs, chunks, GraphRetrievalConfig(include_mentions=False))
    trace = search.retrieve_with_trace("Orion")
    for hit in trace.result.hits:
        for path in hit.paths:
            search.validate_evidence_path(hit.chunk_id, path)
    first_path = next(p for h in trace.result.hits for p in h.paths if len(p.assertion_ids) == 1)
    with pytest.raises(RetrievalError, match="does not support this chunk"):
        search.validate_evidence_path(chunks[1].chunk_id, first_path)
    bad = first_path.model_copy(update={"assertion_ids": ("missing",)})
    with pytest.raises(GraphError, match="unknown assertion"):
        graph.validate_path(bad)
    disconnected = first_path.model_copy(
        update={"entity_ids": tuple(reversed(first_path.entity_ids))}
    )
    with pytest.raises(GraphError, match="direction"):
        graph.validate_path(disconnected)


def test_graph_cannot_be_paired_with_a_different_corpus(chain):
    graph, docs, chunks = chain
    with pytest.raises(GraphError, match="unknown document"):
        GraphRetriever(graph, docs[:1], chunks[:1])


@pytest.mark.parametrize(
    "settings",
    [
        {"max_hops": 3},
        {"max_hops": -1},
        {"max_hops": True},
        {"max_neighbors": 0},
        {"max_paths": 1},
        {"predicates": ("USES", "USES")},
    ],
)
def test_invalid_search_settings_rejected(settings):
    with pytest.raises(ValidationError):
        GraphRetrievalConfig(**settings)


def test_config_overrides_are_validated_and_live_config_is_read_only(chain):
    config = load_graph_retrieval_config(
        ROOT / "configs/graph-retrieval.toml", overrides={"max_hops": 0}
    )
    assert config.max_hops == 0
    graph, docs, chunks = chain
    search = GraphRetriever(graph, docs, chunks, config)
    with pytest.raises(AttributeError):
        search.config = GraphRetrievalConfig()
