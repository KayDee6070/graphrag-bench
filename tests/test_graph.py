from pathlib import Path

import pytest

from graphrag_bench.extraction import extract, load_rules
from graphrag_bench.graph import GraphError, build_graph
from graphrag_bench.ingestion.chunker import chunk_document


@pytest.fixture
def predicted(corpus):
    rules = load_rules(Path(__file__).resolve().parents[1] / "configs" / "extraction.toml")
    chunks = tuple(c for d in corpus.documents for c in chunk_document(d))
    result = extract(corpus.documents, chunks, rules)
    return result, chunks


def test_directed_multigraph_preserves_source_assertions_and_alias_candidates(corpus, predicted):
    result, chunks = predicted
    graph = build_graph(result.entities, result.relations, corpus.documents, chunks)
    assert (graph.node_count, graph.edge_count) == (15, 23)
    assert {e.name for e in graph.lookup("base")} == {"Birch", "Elm"}
    assert graph.lookup(" QZ ", "Encoder")[0].name == "Quartz"
    assert graph.lookup("QZ", "Dataset") == ()
    assert graph.lookup("Unknown") == ()
    cedar = graph.lookup("Cedar")[0]
    accuracy = graph.lookup("Accuracy")[0]
    repeated = graph.outgoing(cedar.entity_id, "REPORTS")
    assert len(repeated) == 2
    assert {r.object_id for r in repeated} == {accuracy.entity_id}
    assert {e.document_id for r in repeated for e in r.evidence} == {"doc-cedar", "doc-lumen"}
    assert graph.outgoing(accuracy.entity_id, "REPORTS") == ()
    assert set(r.assertion_id for r in repeated) <= {
        r.assertion_id for r in graph.incoming(accuracy.entity_id, "REPORTS")
    }
    # Inspect two existing edges; this does not implement query retrieval or ranking.
    alder = graph.lookup("Alder")[0]
    first = graph.outgoing(alder.entity_id, "BASED_ON")[0]
    second = graph.outgoing(first.object_id, "EVALUATED_ON")[0]
    assert graph.entity(second.object_id).name == "Cedar"
    assert first.evidence[0].document_id != second.evidence[0].document_id


@pytest.mark.parametrize("kind", ["entity", "assertion"])
def test_duplicate_ids_rejected(corpus, predicted, kind):
    result, chunks = predicted
    entities, relations = result.entities, result.relations
    if kind == "entity":
        entities = (*entities, entities[0])
    else:
        relations = (*relations, relations[0])
    with pytest.raises(GraphError, match="duplicate"):
        build_graph(entities, relations, corpus.documents, chunks)


def test_dangling_endpoint_is_rejected_instead_of_creating_a_phantom_node(corpus, predicted):
    result, chunks = predicted
    broken = result.relations[0].model_copy(update={"object_id": "missing"})
    with pytest.raises(GraphError, match="unknown relation endpoint"):
        build_graph(result.entities, (broken,), corpus.documents, chunks)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"text": "x"}, "source text mismatch"),
        ({"chunk_id": "missing"}, "unknown chunk"),
        ({"document_id": "missing"}, "unknown document"),
        ({"chunk_id": None}, "requires.*chunk"),
    ],
)
def test_invalid_relation_provenance_rejected(corpus, predicted, change, message):
    result, chunks = predicted
    relation = result.relations[0]
    evidence = relation.evidence[0].model_copy(update=change)
    relation = relation.model_copy(update={"evidence": (evidence,)})
    with pytest.raises(GraphError, match=message):
        build_graph(result.entities, (relation,), corpus.documents, chunks)


def test_evidence_from_wrong_chunk_is_rejected(corpus, predicted):
    result, chunks = predicted
    relation = result.relations[0]
    span = relation.evidence[0]
    wrong = next(c for c in chunks if c.document_id != span.document_id)
    relation = relation.model_copy(
        update={"evidence": (span.model_copy(update={"chunk_id": wrong.chunk_id}),)}
    )
    with pytest.raises(GraphError, match="outside its source chunk"):
        build_graph(result.entities, (relation,), corpus.documents, chunks)


def test_mention_requires_correct_entity_name(corpus, predicted):
    result, chunks = predicted
    entity = result.entities[0].model_copy(update={"name": "Incorrect", "aliases": ()})
    with pytest.raises(GraphError, match="mention does not match"):
        build_graph((entity, *result.entities[1:]), result.relations, corpus.documents, chunks)


def test_graph_preserves_parallel_predicates_qualifiers_and_protects_internal_records(
    corpus, predicted
):
    result, chunks = predicted
    first = result.relations[0].model_copy(update={"qualifiers": {"context": "first"}})
    second = first.model_copy(update={"assertion_id": "another", "predicate": "CUSTOM"})
    graph = build_graph(result.entities, (first, second), corpus.documents, chunks)
    first.qualifiers["context"] = "mutated input"
    returned = graph.outgoing(first.subject_id)
    returned[0].qualifiers["context"] = "mutated output"
    assert graph.edge_count == 2
    assert {r.predicate for r in graph.outgoing(first.subject_id)} == {first.predicate, "CUSTOM"}
    assert all(r.qualifiers == {"context": "first"} for r in graph.snapshot().relations)


def test_unknown_node_inspection_fails_explicitly(corpus, predicted):
    result, chunks = predicted
    graph = build_graph(result.entities, result.relations, corpus.documents, chunks)
    with pytest.raises(GraphError, match="unknown entity"):
        graph.outgoing("missing")
