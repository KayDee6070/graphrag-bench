"""Validate first, then construct a NetworkX directed multigraph."""

from collections.abc import Iterable

import networkx as nx

from graphrag_bench.corpus import CorpusError, CorpusIndex
from graphrag_bench.extraction.registry import normalize_name
from graphrag_bench.models import Chunk, Document, Entity, Record, RelationAssertion, require_unique

GRAPH_VERSION = "assertion-multidigraph-v1"


class GraphError(ValueError):
    """Graph records or artifacts are inconsistent."""


class GraphSnapshot(Record):
    entities: tuple[Entity, ...]
    relations: tuple[RelationAssertion, ...]


class KnowledgeGraph:
    """Read-only graph inspection. Query interpretation/ranking belong to later milestones.

    Each assertion is its own edge, keyed by assertion_id. Returned records are
    defensive copies because dictionaries inside frozen Pydantic models are mutable.
    """

    def __init__(self, snapshot: GraphSnapshot, corpus: CorpusIndex) -> None:
        _validate(snapshot, corpus)
        self._graph = nx.MultiDiGraph()
        for entity in snapshot.entities:
            self._graph.add_node(entity.entity_id, entity=entity.model_copy(deep=True))
        for relation in snapshot.relations:
            if relation.subject_id not in self._graph or relation.object_id not in self._graph:
                raise GraphError(f"unknown relation endpoint: {relation.assertion_id}")
            self._graph.add_edge(
                relation.subject_id,
                relation.object_id,
                key=relation.assertion_id,
                assertion=relation.model_copy(deep=True),
            )
        nx.freeze(self._graph)

    @property
    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    def entity(self, entity_id: str) -> Entity:
        if entity_id not in self._graph:
            raise GraphError(f"unknown entity: {entity_id}")
        return self._graph.nodes[entity_id]["entity"].model_copy(deep=True)

    def snapshot(self) -> GraphSnapshot:
        return GraphSnapshot(
            entities=tuple(self.entity(key) for key in sorted(self._graph)),
            relations=tuple(
                sorted(
                    (
                        data["assertion"].model_copy(deep=True)
                        for *_, data in self._graph.edges(data=True, keys=True)
                    ),
                    key=lambda r: r.assertion_id,
                )
            ),
        )

    def lookup(self, name: str, entity_type: str | None = None) -> tuple[Entity, ...]:
        """Return all matching canonical names/aliases; ambiguous matches stay plural."""
        key = normalize_name(name)
        return tuple(
            entity
            for entity in (self.entity(key) for key in sorted(self._graph))
            if (entity_type is None or entity.entity_type == entity_type)
            and key in {normalize_name(n) for n in (entity.name, *entity.aliases)}
        )

    def outgoing(
        self, entity_id: str, predicate: str | None = None
    ) -> tuple[RelationAssertion, ...]:
        self.entity(entity_id)
        return tuple(
            sorted(
                (
                    data["assertion"].model_copy(deep=True)
                    for *_, data in self._graph.out_edges(entity_id, data=True, keys=True)
                    if predicate is None or data["assertion"].predicate == predicate
                ),
                key=lambda r: r.assertion_id,
            )
        )

    def incoming(
        self, entity_id: str, predicate: str | None = None
    ) -> tuple[RelationAssertion, ...]:
        self.entity(entity_id)
        return tuple(
            sorted(
                (
                    data["assertion"].model_copy(deep=True)
                    for *_, data in self._graph.in_edges(entity_id, data=True, keys=True)
                    if predicate is None or data["assertion"].predicate == predicate
                ),
                key=lambda r: r.assertion_id,
            )
        )


def _validate(snapshot: GraphSnapshot, corpus: CorpusIndex) -> None:
    entities, relations = snapshot.entities, snapshot.relations
    try:
        require_unique(tuple(e.entity_id for e in entities), "entity IDs")
        require_unique(tuple(r.assertion_id for r in relations), "assertion IDs")
        identifiers = {entity.entity_id for entity in entities}
        for entity in entities:
            surfaces = {normalize_name(name) for name in (entity.name, *entity.aliases)}
            for mention in entity.mentions:
                corpus.validate_span(mention, require_chunk=True)
                if normalize_name(mention.text) not in surfaces:
                    raise GraphError(f"mention does not match an entity name: {entity.entity_id}")
        for relation in relations:
            if relation.subject_id not in identifiers or relation.object_id not in identifiers:
                raise GraphError(f"unknown relation endpoint: {relation.assertion_id}")
            for evidence in relation.evidence:
                corpus.validate_span(evidence, require_chunk=True)
    except (ValueError, CorpusError) as error:
        raise GraphError(str(error)) from error


def build_graph(
    entities: Iterable[Entity],
    relations: Iterable[RelationAssertion],
    documents: Iterable[Document],
    chunks: Iterable[Chunk],
) -> KnowledgeGraph:
    """Reject dangling nodes and invalid provenance before NetworkX can add any edges."""
    try:
        corpus = CorpusIndex(documents, chunks)
        snapshot = GraphSnapshot(entities=tuple(entities), relations=tuple(relations))
        return KnowledgeGraph(snapshot, corpus)
    except ValueError as error:
        raise GraphError(str(error)) from error
