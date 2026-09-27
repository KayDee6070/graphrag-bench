"""Bounded evidence traversal with explicit ambiguity, provenance, and ordinal ranking."""

from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from time import perf_counter
from typing import Literal

from graphrag_bench.corpus import CorpusIndex
from graphrag_bench.graph.builder import KnowledgeGraph, build_graph
from graphrag_bench.models import (
    Chunk,
    Document,
    Identifier,
    NonNegativeInt,
    Record,
    RelationAssertion,
    RetrievalHit,
    RetrievalResult,
    TraversalPath,
)
from graphrag_bench.retrieval.graph_config import GraphRetrievalConfig
from graphrag_bench.retrieval.linking import QueryLink, link_query
from graphrag_bench.retrieval.vector import RetrievalError, validate_request

GRAPH_RETRIEVAL_VERSION = "bounded-evidence-walk-v1"


class GraphRank(Record):
    chunk_id: Identifier
    seed_groups: tuple[NonNegativeInt, ...]
    min_hops: NonNegativeInt


class GraphSearchTrace(Record):
    result: RetrievalResult
    links: tuple[QueryLink, ...]
    ranking: tuple[GraphRank, ...]
    config: GraphRetrievalConfig
    admitted_paths: NonNegativeInt
    candidate_chunks: NonNegativeInt
    limits_reached: tuple[Literal["neighbors", "assertions", "paths"], ...]
    retrieval_version: Literal["bounded-evidence-walk-v1"] = GRAPH_RETRIEVAL_VERSION


@dataclass
class _Candidate:
    seed_groups: set[int] = field(default_factory=set)
    paths: set[TraversalPath] = field(default_factory=set)


def _path_key(path: TraversalPath) -> tuple:
    return len(path.assertion_ids), path.entity_ids, path.assertion_ids


class GraphRetriever:
    def __init__(
        self,
        graph: KnowledgeGraph,
        documents: Iterable[Document],
        chunks: Iterable[Chunk],
        config: GraphRetrievalConfig | None = None,
    ) -> None:
        self._config = config if config is not None else GraphRetrievalConfig()
        self._corpus = CorpusIndex(documents, chunks)
        snapshot = graph.snapshot()
        # Also reject a valid graph accidentally paired with a different source corpus.
        self._graph = build_graph(
            snapshot.entities,
            snapshot.relations,
            self._corpus.documents.values(),
            self._corpus.chunks.values(),
        )
        self._entities = {entity.entity_id: entity for entity in snapshot.entities}
        self._assertions = {relation.assertion_id: relation for relation in snapshot.relations}
        self._adjacency: dict[str, list[tuple[str, RelationAssertion]]] = defaultdict(list)
        for relation in snapshot.relations:
            if self.config.predicates and relation.predicate not in self.config.predicates:
                continue
            if self.config.direction in {"outgoing", "both"}:
                self._adjacency[relation.subject_id].append((relation.object_id, relation))
            if self.config.direction in {"incoming", "both"}:
                self._adjacency[relation.object_id].append((relation.subject_id, relation))

    @property
    def config(self) -> GraphRetrievalConfig:
        return self._config

    @property
    def graph(self) -> KnowledgeGraph:
        return self._graph

    def chunk(self, chunk_id: str) -> Chunk:
        if chunk_id not in self._corpus.chunks:
            raise RetrievalError(f"unknown source chunk: {chunk_id}")
        return self._corpus.chunks[chunk_id]

    def validate_evidence_path(self, chunk_id: str, path: TraversalPath) -> None:
        self.chunk(chunk_id)
        self.graph.validate_path(path, direction=self.config.direction)
        on_edge = any(
            span.chunk_id == chunk_id
            for assertion_id in path.assertion_ids
            for span in self._assertions[assertion_id].evidence
        )
        on_mention = self.config.include_mentions and any(
            span.chunk_id == chunk_id for span in self._entities[path.entity_ids[-1]].mentions
        )
        if not on_edge and not on_mention:
            raise RetrievalError(
                "path does not support this chunk through an edge or terminal mention"
            )

    def _extensions(
        self,
        path: TraversalPath,
        limits: set[str],
    ) -> list[TraversalPath]:
        neighbors: dict[str, list[RelationAssertion]] = defaultdict(list)
        for neighbor, assertion in self._adjacency[path.entity_ids[-1]]:
            if neighbor not in path.entity_ids:
                neighbors[neighbor].append(assertion)
        if len(neighbors) > self.config.max_neighbors:
            limits.add("neighbors")
        extended = []
        for neighbor in sorted(neighbors)[: self.config.max_neighbors]:
            assertions = sorted(neighbors[neighbor], key=lambda r: r.assertion_id)
            if len(assertions) > self.config.max_assertions_per_neighbor:
                limits.add("assertions")
            for assertion in assertions[: self.config.max_assertions_per_neighbor]:
                extended.append(
                    TraversalPath(
                        entity_ids=(*path.entity_ids, neighbor),
                        assertion_ids=(*path.assertion_ids, assertion.assertion_id),
                    )
                )
        return extended

    def _evidence(self, path: TraversalPath) -> set[str]:
        # Preserve evidence on every edge of the walk, including its bridge statements.
        chunk_ids = {
            span.chunk_id
            for assertion_id in path.assertion_ids
            for span in self._assertions[assertion_id].evidence
        }
        if self.config.include_mentions:
            chunk_ids.update(span.chunk_id for span in self._entities[path.entity_ids[-1]].mentions)
        return {identifier for identifier in chunk_ids if identifier is not None}

    def retrieve(self, query: str, *, top_k: int = 5) -> RetrievalResult:
        return self.retrieve_with_trace(query, top_k=top_k).result

    def retrieve_with_trace(self, query: str, *, top_k: int = 5) -> GraphSearchTrace:
        validate_request(query, top_k)
        started = perf_counter()
        links = link_query(query, self._entities.values())
        seed_groups: dict[str, set[int]] = defaultdict(set)
        for group, link in enumerate(links):
            for entity_id in link.candidate_entity_ids:
                seed_groups[entity_id].add(group)
        if len(seed_groups) > self.config.max_seeds:
            raise RetrievalError(
                f"query links {len(seed_groups)} candidates; limit is {self.config.max_seeds}. "
                "Narrow the query or increase the limit; ambiguous candidates are not discarded."
            )
        queue = deque(
            TraversalPath(entity_ids=(identifier,), assertion_ids=())
            for identifier in sorted(seed_groups)
        )
        admitted = set(queue)
        candidates: dict[str, _Candidate] = defaultdict(_Candidate)
        limits: set[str] = set()
        while queue:
            path = queue.popleft()
            for chunk_id in self._evidence(path):
                candidates[chunk_id].seed_groups.update(seed_groups[path.entity_ids[0]])
                candidates[chunk_id].paths.add(path)
            if len(path.assertion_ids) == self.config.max_hops:
                continue
            for extension in self._extensions(path, limits):
                if extension in admitted:
                    continue
                if len(admitted) == self.config.max_paths:
                    limits.add("paths")
                    continue
                admitted.add(extension)
                queue.append(extension)
        ordered = sorted(
            candidates,
            key=lambda identifier: (
                -len(candidates[identifier].seed_groups),
                min(len(path.assertion_ids) for path in candidates[identifier].paths),
                identifier,
            ),
        )
        hits = []
        ranking = []
        for position, chunk_id in enumerate(ordered[:top_k]):
            candidate = candidates[chunk_id]
            paths = tuple(sorted(candidate.paths, key=_path_key))
            for path in paths:
                self.validate_evidence_path(chunk_id, path)
            # An ordinal encoding of the declared ordering, not a calibrated relevance score.
            hits.append(
                RetrievalHit(chunk_id=chunk_id, score=float(len(ordered) - position), paths=paths)
            )
            ranking.append(
                GraphRank(
                    chunk_id=chunk_id,
                    seed_groups=tuple(sorted(candidate.seed_groups)),
                    min_hops=len(paths[0].assertion_ids),
                )
            )
        result = RetrievalResult(
            query=query,
            strategy="graph",
            hits=tuple(hits),
            elapsed_ms=(perf_counter() - started) * 1000,
        )
        return GraphSearchTrace(
            result=result,
            links=links,
            ranking=tuple(ranking),
            config=self.config,
            admitted_paths=len(admitted),
            candidate_chunks=len(candidates),
            limits_reached=tuple(sorted(limits)),
        )
