"""Fuse independent vector and graph rankings without mixing incompatible raw scores."""

from time import perf_counter
from typing import Literal

from pydantic import Field

from graphrag_bench.embeddings.base import EmbeddingProvider
from graphrag_bench.models import (
    Chunk,
    Identifier,
    NonNegativeInt,
    PositiveInt,
    Record,
    RetrievalHit,
    RetrievalResult,
)
from graphrag_bench.retrieval.graph import GraphRetriever, GraphSearchTrace
from graphrag_bench.retrieval.hybrid_config import HybridRetrievalConfig
from graphrag_bench.retrieval.vector import (
    VECTOR_VERSION,
    RetrievalError,
    VectorIndex,
    validate_request,
)

HYBRID_RETRIEVAL_VERSION = "reciprocal-rank-fusion-v1"


class FusionRank(Record):
    chunk_id: Identifier
    vector_rank: PositiveInt | None
    graph_rank: PositiveInt | None
    vector_contribution: float = Field(ge=0, strict=True)
    graph_contribution: float = Field(ge=0, strict=True)
    score: float = Field(gt=0, strict=True)


class FusionTrace(Record):
    """All union candidates in fused order; result.hits alone applies the final top K."""

    result: RetrievalResult
    ranking: tuple[FusionRank, ...]


class HybridSearchTrace(FusionTrace):
    config: HybridRetrievalConfig
    top_k: PositiveInt
    vector_result: RetrievalResult
    graph_trace: GraphSearchTrace
    candidate_chunks: NonNegativeInt
    fusion_ms: float = Field(ge=0, strict=True)
    vector_version: Literal["numpy-exact-cosine-v1"] = VECTOR_VERSION
    retrieval_version: Literal["reciprocal-rank-fusion-v1"] = HYBRID_RETRIEVAL_VERSION


def fuse_rrf(
    vector: RetrievalResult,
    graph: RetrievalResult,
    *,
    rank_constant: int = 60,
    top_k: int = 5,
) -> FusionTrace:
    """Sum 1/(c + one-based rank); missing hits add zero; equal scores use chunk IDs.

    Hit order defines rank, irrespective of raw scores. This arithmetic function preserves
    paths but does not verify their source records. HybridRetriever supplies that boundary.
    Its elapsed_ms measures fusion only; the retriever measures the full warm query.
    """
    validate_request(vector.query, top_k)
    if vector.query != graph.query:
        raise RetrievalError("fusion inputs must have exactly the same query")
    if vector.strategy != "vector" or graph.strategy != "graph":
        raise RetrievalError("fusion requires one vector result and one graph result")
    if type(rank_constant) is not int or rank_constant < 0:
        raise RetrievalError("rank_constant must be a nonnegative integer")
    started = perf_counter()
    vector_ranks = {hit.chunk_id: rank for rank, hit in enumerate(vector.hits, start=1)}
    graph_ranks = {hit.chunk_id: rank for rank, hit in enumerate(graph.hits, start=1)}
    paths = {hit.chunk_id: set(hit.paths) for hit in vector.hits}
    for hit in graph.hits:
        paths.setdefault(hit.chunk_id, set()).update(hit.paths)
    details = []
    for chunk_id in vector_ranks.keys() | graph_ranks.keys():
        vector_rank, graph_rank = vector_ranks.get(chunk_id), graph_ranks.get(chunk_id)
        vector_contribution = 0.0 if vector_rank is None else 1.0 / (rank_constant + vector_rank)
        graph_contribution = 0.0 if graph_rank is None else 1.0 / (rank_constant + graph_rank)
        details.append(
            FusionRank(
                chunk_id=chunk_id,
                vector_rank=vector_rank,
                graph_rank=graph_rank,
                vector_contribution=vector_contribution,
                graph_contribution=graph_contribution,
                score=vector_contribution + graph_contribution,
            )
        )
    ranking = tuple(sorted(details, key=lambda detail: (-detail.score, detail.chunk_id)))
    hits = tuple(
        RetrievalHit(
            chunk_id=detail.chunk_id,
            score=detail.score,
            paths=tuple(
                sorted(
                    paths[detail.chunk_id],
                    key=lambda p: (len(p.assertion_ids), p.entity_ids, p.assertion_ids),
                )
            ),
        )
        for detail in ranking[:top_k]
    )
    return FusionTrace(
        result=RetrievalResult(
            query=vector.query,
            strategy="hybrid",
            hits=hits,
            elapsed_ms=(perf_counter() - started) * 1000,
        ),
        ranking=ranking,
    )


class HybridRetriever:
    def __init__(
        self,
        vector: VectorIndex,
        graph: GraphRetriever,
        provider: EmbeddingProvider,
        config: HybridRetrievalConfig | None = None,
    ) -> None:
        # Equal IDs alone would allow fusion to attach evidence from the wrong source text.
        if vector.chunks != graph.chunks:
            raise RetrievalError("hybrid retrievers must use exactly the same source chunks")
        if provider.spec != vector.spec:
            raise RetrievalError(
                "query provider does not match the indexed embedding specification"
            )
        self._vector = vector
        self._graph = graph
        self._provider = provider
        self._config = config if config is not None else HybridRetrievalConfig()

    @property
    def config(self) -> HybridRetrievalConfig:
        return self._config

    def chunk(self, chunk_id: str) -> Chunk:
        return self._vector.chunk(chunk_id)

    def retrieve(self, query: str, *, top_k: int = 5) -> RetrievalResult:
        return self.retrieve_with_trace(query, top_k=top_k).result

    def retrieve_with_trace(self, query: str, *, top_k: int = 5) -> HybridSearchTrace:
        validate_request(query, top_k)
        started = perf_counter()
        # Fail on graph ambiguity limits before encoding. Errors never become silent fallback.
        graph_trace = self._graph.retrieve_with_trace(query, top_k=self.config.graph_candidates)
        vector_result = self._vector.retrieve(
            query, self._provider, top_k=self.config.vector_candidates
        )
        fusion = fuse_rrf(
            vector_result,
            graph_trace.result,
            rank_constant=self.config.rank_constant,
            top_k=top_k,
        )
        for hit in fusion.result.hits:
            self.chunk(hit.chunk_id)
            for path in hit.paths:
                self._graph.validate_evidence_path(hit.chunk_id, path)
        return HybridSearchTrace(
            result=RetrievalResult(
                query=query,
                strategy="hybrid",
                hits=fusion.result.hits,
                elapsed_ms=(perf_counter() - started) * 1000,
            ),
            ranking=fusion.ranking,
            config=self.config,
            top_k=top_k,
            vector_result=vector_result,
            graph_trace=graph_trace,
            candidate_chunks=len(fusion.ranking),
            fusion_ms=fusion.result.elapsed_ms,
        )
