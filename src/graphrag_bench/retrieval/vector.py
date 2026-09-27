"""Exact cosine search: normalize vectors, score every chunk, sort with stable ties."""

from collections.abc import Iterable
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from graphrag_bench.corpus import CorpusIndex
from graphrag_bench.embeddings.base import EmbeddingProvider, EmbeddingSpec
from graphrag_bench.models import Chunk, Document, RetrievalHit, RetrievalResult, require_unique

VECTOR_VERSION = "numpy-exact-cosine-v1"


class RetrievalError(ValueError):
    """Retrieval inputs, vectors, or stored artifacts are inconsistent."""


def validate_request(query: str, top_k: int) -> None:
    if not isinstance(query, str) or not query.strip():
        raise RetrievalError("query must be a nonblank string")
    if type(top_k) is not int or top_k < 1:
        raise RetrievalError("top_k must be a positive integer")


def normalize_vectors(values: NDArray, rows: int, dimensions: int) -> NDArray[np.float32]:
    array = np.asarray(values)
    if array.shape != (rows, dimensions) or array.dtype.kind not in "fiu":
        raise RetrievalError(f"expected a numeric embedding matrix of shape {(rows, dimensions)}")
    with np.errstate(over="ignore", invalid="ignore"):
        array = array.astype(np.float32)
    if not np.isfinite(array).all():
        raise RetrievalError("embedding vectors must be finite float32 values")
    norms = np.linalg.norm(array.astype(np.float64), axis=1)
    if np.any(norms == 0):
        raise RetrievalError("zero embedding vectors have no cosine direction")
    return np.asarray(array / norms[:, None], dtype="<f4", order="C")


class VectorIndex:
    """Rows are already normalized; chunk IDs identify exact sources, not text guesses."""

    def __init__(
        self,
        chunks: tuple[Chunk, ...],
        vectors: NDArray[np.float32],
        spec: EmbeddingSpec,
    ) -> None:
        require_unique(tuple(chunk.chunk_id for chunk in chunks), "chunk IDs")
        if tuple(c.chunk_id for c in chunks) != tuple(sorted(c.chunk_id for c in chunks)):
            raise RetrievalError("index chunks must be sorted by chunk_id")
        array = np.asarray(vectors)
        if array.dtype != np.dtype("<f4") or array.shape != (len(chunks), spec.dimensions):
            raise RetrievalError(
                "index matrix must have float32 rows matching chunks and dimensions"
            )
        if not np.isfinite(array).all() or not np.allclose(
            np.linalg.norm(array.astype(np.float64), axis=1), 1.0, rtol=0, atol=1e-6
        ):
            raise RetrievalError("index vectors must be finite and unit length")
        self._chunks = chunks
        self._by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self._vectors = array.copy(order="C")
        self._vectors.flags.writeable = False
        self._spec = spec.model_copy(deep=True)

    @property
    def chunks(self) -> tuple[Chunk, ...]:
        return self._chunks

    @property
    def vectors(self) -> NDArray[np.float32]:
        return self._vectors.copy()

    @property
    def spec(self) -> EmbeddingSpec:
        return self._spec.model_copy(deep=True)

    def chunk(self, chunk_id: str) -> Chunk:
        try:
            return self._by_id[chunk_id]
        except KeyError as error:
            raise RetrievalError(f"unknown indexed chunk: {chunk_id}") from error

    def retrieve(
        self,
        query: str,
        provider: EmbeddingProvider,
        *,
        top_k: int = 5,
    ) -> RetrievalResult:
        validate_request(query, top_k)
        if provider.spec != self._spec:
            raise RetrievalError(
                "query provider does not match the indexed embedding specification"
            )
        started = perf_counter()
        hits: tuple[RetrievalHit, ...] = ()
        if self._chunks:
            raw_query = np.asarray(provider.embed_query(query))
            if raw_query.shape != (self._spec.dimensions,):
                raise RetrievalError("query embedding has the wrong shape")
            vector = normalize_vectors(raw_query[None, :], 1, self._spec.dimensions)[0]
            scores = np.clip(self._vectors @ vector, -1, 1)
            # Do not quantize close scores into ties. Exactly equal scores use chunk IDs.
            ranked = sorted(
                range(len(self._chunks)),
                key=lambda i: (-float(scores[i]), self._chunks[i].chunk_id),
            )
            hits = tuple(
                RetrievalHit(chunk_id=self._chunks[i].chunk_id, score=float(scores[i]))
                for i in ranked[:top_k]
            )
        return RetrievalResult(
            query=query, strategy="vector", hits=hits, elapsed_ms=(perf_counter() - started) * 1000
        )


def build_vector_index(
    documents: Iterable[Document],
    chunks: Iterable[Chunk],
    provider: EmbeddingProvider,
) -> VectorIndex:
    corpus = CorpusIndex(documents, chunks)
    ordered = tuple(corpus.chunks.values())
    spec = provider.spec
    raw = provider.embed_documents(tuple(chunk.text for chunk in ordered))
    vectors = normalize_vectors(raw, len(ordered), spec.dimensions)
    if provider.spec != spec:
        raise RetrievalError("embedding provider specification changed during indexing")
    return VectorIndex(ordered, vectors, spec)
