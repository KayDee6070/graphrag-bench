"""Replaceable local embedding providers; neural dependencies are imported lazily."""

from graphrag_bench.embeddings.base import EmbeddingError, EmbeddingProvider, EmbeddingSpec

__all__ = ["EmbeddingError", "EmbeddingProvider", "EmbeddingSpec"]
