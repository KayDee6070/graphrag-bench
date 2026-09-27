"""The vector space is part of the contract, not just the number of columns."""

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from graphrag_bench.models import JsonValue, PositiveInt, Record, Text


class EmbeddingError(ValueError):
    """An embedding provider cannot produce the requested vectors."""


class EmbeddingSpec(Record):
    provider: Text
    model_id: Text
    revision: Text
    dimensions: PositiveInt
    settings: dict[str, JsonValue]
    versions: dict[str, str]


class EmbeddingProvider(Protocol):
    @property
    def spec(self) -> EmbeddingSpec: ...

    def embed_documents(self, texts: tuple[str, ...]) -> NDArray[np.float32]: ...

    def embed_query(self, text: str) -> NDArray[np.float32]: ...
