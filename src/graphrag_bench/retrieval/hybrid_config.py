"""Explicit candidate windows and an untuned reciprocal rank fusion constant."""

import tomllib
from pathlib import Path
from typing import Literal

from graphrag_bench.models import NonNegativeInt, PositiveInt, Record
from graphrag_bench.retrieval.vector import RetrievalError


class HybridRetrievalConfig(Record):
    method: Literal["rrf"] = "rrf"
    rank_constant: NonNegativeInt = 60
    vector_candidates: PositiveInt = 20
    graph_candidates: PositiveInt = 20


def load_hybrid_retrieval_config(
    path: Path | None = None,
    *,
    overrides: dict | None = None,
) -> HybridRetrievalConfig:
    try:
        settings = {}
        if path is not None:
            with path.open("rb") as stream:
                data = tomllib.load(stream)
            if set(data) != {"hybrid_retrieval"}:
                raise ValueError("configuration requires exactly one [hybrid_retrieval] table")
            settings = data["hybrid_retrieval"]
        return HybridRetrievalConfig.model_validate(settings | (overrides or {}))
    except (OSError, UnicodeError, ValueError, TypeError) as error:
        raise RetrievalError(f"cannot load hybrid retrieval configuration: {error}") from error
