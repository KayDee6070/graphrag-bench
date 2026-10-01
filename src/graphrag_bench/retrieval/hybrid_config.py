"""Explicit candidate windows, an untuned fusion constant, and untuned method weights."""

import tomllib
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from graphrag_bench.models import NonNegativeInt, PositiveInt, Record
from graphrag_bench.retrieval.vector import RetrievalError


class HybridRetrievalConfig(Record):
    """Weights default to 1.0, which is exactly unweighted reciprocal rank fusion.

    Weights exist because the candidate windows and rank constant cannot change the
    relative order of two single-method candidates: 1/(c+g) > 1/(c+v) iff g < v, for any
    c. Scaling one method's contribution is the only way to express that its candidates
    should carry less weight than the other's. The default values are not tuned.
    """

    method: Literal["rrf"] = "rrf"
    rank_constant: NonNegativeInt = 60
    vector_candidates: PositiveInt = 20
    graph_candidates: PositiveInt = 20
    vector_weight: float = Field(default=1.0, ge=0)
    graph_weight: float = Field(default=1.0, ge=0)

    @model_validator(mode="after")
    def one_method_must_contribute(self) -> Self:
        if self.vector_weight == 0 and self.graph_weight == 0:
            raise ValueError("at least one fusion weight must be greater than zero")
        return self


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
