"""Bounded graph retrieval settings; defaults are explicit and have not been benchmark-tuned."""

import tomllib
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from graphrag_bench.models import Identifier, PositiveInt, Record, require_unique
from graphrag_bench.retrieval.vector import RetrievalError


class GraphRetrievalConfig(Record):
    max_hops: Annotated[int, Field(ge=0, le=2, strict=True)] = 2
    direction: Literal["outgoing", "incoming", "both"] = "both"
    max_neighbors: PositiveInt = 16
    max_assertions_per_neighbor: PositiveInt = 4
    max_paths: PositiveInt = 256
    max_seeds: PositiveInt = 16
    include_mentions: bool = True
    predicates: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def check_limits(self) -> Self:
        if self.max_paths < self.max_seeds:
            raise ValueError("max_paths must be at least max_seeds so every seed can be admitted")
        require_unique(self.predicates, "predicate filters")
        return self


def load_graph_retrieval_config(
    path: Path | None = None,
    *,
    overrides: dict | None = None,
) -> GraphRetrievalConfig:
    try:
        settings = {}
        if path is not None:
            with path.open("rb") as stream:
                data = tomllib.load(stream)
            if set(data) != {"graph_retrieval"}:
                raise ValueError("configuration requires exactly one [graph_retrieval] table")
            settings = data["graph_retrieval"]
        return GraphRetrievalConfig.model_validate(settings | (overrides or {}))
    except (OSError, UnicodeError, ValueError, TypeError) as error:
        raise RetrievalError(f"cannot load graph retrieval configuration: {error}") from error
