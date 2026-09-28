"""One answer configuration shared across the four retrieval strategies."""

import tomllib
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from graphrag_bench.extraction.llm.config import LocalModelConfig
from graphrag_bench.models import PositiveInt, Record
from graphrag_bench.retrieval.bm25 import BM25Config
from graphrag_bench.retrieval.graph_config import GraphRetrievalConfig
from graphrag_bench.retrieval.hybrid_config import HybridRetrievalConfig


class GenerationError(ValueError):
    """An answer cannot be generated or replayed faithfully."""


class AnswerRetrievalConfig(Record):
    strategy: Literal["bm25", "vector", "graph", "hybrid"] = "graph"
    top_k: PositiveInt = 5
    bm25: BM25Config = Field(default_factory=BM25Config)
    graph: GraphRetrievalConfig = Field(default_factory=GraphRetrievalConfig)
    hybrid: HybridRetrievalConfig = Field(default_factory=HybridRetrievalConfig)


class GenerationConfig(Record):
    model: LocalModelConfig
    context_tokens: PositiveInt = 1500
    retrieval: AnswerRetrievalConfig = Field(default_factory=AnswerRetrievalConfig)

    @model_validator(mode="after")
    def check_budget(self) -> Self:
        if self.context_tokens >= self.model.max_input_tokens:
            raise ValueError("context_tokens must leave input space for question and instructions")
        return self


def load_generation_config(path: Path, *, overrides: dict | None = None) -> GenerationConfig:
    try:
        with path.open("rb") as stream:
            data = tomllib.load(stream)
        if set(data) != {"generation"}:
            raise ValueError("expected one [generation] table")
        settings = data["generation"]
        settings["retrieval"] = settings.get("retrieval", {}) | (overrides or {})
        return GenerationConfig.model_validate(settings)
    except (OSError, UnicodeError, ValueError, TypeError) as error:
        raise GenerationError(f"cannot load generation configuration: {error}") from error
