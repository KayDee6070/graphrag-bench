"""A single recorded comparison policy shared by every retrieval strategy."""

import tomllib
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from graphrag_bench.benchmark.dataset import BenchmarkError
from graphrag_bench.config import ChunkingConfig
from graphrag_bench.models import NonNegativeInt, PositiveInt, Record
from graphrag_bench.retrieval.bm25 import BM25Config
from graphrag_bench.retrieval.graph_config import GraphRetrievalConfig
from graphrag_bench.retrieval.hybrid_config import HybridRetrievalConfig

Strategy = Literal["vector", "graph", "hybrid", "bm25"]


class BenchmarkConfig(Record):
    strategies: tuple[Strategy, ...] = ("vector", "graph", "hybrid", "bm25")
    cutoffs: tuple[PositiveInt, ...] = (5, 10)
    max_context_tokens: PositiveInt = 2000
    repeats: PositiveInt = 3
    seed: NonNegativeInt = 0
    chunking: ChunkingConfig = Field(
        default_factory=lambda: ChunkingConfig(max_units=1, overlap_units=0)
    )
    graph: GraphRetrievalConfig = Field(default_factory=GraphRetrievalConfig)
    hybrid: HybridRetrievalConfig = Field(default_factory=HybridRetrievalConfig)
    bm25: BM25Config = Field(default_factory=BM25Config)

    @model_validator(mode="after")
    def check_comparison(self) -> Self:
        if not self.strategies or len(set(self.strategies)) != len(self.strategies):
            raise ValueError("strategies must be nonempty and unique")
        if not self.cutoffs or tuple(sorted(set(self.cutoffs))) != self.cutoffs:
            raise ValueError("cutoffs must be nonempty, unique, and increasing")
        return self


def load_benchmark_config(path: Path | None = None) -> BenchmarkConfig:
    if path is None:
        return BenchmarkConfig()
    try:
        with path.open("rb") as stream:
            data = tomllib.load(stream)
        if set(data) != {"benchmark"}:
            raise ValueError("configuration requires exactly one [benchmark] table")
        return BenchmarkConfig.model_validate(data["benchmark"])
    except (OSError, UnicodeError, ValueError) as error:
        raise BenchmarkError(f"cannot load benchmark configuration: {error}") from error
