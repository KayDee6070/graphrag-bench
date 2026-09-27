"""Explicit model revision, text prefixes, and CPU execution settings."""

import tomllib
from pathlib import Path
from typing import Annotated, Literal

from pydantic import StringConstraints

from graphrag_bench.embeddings.base import EmbeddingError
from graphrag_bench.models import PositiveInt, Record


class EmbeddingConfig(Record):
    model_id: Annotated[
        str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")
    ]
    revision: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
    max_seq_length: PositiveInt = 384
    batch_size: PositiveInt = 16
    cpu_threads: PositiveInt = 4
    device: Literal["cpu"] = "cpu"
    query_prefix: str = ""
    document_prefix: str = ""


def load_embedding_config(path: Path) -> EmbeddingConfig:
    try:
        with path.open("rb") as stream:
            data = tomllib.load(stream)
        if set(data) != {"embedding"}:
            raise EmbeddingError("configuration requires exactly one [embedding] table")
        return EmbeddingConfig.model_validate(data["embedding"])
    except (OSError, UnicodeError, ValueError) as error:
        raise EmbeddingError(f"cannot load embedding configuration from {path}: {error}") from error
