"""Validated chunking parameters and a small TOML configuration reader."""

import tomllib
from pathlib import Path
from typing import Self

from pydantic import ValidationError, model_validator

from graphrag_bench.models import NonNegativeInt, PositiveInt, Record


class ConfigurationError(ValueError):
    """A configuration file or parameter cannot be used."""


class ChunkingConfig(Record):
    """A unit is a detected sentence or a fragment of an oversized sentence."""

    max_chars: PositiveInt = 800
    max_units: PositiveInt = 5
    overlap_units: NonNegativeInt = 1

    @model_validator(mode="after")
    def require_progress(self) -> Self:
        if self.overlap_units >= self.max_units:
            raise ValueError("overlap_units must be less than max_units")
        return self


def load_chunking_config(
    path: Path | None = None,
    *,
    overrides: dict[str, int] | None = None,
) -> ChunkingConfig:
    """Validate the merged settings: defaults, optional TOML, then explicit overrides."""
    try:
        data = {}
        if path is not None:
            with path.open("rb") as stream:
                data = tomllib.load(stream)
        if set(data) - {"chunking"}:
            raise ConfigurationError("only the [chunking] configuration table is supported")
        parameters = data.get("chunking", {})
        if not isinstance(parameters, dict):
            raise ConfigurationError("chunking must be a TOML table")
        return ChunkingConfig.model_validate({**parameters, **(overrides or {})})
    except (OSError, UnicodeError, tomllib.TOMLDecodeError, ValidationError) as error:
        location = f" at {path}" if path is not None else ""
        raise ConfigurationError(f"invalid configuration{location}: {error}") from error
