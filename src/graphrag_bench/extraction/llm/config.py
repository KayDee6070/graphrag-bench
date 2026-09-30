"""Pinned local inference and a configurable ontology containing no corpus answers."""

import tomllib
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from graphrag_bench.extraction.config import ExtractionError
from graphrag_bench.extraction.llm.contracts import ProposedExtraction
from graphrag_bench.models import Identifier, PositiveInt, Record, Text, require_unique


class LLMError(ExtractionError):
    """Local inference or replay cannot be completed faithfully."""


class LocalModelConfig(Record):
    """Device and dtype are part of the recipe: changing either changes the fingerprint.

    Reduced precision is not a free speedup. float16 and bfloat16 can produce different
    tokens than float32 for the same prompt, so a recipe that changes dtype must have its
    extraction quality re-measured rather than assumed.
    """

    model_id: Annotated[str, StringConstraints(pattern=r"^[\w.-]+/[\w.-]+$")]
    revision: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
    device: Literal["cpu", "cuda"] = "cpu"
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    cpu_threads: Annotated[int, Field(gt=0, le=32, strict=True)] = 4
    max_input_tokens: PositiveInt = 2048
    max_new_tokens: PositiveInt = 768

    @model_validator(mode="after")
    def reduced_precision_needs_a_gpu(self) -> Self:
        if self.device == "cpu" and self.dtype != "float32":
            raise ValueError("cpu inference in this project is pinned to float32")
        return self


class RelationType(Record):
    predicate: Identifier
    description: Text
    subject_types: tuple[Identifier, ...] = Field(min_length=1)
    object_types: tuple[Identifier, ...] = Field(min_length=1)


class PromptExample(Record):
    source: Text
    response: Text


class LLMExtractionConfig(Record):
    model: LocalModelConfig
    prompt_style: Literal["standard", "compact-v1", "focused-v1", "indexed-v1"] = "standard"
    passage_selection: Literal["full", "relation-cues-v1", "relation-cues-v2"] = "full"
    strip_entity_whitespace: bool = Field(default=False, strict=True)
    entity_types: dict[Identifier, Text] = Field(min_length=1)
    relations: tuple[RelationType, ...] = Field(min_length=1)
    examples: tuple[PromptExample, ...] = Field(default=(), max_length=5)

    @model_validator(mode="after")
    def check_ontology(self) -> Self:
        if self.prompt_style == "compact-v1" and self.examples:
            raise ValueError("compact-v1 is example-free; remove examples explicitly")
        require_unique(tuple(r.predicate for r in self.relations), "ontology predicates")
        for relation in self.relations:
            for types in (relation.subject_types, relation.object_types):
                require_unique(types, "relation endpoint types")
                if set(types) - self.entity_types.keys():
                    raise ValueError("relation endpoint types must be declared in entity_types")
        allowed = {r.predicate: r for r in self.relations}
        for example in self.examples:
            proposed = ProposedExtraction.model_validate_json(example.response)
            entities = {e.id: e for e in proposed.entities}
            if any(
                e.entity_type not in self.entity_types or e.name not in example.source
                for e in entities.values()
            ):
                raise ValueError("prompt example has an unknown entity type or missing source name")
            for relation in proposed.relations:
                rule = allowed.get(relation.predicate)
                subject, object_entity = (
                    entities.get(relation.subject),
                    entities.get(relation.object),
                )
                if (
                    rule is None
                    or subject is None
                    or object_entity is None
                    or subject.entity_type not in rule.subject_types
                    or object_entity.entity_type not in rule.object_types
                    or relation.quote not in example.source
                    or subject.name not in relation.quote
                    or object_entity.name not in relation.quote
                ):
                    raise ValueError("prompt example relation violates the ontology or source")
        return self


def load_llm_config(path: Path) -> LLMExtractionConfig:
    try:
        with path.open("rb") as stream:
            data = tomllib.load(stream)
        if set(data) != {"llm_extraction"}:
            raise ValueError("expected one [llm_extraction] table")
        return LLMExtractionConfig.model_validate(data["llm_extraction"])
    except (OSError, UnicodeError, ValueError) as error:
        raise LLMError(f"cannot load LLM extraction configuration: {error}") from error
