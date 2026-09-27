"""Rules describe syntax and open ontology labels, never benchmark answers."""

import json
import re
import tomllib
from pathlib import Path
from typing import Self

from pydantic import Field, model_validator

from graphrag_bench.models import Identifier, Record, Text, require_unique, text_sha256


class ExtractionError(ValueError):
    """Extraction rules or inputs cannot be processed."""


class ExtractionRule(Record):
    rule_id: Identifier
    pattern: Text
    subject_type: Identifier | None = None
    object_type: Identifier | None = None
    predicate: Identifier | None = None
    declares_subject: bool = False

    @model_validator(mode="after")
    def validate_pattern(self) -> Self:
        try:
            groups = re.compile(self.pattern).groupindex
        except re.error as error:
            raise ValueError(f"invalid regex: {error}") from error
        if "subject" not in groups:
            raise ValueError("rule requires a named subject group")
        if self.predicate is not None:
            if "object" not in groups or self.object_type is None:
                raise ValueError("relation rule requires an object group and object_type")
        elif self.object_type is not None or "object" in groups or not self.declares_subject:
            raise ValueError("entity-only rule must declare a subject and have no object")
        if self.declares_subject and self.subject_type is None:
            raise ValueError("subject declaration requires subject_type")
        if "alias" in groups and not self.declares_subject:
            raise ValueError("alias group requires a subject declaration")
        return self


class ExtractionRules(Record):
    rules: tuple[ExtractionRule, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_rules(self) -> Self:
        require_unique(tuple(rule.rule_id for rule in self.rules), "rule IDs")
        return self

    @property
    def fingerprint(self) -> str:
        # Rule order is immaterial: multiple matching rules are rejected, never prioritized.
        values = [r.model_dump(mode="json") for r in sorted(self.rules, key=lambda r: r.rule_id)]
        return text_sha256(json.dumps(values, sort_keys=True, ensure_ascii=False))


def load_rules(path: Path) -> ExtractionRules:
    try:
        with path.open("rb") as stream:
            return ExtractionRules.model_validate(tomllib.load(stream))
    except (OSError, UnicodeError, ValueError) as error:
        raise ExtractionError(f"cannot load extraction rules from {path}: {error}") from error
