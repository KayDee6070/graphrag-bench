"""Conservative name resolution: normalized spelling plus type, with ambiguous aliases."""

import json
import unicodedata
from collections import defaultdict

from graphrag_bench.models import Entity, EvidenceSpan, text_sha256


def normalize_name(name: str) -> str:
    return " ".join(unicodedata.normalize("NFC", name).casefold().split())


def stable_id(prefix: str, *parts: str) -> str:
    return f"{prefix}-{text_sha256(json.dumps(parts, ensure_ascii=False))}"


class EntityRegistry:
    """Different types remain distinct; aliases map to sets, never a last-seen winner."""

    def __init__(self) -> None:
        self.names: dict[str, set[str]] = defaultdict(set)
        self.types: dict[str, str] = {}
        self.aliases: dict[str, set[str]] = defaultdict(set)
        self.mentions: dict[str, set[EvidenceSpan]] = defaultdict(set)

    def declare(self, name: str, entity_type: str, alias: str | None = None) -> str:
        identifier = stable_id("entity", entity_type, normalize_name(name))
        self.names[identifier].add(name)
        self.types[identifier] = entity_type
        if alias and normalize_name(alias) != normalize_name(name):
            self.aliases[identifier].add(alias)
        return identifier

    def candidates(self, name: str, entity_type: str | None = None) -> tuple[str, ...]:
        key = normalize_name(name)
        return tuple(
            identifier
            for identifier in sorted(self.names)
            if (entity_type is None or self.types[identifier] == entity_type)
            and key
            in {normalize_name(n) for n in self.names[identifier] | self.aliases[identifier]}
        )

    def entities(self) -> tuple[Entity, ...]:
        entities = []
        for identifier in sorted(self.names):
            name = min(self.names[identifier])
            aliases: dict[str, str] = {}
            for alias in sorted(self.aliases[identifier]):
                aliases.setdefault(normalize_name(alias), alias)
            entities.append(
                Entity(
                    entity_id=identifier,
                    name=name,
                    entity_type=self.types[identifier],
                    aliases=tuple(sorted(aliases.values())),
                    mentions=tuple(
                        sorted(
                            self.mentions[identifier],
                            key=lambda s: (s.document_id, s.start, s.end, s.chunk_id or ""),
                        )
                    ),
                )
            )
        return tuple(entities)
