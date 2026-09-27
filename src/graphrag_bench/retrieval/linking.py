"""Link query names/aliases to graph entities without benchmark labels or model calls."""

import re
from collections import defaultdict
from collections.abc import Iterable

from pydantic import Field

from graphrag_bench.extraction.registry import normalize_name
from graphrag_bench.models import Entity, Identifier, Record, Text


class QueryLink(Record):
    surfaces: tuple[Text, ...] = Field(min_length=1)
    candidate_entity_ids: tuple[Identifier, ...] = Field(min_length=1)


def link_query(query: str, entities: Iterable[Entity]) -> tuple[QueryLink, ...]:
    """Longest non-overlapping normalized surface matches; equal candidate sets merge.

    Surfaces are normalized names, not offsets into the user's original query.
    A shared alias yields multiple candidates, never a last-seen winner.
    """
    normalized = normalize_name(query)
    surfaces: dict[str, set[str]] = defaultdict(set)
    for entity in entities:
        for name in (entity.name, *entity.aliases):
            surfaces[normalize_name(name)].add(entity.entity_id)
    occurrences = [
        (match.start(), match.end(), surface)
        for surface in sorted(surfaces)
        for match in re.finditer(r"(?<![\w-])" + re.escape(surface) + r"(?![\w-])", normalized)
    ]
    selected: list[tuple[int, int, str]] = []
    for start, end, surface in sorted(occurrences, key=lambda m: (-(m[1] - m[0]), m[0], m[2])):
        if not any(
            start < other_end and other_start < end for other_start, other_end, _ in selected
        ):
            selected.append((start, end, surface))
    groups: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for _, _, surface in selected:
        groups[tuple(sorted(surfaces[surface]))].add(surface)
    return tuple(
        QueryLink(surfaces=tuple(sorted(names)), candidate_entity_ids=identifiers)
        for identifiers, names in sorted(groups.items())
    )
