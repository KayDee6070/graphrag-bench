"""Extract explicit, whole-line statements; never infer unstated or reverse relations."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Literal

from graphrag_bench.corpus import CorpusIndex
from graphrag_bench.extraction.config import ExtractionRule, ExtractionRules
from graphrag_bench.extraction.registry import EntityRegistry, normalize_name, stable_id
from graphrag_bench.models import (
    Chunk,
    Document,
    Entity,
    EvidenceSpan,
    Record,
    RelationAssertion,
    Text,
)

EXTRACTOR_VERSION = "line-rules-v1"


class ExtractionIssue(Record):
    code: Literal[
        "unmatched_statement",
        "uncovered_statement",
        "multiple_rules",
        "invalid_capture",
        "unknown_entity",
        "ambiguous_entity",
        "ambiguous_mention",
    ]
    message: Text
    span: EvidenceSpan


class ExtractionResult(Record):
    entities: tuple[Entity, ...]
    relations: tuple[RelationAssertion, ...]
    issues: tuple[ExtractionIssue, ...]


@dataclass(frozen=True)
class MatchedStatement:
    span: EvidenceSpan
    rule: ExtractionRule
    match: re.Match[str]


def _statements(document: Document) -> Iterator[EvidenceSpan]:
    """Lines are the grammar unit. Ignore blanks, ATX headings, and fenced code."""
    position = 0
    fence: str | None = None
    fence_length = 0
    for line in document.text.splitlines(keepends=True):
        stripped = line.strip()
        start = position + len(line) - len(line.lstrip())
        position += len(line)
        marker = re.match(r"^(`{3,}|~{3,})(.*)$", stripped)
        if fence is not None:
            if (
                marker
                and marker[1][0] == fence
                and len(marker[1]) >= fence_length
                and not marker[2].strip()
            ):
                fence = None
            continue
        if marker:
            fence, fence_length = marker[1][0], len(marker[1])
            continue
        if not stripped or re.match(r"^#{1,6}(?:\s|$)", stripped):
            continue
        yield EvidenceSpan(
            document_id=document.document_id,
            start=start,
            end=start + len(stripped),
            text=stripped,
        )


def _match_statements(
    corpus: CorpusIndex, rules: ExtractionRules, issues: list[ExtractionIssue]
) -> tuple[list[MatchedStatement], list[EvidenceSpan]]:
    compiled = [(rule, re.compile(rule.pattern)) for rule in rules.rules]
    matched: list[MatchedStatement] = []
    statements: list[EvidenceSpan] = []
    for document in corpus.documents.values():
        for span in _statements(document):
            statements.append(span)
            evidence = corpus.evidence(span.document_id, span.start, span.end)
            if not evidence:
                issues.append(
                    ExtractionIssue(
                        code="uncovered_statement",
                        message="No chunk contains the whole statement; extraction skipped.",
                        span=span,
                    )
                )
                continue
            span = evidence[0]
            matches = [(rule, pattern.fullmatch(span.text)) for rule, pattern in compiled]
            matches = [(rule, match) for rule, match in matches if match is not None]
            if len(matches) != 1:
                issues.append(
                    ExtractionIssue(
                        code="unmatched_statement" if not matches else "multiple_rules",
                        message="Expected exactly one matching rule; extraction skipped.",
                        span=span,
                    )
                )
                continue
            rule, match = matches[0]
            required = ("subject", "object") if rule.predicate else ("subject",)
            if any(not (match[name] or "").strip() for name in required):
                issues.append(
                    ExtractionIssue(
                        code="invalid_capture", message="Empty required rule capture.", span=span
                    )
                )
                continue
            matched.append(MatchedStatement(span, rule, match))
    return matched, statements


def _capture_span(item: MatchedStatement, group: str) -> tuple[str, int, int]:
    start, end = item.match.span(group)
    return item.span.document_id, item.span.start + start, item.span.start + end


def _collect_mentions(
    registry: EntityRegistry,
    corpus: CorpusIndex,
    statements: list[EvidenceSpan],
    owners: dict[tuple[str, int, int], set[str]],
    issues: list[ExtractionIssue],
) -> None:
    # Explicit typed captures disambiguate their own occurrences, including aliases.
    for (document_id, start, end), identifiers in owners.items():
        for identifier in identifiers:
            registry.mentions[identifier].update(corpus.evidence(document_id, start, end))
    surfaces = sorted({n for values in registry.names.values() for n in values})
    surfaces += sorted({n for values in registry.aliases.values() for n in values})
    seen: set[tuple[str, int, int]] = set()
    for surface in surfaces:
        tokens = [re.escape(token) for token in surface.split()]
        pattern = re.compile(r"(?<!\w)" + r"\s+".join(tokens) + r"(?!\w)", re.IGNORECASE)
        identifiers = registry.candidates(surface)
        for statement in statements:
            for match in pattern.finditer(statement.text):
                if normalize_name(match[0]) != normalize_name(surface):
                    # Python IGNORECASE includes a few characters that casefold distinguishes.
                    continue
                location = (
                    statement.document_id,
                    statement.start + match.start(),
                    statement.start + match.end(),
                )
                if location in owners or location in seen:
                    continue
                seen.add(location)
                evidence = corpus.evidence(*location)
                if not evidence:
                    continue
                if len(identifiers) == 1:
                    registry.mentions[identifiers[0]].update(evidence)
                else:
                    issues.append(
                        ExtractionIssue(
                            code="ambiguous_mention",
                            message=f"Mention {match[0]!r} has multiple candidate entities.",
                            span=evidence[0],
                        )
                    )


def extract(
    documents: Iterable[Document], chunks: Iterable[Chunk], rules: ExtractionRules
) -> ExtractionResult:
    """Build predictions from source records and syntax rules only; no gold inputs."""
    corpus = CorpusIndex(documents, chunks)
    issues: list[ExtractionIssue] = []
    matched, statements = _match_statements(corpus, rules, issues)
    registry = EntityRegistry()
    owners: dict[tuple[str, int, int], set[str]] = defaultdict(set)

    # Declarations come first so aliases resolve consistently regardless of file order.
    for item in matched:
        if item.rule.declares_subject:
            assert item.rule.subject_type is not None
            alias = item.match.groupdict().get("alias")
            identifier = registry.declare(item.match["subject"], item.rule.subject_type, alias)
            owners[_capture_span(item, "subject")].add(identifier)
            if alias:
                owners[_capture_span(item, "alias")].add(identifier)

    # Typed references introduce new names only if no declaration/alias already matches.
    references = sorted(
        {
            (item.match[group], entity_type)
            for item in matched
            for group, entity_type in (
                ("subject", item.rule.subject_type),
                ("object", item.rule.object_type),
            )
            if entity_type is not None
        }
    )
    for name, entity_type in references:
        if not registry.candidates(name, entity_type):
            registry.declare(name, entity_type)

    relations: list[RelationAssertion] = []
    version = f"{EXTRACTOR_VERSION}:{rules.fingerprint}"
    for item in matched:
        endpoints: dict[str, str] = {}
        for group, entity_type in (
            ("subject", item.rule.subject_type),
            ("object", item.rule.object_type),
        ):
            if group == "object" and item.rule.predicate is None:
                continue
            if group == "subject" and item.rule.declares_subject:
                candidates = (stable_id("entity", entity_type, normalize_name(item.match[group])),)
            else:
                candidates = registry.candidates(item.match[group], entity_type)
            if len(candidates) == 1:
                endpoints[group] = candidates[0]
                owners[_capture_span(item, group)].add(candidates[0])
            else:
                issues.append(
                    ExtractionIssue(
                        code="unknown_entity" if not candidates else "ambiguous_entity",
                        message=f"Cannot uniquely resolve {group} {item.match[group]!r}.",
                        span=item.span,
                    )
                )
        if item.rule.predicate is None or len(endpoints) != 2:
            continue
        span = item.span
        relations.append(
            RelationAssertion(
                assertion_id=stable_id(
                    "assertion",
                    version,
                    span.document_id,
                    str(span.start),
                    str(span.end),
                    span.text,
                    endpoints["subject"],
                    item.rule.predicate,
                    endpoints["object"],
                ),
                subject_id=endpoints["subject"],
                predicate=item.rule.predicate,
                object_id=endpoints["object"],
                evidence=corpus.evidence(span.document_id, span.start, span.end),
                extraction_method="deterministic-rules",
                extraction_version=version,
            )
        )
    _collect_mentions(registry, corpus, statements, owners, issues)
    return ExtractionResult(
        entities=registry.entities(),
        relations=tuple(sorted(relations, key=lambda r: r.assertion_id)),
        issues=tuple(
            sorted(
                issues,
                key=lambda i: (i.span.document_id, i.span.start, i.span.end, i.code, i.message),
            )
        ),
    )
