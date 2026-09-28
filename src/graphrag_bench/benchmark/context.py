"""Greedy whole-passage selection under an exact shared tokenizer budget."""

from collections.abc import Callable, Iterable
from typing import Protocol

from graphrag_bench.benchmark.dataset import BenchmarkError
from graphrag_bench.corpus import CorpusIndex
from graphrag_bench.models import (
    Identifier,
    NonNegativeInt,
    PositiveInt,
    Record,
    RetrievalResult,
    TextSpan,
)

CONTEXT_VERSION = "source-union-greedy-v1"


class TokenCounter(Protocol):
    def count_tokens(self, text: str) -> int: ...


class ContextPiece(TextSpan):
    chunk_id: Identifier


class SelectedContext(Record):
    text: str
    pieces: tuple[ContextPiece, ...]
    selected_chunk_ids: tuple[Identifier, ...]
    skipped_budget: tuple[Identifier, ...]
    skipped_overlap: tuple[Identifier, ...]
    token_count: NonNegativeInt
    max_tokens: PositiveInt


def merge_intervals(intervals: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return tuple(merged)


def _uncovered(start: int, end: int, covered: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    result = []
    cursor = start
    for left, right in merge_intervals(covered):
        if right <= cursor:
            continue
        if left >= end:
            break
        if cursor < left:
            result.append((cursor, left))
        cursor = max(cursor, right)
    if cursor < end:
        result.append((cursor, end))
    return result


def render_context(pieces: Iterable[ContextPiece]) -> str:
    return "\n\n".join(f"[{p.document_id}:{p.start}:{p.end}]\n{p.text}" for p in pieces)


def assemble_context(
    result: RetrievalResult,
    corpus: CorpusIndex,
    counter: TokenCounter,
    *,
    max_tokens: int,
    renderer: Callable[[Iterable[ContextPiece]], str] = render_context,
) -> SelectedContext:
    if type(max_tokens) is not int or max_tokens < 1:
        raise BenchmarkError("max_tokens must be a positive integer")
    pieces: list[ContextPiece] = []
    selected, skipped_budget, skipped_overlap = [], [], []
    text, tokens = "", 0
    for hit in result.hits:
        chunk = corpus.chunks.get(hit.chunk_id)
        if chunk is None:
            raise BenchmarkError(f"unknown retrieved chunk: {hit.chunk_id}")
        uncovered = _uncovered(
            chunk.start,
            chunk.end,
            ((p.start, p.end) for p in pieces if p.document_id == chunk.document_id),
        )
        if not uncovered:
            skipped_overlap.append(chunk.chunk_id)
            continue
        additions = [
            ContextPiece(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                start=start,
                end=end,
                text=chunk.text[start - chunk.start : end - chunk.start],
            )
            for start, end in uncovered
        ]
        candidate = renderer((*pieces, *additions))
        count = counter.count_tokens(candidate)
        if type(count) is not int or count < 0 or (candidate.strip() and count == 0):
            raise BenchmarkError(
                "token counter must return a positive integer for nonblank context"
            )
        if count > max_tokens:
            skipped_budget.append(chunk.chunk_id)
            continue
        pieces.extend(additions)
        selected.append(chunk.chunk_id)
        text, tokens = candidate, count
    return SelectedContext(
        text=text,
        pieces=tuple(pieces),
        selected_chunk_ids=tuple(selected),
        skipped_budget=tuple(skipped_budget),
        skipped_overlap=tuple(skipped_overlap),
        token_count=tokens,
        max_tokens=max_tokens,
    )
