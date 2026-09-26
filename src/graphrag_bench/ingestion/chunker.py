"""Greedy sentence windows over exact document slices, with bounded overlap."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator

from graphrag_bench.config import ChunkingConfig
from graphrag_bench.ingestion.types import IngestionError, Section
from graphrag_bench.models import Chunk, Document, text_sha256

CHUNKER_VERSION = "sentence-windows-v1"
_ENDING = re.compile(r"""[.!?]+["'”’)\]]*(?=\s|$)""")
_ABBREVIATIONS = {
    "dr.",
    "mr.",
    "mrs.",
    "ms.",
    "prof.",
    "fig.",
    "eq.",
    "sec.",
    "no.",
    "vs.",
    "e.g.",
    "i.e.",
    "al.",
}
Span = tuple[int, int]


def _trim(text: str, start: int, end: int) -> Span:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _paragraphs(text: str, section: Section) -> Iterator[Span]:
    """Blank lines end paragraphs; a single wrapped newline stays inside one."""
    start, offset = section.start, section.start
    for line in text[section.start : section.end].splitlines(keepends=True):
        if not line.strip():
            begin, end = _trim(text, start, offset)
            if begin < end:
                yield begin, end
            start = offset + len(line)
        offset += len(line)
    begin, end = _trim(text, start, section.end)
    if begin < end:
        yield begin, end


def _sentences(text: str, start: int, end: int) -> Iterator[Span]:
    """A deterministic heuristic, not a linguistic sentence tokenizer."""
    cursor = start
    for match in _ENDING.finditer(text, start, end):
        prefix = text[cursor : match.start() + 1]
        token = prefix.rsplit(maxsplit=1)[-1] if prefix.strip() else ""
        if text[match.start()] == "." and (
            token.casefold() in _ABBREVIATIONS or re.fullmatch(r"(?:[A-Za-z]\.)+", token)
        ):
            continue
        begin, stop = _trim(text, cursor, match.end())
        if begin < stop:
            yield begin, stop
        cursor = match.end()
    begin, stop = _trim(text, cursor, end)
    if begin < stop:
        yield begin, stop


def _split_long(text: str, start: int, end: int, max_chars: int) -> Iterator[Span]:
    """Prefer whitespace cuts; split unbroken strings at a character boundary."""
    while end - start > max_chars:
        limit = start + max_chars
        whitespace = list(re.finditer(r"\s+", text[start : limit + 1]))
        stop = start + whitespace[-1].start() if whitespace else limit
        if stop <= start:
            stop = limit
        begin, trimmed_end = _trim(text, start, stop)
        if begin < trimmed_end:
            yield begin, trimmed_end
        start, _ = _trim(text, stop, end)
    if start < end:
        yield start, end


def _windows(units: list[Span], config: ChunkingConfig) -> Iterator[Span]:
    """Pack units; shrink requested overlap until the next window adds new text."""
    cursor = 0
    while cursor < len(units):
        stop = cursor + 1
        while (
            stop < len(units)
            and stop - cursor < config.max_units
            and units[stop][1] - units[cursor][0] <= config.max_chars
        ):
            stop += 1
        yield units[cursor][0], units[stop - 1][1]
        if stop == len(units):
            break
        cursor = max(cursor + 1, stop - config.overlap_units)
        while cursor < stop and units[stop][1] - units[cursor][0] > config.max_chars:
            cursor += 1


def chunk_document(
    document: Document,
    config: ChunkingConfig | None = None,
    *,
    sections: tuple[Section, ...] | None = None,
) -> tuple[Chunk, ...]:
    """Create ordered chunks without crossing sections or blank-line paragraphs.

    Every chunk is a verbatim document slice. Leading/trailing whitespace is
    omitted by adjusting offsets, never by rewriting the slice. IDs bind the
    document, content hash, algorithm, configuration, range, and section label.
    """
    config = config if config is not None else ChunkingConfig()
    sections = sections if sections is not None else (Section(start=0, end=len(document.text)),)
    cursor = 0
    for section in sections:
        if section.start != cursor or section.end > len(document.text):
            raise IngestionError("sections must partition the document without gaps or overlaps")
        cursor = section.end
    if cursor != len(document.text):
        raise IngestionError("sections must cover the entire document")
    config_hash = text_sha256(
        json.dumps(config.model_dump(), sort_keys=True, separators=(",", ":"))
    )
    chunks: list[Chunk] = []
    for section in sections:
        for start, end in _paragraphs(document.text, section):
            units = [
                span
                for begin, stop in _sentences(document.text, start, end)
                for span in _split_long(document.text, begin, stop, config.max_chars)
            ]
            for begin, stop in _windows(units, config):
                identity = json.dumps(
                    [
                        document.document_id,
                        document.content_sha256,
                        CHUNKER_VERSION,
                        config_hash,
                        begin,
                        stop,
                        section.title,
                    ],
                    separators=(",", ":"),
                )
                chunks.append(
                    Chunk(
                        document_id=document.document_id,
                        chunk_id=f"chunk-{text_sha256(identity)}",
                        ordinal=len(chunks),
                        start=begin,
                        end=stop,
                        text=document.text[begin:stop],
                        section=section.title,
                    )
                )
    return tuple(chunks)
