"""Parse UTF-8 text and a limited Markdown heading structure without rewriting text."""

from __future__ import annotations

import codecs
import json
import re
from hashlib import sha256
from pathlib import Path, PurePosixPath
from urllib.parse import quote

from graphrag_bench.ingestion.types import IngestionError, ParsedDocument, Section, SourceRecord
from graphrag_bench.models import Document, text_sha256

PARSER_VERSION = "text-markdown-v1"
SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown"}
_HEADING = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*)|[ \t]*)$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


def _headings(text: str) -> list[tuple[int, int, str]]:
    """Find ATX headings outside top-level backtick or tilde fences."""
    headings: list[tuple[int, int, str]] = []
    fence_char, fence_size = "", 0
    offset = 0
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        fence = _FENCE.match(body)
        if fence:
            marker, tail = fence.groups()
            if fence_char:
                if marker[0] == fence_char and len(marker) >= fence_size and not tail.strip():
                    fence_char, fence_size = "", 0
            elif marker[0] != "`" or "`" not in tail:
                fence_char, fence_size = marker[0], len(marker)
        elif not fence_char and (heading := _HEADING.match(body)):
            level = len(heading.group(1))
            title = re.sub(r"(?:^|[ \t]+)#+[ \t]*$", "", heading.group(2) or "").strip()
            headings.append((offset, level, title or "Untitled section"))
        offset += len(line)
    return headings


def _sections(text: str, headings: list[tuple[int, int, str]]) -> tuple[Section, ...]:
    """Partition the whole document; retain parent headings in section labels."""
    if not headings:
        return (Section(start=0, end=len(text)),)
    sections: list[Section] = []
    if headings[0][0] > 0:
        sections.append(Section(start=0, end=headings[0][0]))
    parents: list[tuple[int, str]] = []
    for index, (start, level, title) in enumerate(headings):
        while parents and parents[-1][0] >= level:
            parents.pop()
        parents.append((level, title))
        end = headings[index + 1][0] if index + 1 < len(headings) else len(text)
        sections.append(Section(start=start, end=end, title=" / ".join(t for _, t in parents)))
    return tuple(sections)


def parse_bytes(raw: bytes, *, relative_path: str) -> ParsedDocument:
    """Decode one source. The relative path and canonical text determine its identity.

    UTF-8 BOM removal is the only transformation. CRLF, indentation, Markdown,
    Unicode normalization, and all other characters remain unchanged.
    """
    path = PurePosixPath(relative_path)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != relative_path:
        raise IngestionError("relative_path must be a normalized path inside the corpus")
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise IngestionError(f"unsupported file format: {relative_path}")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise IngestionError(f"{relative_path}: input must be valid UTF-8") from error
    if not text.strip():
        raise IngestionError(f"{relative_path}: document is empty or whitespace-only")
    if "\x00" in text:
        raise IngestionError(f"{relative_path}: NUL bytes are not supported in text")
    source_format = "text" if path.suffix.lower() == ".txt" else "markdown"
    headings = _headings(text) if source_format == "markdown" else []
    title = next(
        (title for _, level, title in headings if level == 1),
        path.stem.strip() or "Untitled document",
    )
    content_hash = text_sha256(text)
    identity = json.dumps([PARSER_VERSION, relative_path, content_hash], separators=(",", ":"))
    document = Document(
        document_id=f"doc-{text_sha256(identity)}",
        title=title,
        text=text,
        content_sha256=content_hash,
        source_uri=f"corpus:///{quote(relative_path, safe='/')}",
    )
    source = SourceRecord(
        document_id=document.document_id,
        relative_path=relative_path,
        format=source_format,
        raw_sha256=sha256(raw).hexdigest(),
        raw_bytes=len(raw),
        had_utf8_bom=raw.startswith(codecs.BOM_UTF8),
        sections=_sections(text, headings),
    )
    return ParsedDocument(document=document, source=source)


def parse_file(path: Path, *, source_root: Path) -> ParsedDocument:
    """Read a local file, using its corpus-relative path instead of an absolute path."""
    if path.is_symlink():
        raise IngestionError(f"symlinked input files are not supported: {path}")
    try:
        relative_path = path.resolve().relative_to(source_root.resolve()).as_posix()
    except ValueError as error:
        raise IngestionError(f"input is outside source_root: {path}") from error
    try:
        return parse_bytes(path.read_bytes(), relative_path=relative_path)
    except OSError as error:
        raise IngestionError(f"cannot read {path}: {error}") from error
