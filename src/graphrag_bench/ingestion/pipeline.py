"""Ingest a source directory and export deterministic, inspectable JSON artifacts."""

from __future__ import annotations

import os
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from pydantic import Field

from graphrag_bench import __version__
from graphrag_bench.config import ChunkingConfig
from graphrag_bench.ingestion.chunker import CHUNKER_VERSION, chunk_document
from graphrag_bench.ingestion.parser import PARSER_VERSION, SUPPORTED_SUFFIXES, parse_file
from graphrag_bench.ingestion.types import IngestionError, SourceRecord
from graphrag_bench.models import Chunk, Document, PositiveInt, Record, Sha256, Text
from graphrag_bench.serialization import json_bytes


@dataclass(frozen=True)
class IngestionBatch:
    documents: tuple[Document, ...]
    chunks: tuple[Chunk, ...]
    sources: tuple[SourceRecord, ...]
    ignored_files: tuple[str, ...]
    config: ChunkingConfig


class IngestionManifest(Record):
    package_version: Text
    parser_version: Text
    chunker_version: Text
    chunking: ChunkingConfig
    document_count: PositiveInt
    chunk_count: PositiveInt
    sources: tuple[SourceRecord, ...] = Field(min_length=1)
    ignored_files: tuple[str, ...]
    artifact_hashes: dict[str, Sha256]


def _discover(root: Path) -> tuple[list[Path], tuple[str, ...]]:
    if not root.is_dir():
        raise IngestionError(f"source directory does not exist: {root}")
    paths: list[Path] = []
    ignored: list[str] = []

    def fail_walk(error: OSError) -> None:
        raise IngestionError(f"cannot enumerate {root}: {error}") from error

    for directory, names, files in os.walk(root, followlinks=False, onerror=fail_walk):
        parent = Path(directory)
        names[:] = sorted(
            name for name in names if not name.startswith(".") and not (parent / name).is_symlink()
        )
        for name in sorted(files):
            path = parent / name
            if name.startswith(".") or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                ignored.append(path.relative_to(root).as_posix())
            else:
                paths.append(path)
    if not paths:
        raise IngestionError(f"no .txt, .md, or .markdown source files found in {root}")
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix()), tuple(sorted(ignored))


def ingest_directory(root: Path, config: ChunkingConfig | None = None) -> IngestionBatch:
    """Read source files only. No fixture loader, gold labels, or network is involved."""
    config = config if config is not None else ChunkingConfig()
    paths, ignored = _discover(root)
    documents: list[Document] = []
    chunks: list[Chunk] = []
    sources: list[SourceRecord] = []
    for path in paths:
        parsed = parse_file(path, source_root=root)
        documents.append(parsed.document)
        sources.append(parsed.source)
        chunks.extend(chunk_document(parsed.document, config, sections=parsed.source.sections))
    return IngestionBatch(tuple(documents), tuple(chunks), tuple(sources), ignored, config)


def write_artifacts(batch: IngestionBatch, output: Path) -> IngestionManifest:
    """Create a new directory; write the manifest last as the completion marker.

    Existing directories are never reused. An I/O failure may leave a partial
    new directory, which must not be consumed without its completed manifest.
    """
    artifacts = {
        "documents.jsonl": b"".join(json_bytes(document) for document in batch.documents),
        "chunks.jsonl": b"".join(json_bytes(chunk) for chunk in batch.chunks),
    }
    manifest = IngestionManifest(
        package_version=__version__,
        parser_version=PARSER_VERSION,
        chunker_version=CHUNKER_VERSION,
        chunking=batch.config,
        document_count=len(batch.documents),
        chunk_count=len(batch.chunks),
        sources=batch.sources,
        ignored_files=batch.ignored_files,
        artifact_hashes={name: sha256(content).hexdigest() for name, content in artifacts.items()},
    )
    try:
        output.mkdir(parents=True, exist_ok=False)
        for name, content in artifacts.items():
            with (output / name).open("xb") as stream:
                stream.write(content)
        with (output / "manifest.json").open("xb") as stream:
            stream.write(json_bytes(manifest))
    except FileExistsError as error:
        raise IngestionError(f"output already exists; choose a new directory: {output}") from error
    except OSError as error:
        raise IngestionError(f"cannot write {output}: {error}") from error
    return manifest


def ingest_to_directory(
    source: Path,
    output: Path,
    config: ChunkingConfig | None = None,
) -> IngestionManifest:
    """Validate all inputs before creating output files."""
    if output.exists() or output.is_symlink():
        raise IngestionError(f"output already exists; choose a new directory: {output}")
    if output.resolve().is_relative_to(source.resolve()):
        raise IngestionError("output directory must be outside the source directory")
    return write_artifacts(ingest_directory(source, config), output)
