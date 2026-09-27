"""Read persisted M2 artifacts without consulting fixture annotations."""

import json
from hashlib import sha256
from pathlib import Path

from pydantic import ValidationError

from graphrag_bench.corpus import CorpusError, CorpusIndex
from graphrag_bench.ingestion.pipeline import IngestionBatch, IngestionManifest
from graphrag_bench.ingestion.types import IngestionError
from graphrag_bench.models import Chunk, Document


def load_ingestion(directory: Path) -> tuple[IngestionBatch, dict[str, str]]:
    """Verify checksums, counts, coordinates, and source sections before returning data.

    Hashes detect changes against this manifest, not malicious replacement of both.
    Original raw source bytes are not available here and cannot be checked again.
    """
    try:
        raw_manifest = (directory / "manifest.json").read_bytes()
        manifest = IngestionManifest.model_validate_json(raw_manifest)
        expected = {"documents.jsonl", "chunks.jsonl"}
        if set(manifest.artifact_hashes) != expected:
            raise IngestionError("ingestion manifest must hash documents.jsonl and chunks.jsonl")
        raw = {name: (directory / name).read_bytes() for name in sorted(expected)}
        hashes = {name: sha256(content).hexdigest() for name, content in raw.items()}
        if hashes != manifest.artifact_hashes:
            raise IngestionError("ingestion artifact checksum mismatch")
        documents = tuple(
            Document.model_validate(json.loads(line))
            for line in raw["documents.jsonl"].decode("utf-8").splitlines()
        )
        chunks = tuple(
            Chunk.model_validate(json.loads(line))
            for line in raw["chunks.jsonl"].decode("utf-8").splitlines()
        )
        if (len(documents), len(chunks)) != (manifest.document_count, manifest.chunk_count):
            raise IngestionError("ingestion manifest count mismatch")
        corpus = CorpusIndex(documents, chunks)
        source_ids = [source.document_id for source in manifest.sources]
        if len(source_ids) != len(set(source_ids)) or set(source_ids) != set(corpus.documents):
            raise IngestionError("source records must identify every document exactly once")
        paths = [source.relative_path for source in manifest.sources]
        if len(paths) != len(set(paths)):
            raise IngestionError("duplicate source paths")
        for source in manifest.sources:
            position = 0
            for section in source.sections:
                if section.start != position:
                    raise IngestionError("source sections must form a contiguous partition")
                position = section.end
            if position != len(corpus.documents[source.document_id].text):
                raise IngestionError("source sections must cover the document")
            if not corpus.by_document[source.document_id]:
                raise IngestionError("every ingested document must have chunks")
        hashes["manifest.json"] = sha256(raw_manifest).hexdigest()
        return IngestionBatch(
            documents, chunks, manifest.sources, manifest.ignored_files, manifest.chunking
        ), hashes
    except (OSError, UnicodeError, ValueError, ValidationError, CorpusError) as error:
        raise IngestionError(
            f"cannot load ingestion artifacts from {directory}: {error}"
        ) from error
