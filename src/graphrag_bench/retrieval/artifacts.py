"""Portable dense indexes bound to exact ingestion artifacts and an embedding space."""

import io
import platform
from hashlib import sha256
from pathlib import Path

import numpy as np

from graphrag_bench import __version__
from graphrag_bench.embeddings.base import EmbeddingProvider, EmbeddingSpec
from graphrag_bench.ingestion.reader import load_ingestion
from graphrag_bench.models import Identifier, NonNegativeInt, Record, Sha256, Text, require_unique
from graphrag_bench.retrieval.vector import (
    VECTOR_VERSION,
    RetrievalError,
    VectorIndex,
    build_vector_index,
)
from graphrag_bench.serialization import json_bytes


class VectorRows(Record):
    chunk_ids: tuple[Identifier, ...]


class VectorManifest(Record):
    package_version: Text
    index_version: Text
    python_version: Text
    numpy_version: Text
    embedding: EmbeddingSpec
    chunk_count: NonNegativeInt
    input_hashes: dict[str, Sha256]
    artifact_hashes: dict[str, Sha256]


def validate_output(source: Path, output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise RetrievalError(f"output already exists; choose a new directory: {output}")
    if output.resolve().is_relative_to(source.resolve()):
        raise RetrievalError("output must be outside the ingestion directory")


def build_vector_to_directory(
    source: Path,
    output: Path,
    provider: EmbeddingProvider,
) -> VectorManifest:
    validate_output(source, output)
    batch, input_hashes = load_ingestion(source)
    index = build_vector_index(batch.documents, batch.chunks, provider)
    buffer = io.BytesIO()
    np.save(buffer, index.vectors, allow_pickle=False)
    artifacts = {
        "vectors.npy": buffer.getvalue(),
        "rows.json": json_bytes(VectorRows(chunk_ids=tuple(c.chunk_id for c in index.chunks))),
    }
    manifest = VectorManifest(
        package_version=__version__,
        index_version=VECTOR_VERSION,
        python_version=platform.python_version(),
        numpy_version=np.__version__,
        embedding=index.spec,
        chunk_count=len(index.chunks),
        input_hashes=input_hashes,
        artifact_hashes={name: sha256(data).hexdigest() for name, data in artifacts.items()},
    )
    try:
        output.mkdir(parents=True, exist_ok=False)
        for name, data in artifacts.items():
            with (output / name).open("xb") as stream:
                stream.write(data)
        with (output / "manifest.json").open("xb") as stream:
            stream.write(json_bytes(manifest))
    except OSError as error:
        raise RetrievalError(f"cannot write vector index to {output}: {error}") from error
    return manifest


def load_vector_index(directory: Path, ingestion_directory: Path) -> VectorIndex:
    """Load and verify without loading the neural model or importing its dependencies."""
    try:
        manifest = VectorManifest.model_validate_json((directory / "manifest.json").read_bytes())
        if manifest.index_version != VECTOR_VERSION:
            raise RetrievalError(f"unsupported vector index version: {manifest.index_version}")
        expected = {"vectors.npy", "rows.json"}
        if set(manifest.artifact_hashes) != expected:
            raise RetrievalError("vector manifest must hash vectors.npy and rows.json")
        raw = {name: (directory / name).read_bytes() for name in sorted(expected)}
        if {n: sha256(data).hexdigest() for n, data in raw.items()} != manifest.artifact_hashes:
            raise RetrievalError("vector artifact checksum mismatch")
        batch, input_hashes = load_ingestion(ingestion_directory)
        if input_hashes != manifest.input_hashes:
            raise RetrievalError("vector index was built from different ingestion artifacts")
        rows = VectorRows.model_validate_json(raw["rows.json"])
        require_unique(rows.chunk_ids, "vector row chunk IDs")
        chunks = tuple(sorted(batch.chunks, key=lambda c: c.chunk_id))
        if rows.chunk_ids != tuple(chunk.chunk_id for chunk in chunks):
            raise RetrievalError("vector rows must identify every source chunk in sorted order")
        if manifest.chunk_count != len(chunks):
            raise RetrievalError("vector manifest count mismatch")
        buffer = io.BytesIO(raw["vectors.npy"])
        vectors = np.load(buffer, allow_pickle=False)
        if not isinstance(vectors, np.ndarray) or buffer.read():
            raise RetrievalError("expected exactly one NumPy array")
        return VectorIndex(chunks, vectors, manifest.embedding)
    except (OSError, ValueError, EOFError) as error:
        raise RetrievalError(f"cannot load vector index from {directory}: {error}") from error
