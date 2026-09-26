import json
import shutil
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import pytest

from graphrag_bench.config import ChunkingConfig
from graphrag_bench.ingestion.pipeline import ingest_directory, ingest_to_directory
from graphrag_bench.ingestion.types import IngestionError
from graphrag_bench.models import Chunk, Document


def make_sources(root):
    root.mkdir()
    (root / "b.txt").write_bytes(b"Birch uses Cedar.\r\nAnother fact.")
    (root / "a.md").write_text("# Alder\nAlder builds on Birch.\n")
    (root / "ignored.pdf").write_bytes(b"not parsed")
    return root


def test_exports_are_byte_identical_after_relocation(tmp_path):
    source = make_sources(tmp_path / "first-source")
    relocated = tmp_path / "second-source"
    shutil.copytree(source, relocated)
    first, second = tmp_path / "run-1", tmp_path / "run-2"
    manifest = ingest_to_directory(source, first)
    assert ingest_to_directory(relocated, second) == manifest
    for name in ("documents.jsonl", "chunks.jsonl", "manifest.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    assert [item.relative_path for item in manifest.sources] == ["a.md", "b.txt"]
    assert manifest.ignored_files == ("ignored.pdf",)
    for name, digest in manifest.artifact_hashes.items():
        assert sha256((first / name).read_bytes()).hexdigest() == digest
    records = [
        Document.model_validate_json(line)
        for line in (first / "documents.jsonl").read_bytes().splitlines()
    ]
    documents = {document.document_id: document for document in records}
    chunks = [
        Chunk.model_validate_json(line)
        for line in (first / "chunks.jsonl").read_bytes().splitlines()
    ]
    assert len(documents) == manifest.document_count == 2
    assert len(chunks) == manifest.chunk_count == 2
    assert all(documents[c.document_id].text[c.start : c.end] == c.text for c in chunks)
    assert str(tmp_path) not in (first / "manifest.json").read_text()


def test_different_paths_keep_identical_content_as_distinct_documents(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for name in ("first.txt", "second.txt"):
        (source / name).write_text("Same content.")
    batch = ingest_directory(source)
    assert len({document.document_id for document in batch.documents}) == 2
    assert len({document.content_sha256 for document in batch.documents}) == 1


def test_invalid_input_does_not_create_output(tmp_path):
    source = make_sources(tmp_path / "source")
    (source / "broken.txt").write_bytes(b"\xff")
    output = tmp_path / "output"
    with pytest.raises(IngestionError, match="valid UTF-8"):
        ingest_to_directory(source, output)
    assert not output.exists()


def test_existing_output_is_never_overwritten(tmp_path):
    source = make_sources(tmp_path / "source")
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("Keep this file.")
    with pytest.raises(IngestionError, match="output already exists"):
        ingest_to_directory(source, output)
    assert sentinel.read_text() == "Keep this file."
    assert list(output.iterdir()) == [sentinel]


def test_output_cannot_be_inside_source(tmp_path):
    source = make_sources(tmp_path / "source")
    with pytest.raises(IngestionError, match="outside the source directory"):
        ingest_to_directory(source, source / "output")
    assert not (source / "output").exists()


@pytest.mark.parametrize("create", [True, False])
def test_empty_or_missing_source_is_rejected(tmp_path, create):
    source = tmp_path / "source"
    if create:
        source.mkdir()
    with pytest.raises(IngestionError):
        ingest_directory(source)


def test_hidden_directories_and_directory_symlinks_are_not_traversed(tmp_path):
    source = make_sources(tmp_path / "source")
    hidden = source / ".hidden"
    hidden.mkdir()
    (hidden / "broken.txt").write_bytes(b"\xff")
    (source / "cycle").symlink_to(source, target_is_directory=True)
    assert len(ingest_directory(source).documents) == 2


def test_documented_example_configuration(tmp_path):

    source = Path(__file__).resolve().parents[1] / "datasets/examples/ingestion"
    output = tmp_path / "example"
    manifest = ingest_to_directory(source, output, ChunkingConfig())
    assert (manifest.document_count, manifest.chunk_count) == (2, 4)
    assert json.loads((output / "manifest.json").read_text())["chunking"]["max_chars"] == 800


def test_study_script_demonstrates_the_documented_offsets(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/study_m2.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "chunk 0: [0, 12) 'Alpha. Beta.'" in result.stdout
    assert "chunk 2: [13, 26) 'Gamma. Delta.'" in result.stdout
    assert "Raw bytes: 13; stored characters: 9" in result.stdout
    assert "characters [0, 2); raw bytes [3, 6): 'α.'" in result.stdout


def test_all_fixture_sources_pass_through_parsing_and_chunking(corpus, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for document in corpus.documents:
        (source / f"{document.document_id}.txt").write_bytes(document.text.encode("utf-8"))
    batch = ingest_directory(source, ChunkingConfig(max_units=1, overlap_units=0))
    original_by_hash = {
        document.content_sha256: document.document_id for document in corpus.documents
    }
    original_ids = {
        document.document_id: original_by_hash[document.content_sha256]
        for document in batch.documents
    }
    actual = sorted(
        (original_ids[chunk.document_id], chunk.ordinal, chunk.start, chunk.end, chunk.text)
        for chunk in batch.chunks
    )
    expected = sorted(
        (chunk.document_id, chunk.ordinal, chunk.start, chunk.end, chunk.text)
        for chunk in corpus.chunks
    )
    assert len(batch.documents) == 8
    assert len(batch.chunks) == 25
    assert actual == expected
