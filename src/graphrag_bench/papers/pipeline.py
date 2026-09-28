"""Prepare a small paper bundle and verify it offline, optionally against raw PDFs."""

from hashlib import sha256
from pathlib import Path
from typing import Literal

from graphrag_bench import __version__
from graphrag_bench.config import ChunkingConfig
from graphrag_bench.ingestion.pipeline import IngestionBatch, write_artifacts
from graphrag_bench.ingestion.reader import load_ingestion
from graphrag_bench.ingestion.types import ParsedDocument
from graphrag_bench.models import PositiveInt, Record, Sha256, Text
from graphrag_bench.papers.acquire import checked_pdf
from graphrag_bench.papers.annotations import (
    compile_annotations,
    load_annotations,
    render_review,
)
from graphrag_bench.papers.catalog import PaperCatalog, PaperError, load_catalog
from graphrag_bench.papers.pdf import (
    PAGE_SEPARATOR,
    PDF_PARSER_VERSION,
    chunk_pdf,
    paper_document_id,
    parse_pdf,
)
from graphrag_bench.serialization import json_bytes

INGESTION_FILES = ("documents.jsonl", "chunks.jsonl", "manifest.json")
BUNDLE_FILES = frozenset(
    {
        "catalog.json",
        "annotations.json",
        "questions.jsonl",
        "review.md",
        "attribution.md",
        *(f"ingestion/{name}" for name in INGESTION_FILES),
    }
)


class PaperManifest(Record):
    bundle_version: Literal["paper-pilot-v1"] = "paper-pilot-v1"
    package_version: Text
    parser_version: Literal["pypdf-6.19.0-plain-pages-v1"] = PDF_PARSER_VERSION
    catalog_id: Text
    document_count: PositiveInt
    page_count: PositiveInt
    chunk_count: PositiveInt
    question_count: PositiveInt
    review_status: Literal["pending-independent-review"] = "pending-independent-review"
    split: Literal["dev"] = "dev"
    artifact_hashes: dict[str, Sha256]


def build_paper_corpus(
    catalog: PaperCatalog, raw_directory: Path, config: ChunkingConfig
) -> IngestionBatch:
    """Source-only ingestion. Annotations never enter the corpus or its identifiers."""
    parsed = tuple(
        parse_pdf(checked_pdf(raw_directory / paper.filename, paper), paper)
        for paper in sorted(catalog.papers, key=lambda p: p.paper_id)
    )
    return IngestionBatch(
        tuple(p.document for p in parsed),
        tuple(chunk for p in parsed for chunk in chunk_pdf(p, config)),
        tuple(p.source for p in parsed),
        (),
        config,
    )


def attribution(catalog: PaperCatalog) -> str:
    lines = [
        "# Third-party paper attribution",
        "",
        "Paper text and excerpts retain the source licenses below, separate from the MIT code.",
        "Changes: pypdf plain text extraction, page separators, chunking, and selected quotations.",
        "Reference answers are our paraphrases; paper authors do not endorse this project.",
        "The combined pilot annotation collection is distributed under CC BY-NC-SA 4.0.",
        "Original GraphRAG and RAPTOR material remains available under CC BY 4.0.",
        "",
    ]
    for paper in sorted(catalog.papers, key=lambda p: p.paper_id):
        lines.extend(
            [
                f"## {paper.title}",
                "",
                ", ".join(paper.authors) + ".",
                "",
                f"[{paper.arxiv_version}]({paper.metadata_url}) — "
                f"[{paper.license_id}]({paper.license_url}).",
                "",
                f"Source license checked on the versioned arXiv metadata page: "
                f"{paper.metadata_checked_on.isoformat()}.",
                "",
                f"PDF SHA-256: `{paper.pdf_sha256}`; bytes: {paper.pdf_bytes}.",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def prepare_papers(
    catalog_path: Path,
    annotations_path: Path,
    raw_directory: Path,
    output: Path,
    config: ChunkingConfig | None = None,
) -> PaperManifest:
    """No download or inference. Validate inputs before creating a fresh output bundle."""
    if output.exists() or output.is_symlink():
        raise PaperError(f"output already exists: {output}")
    if output.resolve().is_relative_to(raw_directory.resolve()):
        raise PaperError("paper output must be outside the raw directory")
    try:
        catalog = load_catalog(catalog_path)
        annotations = load_annotations(annotations_path)
        batch = build_paper_corpus(catalog, raw_directory, config or ChunkingConfig())
        questions = compile_annotations(catalog, annotations, batch)
        artifacts = {
            "catalog.json": json_bytes(catalog),
            "annotations.json": json_bytes(annotations),
            "questions.jsonl": b"".join(json_bytes(q) for q in questions),
            "review.md": render_review(catalog, annotations).encode("utf-8"),
            "attribution.md": attribution(catalog).encode("utf-8"),
        }
        output.mkdir(parents=True, exist_ok=False)
        write_artifacts(batch, output / "ingestion", parser_version=PDF_PARSER_VERSION)
        for name, content in artifacts.items():
            (output / name).write_bytes(content)
        manifest = PaperManifest(
            package_version=__version__,
            catalog_id=catalog.catalog_id,
            document_count=len(batch.documents),
            chunk_count=len(batch.chunks),
            page_count=sum(len(s.sections) for s in batch.sources),
            question_count=len(questions),
            artifact_hashes={
                name: sha256((output / name).read_bytes()).hexdigest()
                for name in sorted(BUNDLE_FILES)
            },
        )
        # The top-level manifest is the completion marker; partial output is not valid.
        (output / "manifest.json").write_bytes(json_bytes(manifest))
        return manifest
    except (OSError, ValueError) as error:
        raise PaperError(f"cannot prepare paper corpus: {error}") from error


def verify_papers(directory: Path, *, raw_directory: Path | None = None) -> PaperManifest:
    """Check saved coordinates/labels offline; --raw additionally repeats PDF extraction.

    Checksums are integrity checks, not signatures or evidence of semantic correctness.
    No retrieval quality or independent annotation approval is inferred here.
    """
    try:
        manifest = PaperManifest.model_validate_json((directory / "manifest.json").read_bytes())
        if set(manifest.artifact_hashes) != BUNDLE_FILES:
            raise PaperError("paper manifest has unexpected artifact names")
        raw = {name: (directory / name).read_bytes() for name in sorted(BUNDLE_FILES)}
        if {
            name: sha256(data).hexdigest() for name, data in raw.items()
        } != manifest.artifact_hashes:
            raise PaperError("paper artifact checksum mismatch")
        catalog = load_catalog(directory / "catalog.json")
        annotations = load_annotations(directory / "annotations.json")
        batch, _ = load_ingestion(directory / "ingestion")
        papers = {paper.filename: paper for paper in catalog.papers}
        documents = {d.document_id: d for d in batch.documents}
        if set(papers) != {s.relative_path for s in batch.sources}:
            raise PaperError("paper catalog and source files differ")
        rebuilt_chunks = []
        for source in batch.sources:
            paper = papers[source.relative_path]
            doc = documents[source.document_id]
            if (
                source.format != "pdf"
                or source.raw_sha256 != paper.pdf_sha256
                or source.raw_bytes != paper.pdf_bytes
                or doc.title != paper.title
                or doc.source_uri != paper.metadata_url
                or doc.document_id != paper_document_id(paper)
            ):
                raise PaperError("paper source provenance mismatch")
            if any(
                not doc.text[section.start : section.end].endswith(PAGE_SEPARATOR)
                for section in source.sections
            ):
                raise PaperError("PDF page separator mismatch")
            rebuilt_chunks.extend(chunk_pdf(ParsedDocument(doc, source), batch.config))
        if tuple(rebuilt_chunks) != batch.chunks:
            raise PaperError("paper chunk reconstruction mismatch")
        questions = compile_annotations(catalog, annotations, batch)
        if b"".join(json_bytes(q) for q in questions) != raw["questions.jsonl"]:
            raise PaperError("compiled paper annotations mismatch")
        if render_review(catalog, annotations).encode("utf-8") != raw["review.md"]:
            raise PaperError("paper review worksheet mismatch")
        if attribution(catalog).encode("utf-8") != raw["attribution.md"]:
            raise PaperError("paper attribution mismatch")
        counts = (
            len(batch.documents),
            sum(len(s.sections) for s in batch.sources),
            len(batch.chunks),
            len(questions),
        )
        if counts != (
            manifest.document_count,
            manifest.page_count,
            manifest.chunk_count,
            manifest.question_count,
        ):
            raise PaperError("paper manifest count mismatch")
        if manifest.catalog_id != catalog.catalog_id:
            raise PaperError("paper manifest catalog identity mismatch")
        if raw_directory is not None:
            rebuilt = build_paper_corpus(catalog, raw_directory, batch.config)
            if rebuilt != batch:
                raise PaperError("PDF extraction does not reproduce saved corpus")
        return manifest
    except (OSError, ValueError, KeyError) as error:
        raise PaperError(f"cannot verify paper corpus: {error}") from error
