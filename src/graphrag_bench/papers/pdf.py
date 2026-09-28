"""PDF-page provenance in a fixed, explicitly derived Unicode text representation."""

from importlib.util import find_spec
from io import BytesIO

from graphrag_bench.config import ChunkingConfig
from graphrag_bench.ingestion.chunker import chunk_document
from graphrag_bench.ingestion.types import ParsedDocument, Section, SourceRecord
from graphrag_bench.models import Chunk, Document, text_sha256
from graphrag_bench.papers.acquire import validate_pdf_bytes
from graphrag_bench.papers.catalog import Paper, PaperError

PYPDF_VERSION = "6.19.0"
PDF_PARSER_VERSION = "pypdf-6.19.0-plain-pages-v1"
PAGE_SEPARATOR = "\n\f\n"


def extract_pages(raw: bytes) -> tuple[str, ...]:
    """Use plain extraction, preserving its whitespace and hyphenation. No OCR."""
    try:
        import pypdf
    except ImportError as error:
        raise PaperError("PDF parsing requires graphrag-bench[papers] (pypdf==6.19.0)") from error
    if pypdf.__version__ != PYPDF_VERSION:
        raise PaperError(f"PDF parser requires pypdf=={PYPDF_VERSION}")
    if find_spec("fontTools") is not None:
        raise PaperError(
            "plain-pages-v1 requires an environment without optional fontTools; "
            "its presence can change extracted text coordinates"
        )
    try:
        reader = pypdf.PdfReader(BytesIO(raw), strict=True)
        if reader.is_encrypted or not 1 <= len(reader.pages) <= 100:
            raise PaperError("PDF must be unencrypted with 1–100 pages")
        pages = []
        for page in reader.pages:
            contents = page.get_contents()
            if contents is not None and len(contents.get_data()) > 16_000_000:
                raise PaperError("PDF page content exceeds pilot parser limit")
            text = page.extract_text(extraction_mode="plain") or ""
            pages.append(text)
            if len(text) > 1_000_000 or sum(map(len, pages)) > 5_000_000:
                raise PaperError("extracted PDF text exceeds pilot parser limit")
        return tuple(pages)
    except (pypdf.errors.PyPdfError, ValueError, TypeError, KeyError, RecursionError) as error:
        raise PaperError(f"cannot extract PDF text: {error}") from error


def paper_document_id(paper: Paper) -> str:
    identity = text_sha256(f"{PDF_PARSER_VERSION}:{paper.arxiv_version}:{paper.pdf_sha256}")
    return f"paper-{identity}"


def parse_pdf(raw: bytes, paper: Paper) -> ParsedDocument:
    validate_pdf_bytes(raw, paper)
    pages = extract_pages(raw)
    if not any(text.strip() for text in pages):
        raise PaperError("PDF contains no extractable text; OCR is outside this pilot")
    sections = []
    offset = 0
    for number, text in enumerate(pages, 1):
        end = offset + len(text) + len(PAGE_SEPARATOR)
        sections.append(Section(start=offset, end=end, title=f"PDF page {number}"))
        offset = end
    text = "".join(page + PAGE_SEPARATOR for page in pages)
    document = Document(
        document_id=paper_document_id(paper),
        title=paper.title,
        text=text,
        content_sha256=text_sha256(text),
        source_uri=paper.metadata_url,
    )
    return ParsedDocument(
        document,
        SourceRecord(
            document_id=document.document_id,
            relative_path=paper.filename,
            format="pdf",
            raw_sha256=paper.pdf_sha256,
            raw_bytes=len(raw),
            had_utf8_bom=False,
            sections=tuple(sections),
        ),
    )


def chunk_pdf(parsed: ParsedDocument, config: ChunkingConfig) -> tuple[Chunk, ...]:
    pages = {section.title: number for number, section in enumerate(parsed.source.sections, 1)}
    return tuple(
        chunk.model_copy(update={"page": pages[chunk.section]})
        for chunk in chunk_document(parsed.document, config, sections=parsed.source.sections)
    )


def validate_page_chunks(source: SourceRecord, chunks: tuple[Chunk, ...]) -> None:
    """Reject missing/wrong physical PDF-page labels and cross-page chunks."""
    if source.had_utf8_bom:
        raise PaperError("PDF source cannot have a UTF-8 BOM")
    for number, section in enumerate(source.sections, 1):
        if section.title != f"PDF page {number}":
            raise PaperError("PDF sections must identify physical pages in order")
    for chunk in chunks:
        if chunk.page is None or chunk.page > len(source.sections):
            raise PaperError("PDF chunk requires a valid physical page")
        section = source.sections[chunk.page - 1]
        if not section.start <= chunk.start < chunk.end <= section.end:
            raise PaperError("PDF chunk crosses its page boundary")
        if chunk.section != section.title:
            raise PaperError("PDF chunk section/page mismatch")
