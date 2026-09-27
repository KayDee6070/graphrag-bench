"""Cross-record source checks shared by extraction, graph building, and artifact loading."""

from collections.abc import Iterable

from graphrag_bench.models import Chunk, Document, EvidenceSpan, TextSpan, require_unique


class CorpusError(ValueError):
    """Records disagree with their source documents or chunks."""


class CorpusIndex:
    """Index validated source records; offsets always address the original document."""

    def __init__(self, documents: Iterable[Document], chunks: Iterable[Chunk]) -> None:
        documents, chunks = tuple(documents), tuple(chunks)
        try:
            require_unique(tuple(d.document_id for d in documents), "document IDs")
            require_unique(tuple(c.chunk_id for c in chunks), "chunk IDs")
        except ValueError as error:
            raise CorpusError(str(error)) from error
        self.documents = {d.document_id: d for d in sorted(documents, key=lambda d: d.document_id)}
        self.chunks = {c.chunk_id: c for c in sorted(chunks, key=lambda c: c.chunk_id)}
        self.by_document: dict[str, tuple[Chunk, ...]] = {}
        for chunk in chunks:
            self.validate_span(chunk)
        for document_id in self.documents:
            ordered = tuple(
                sorted((c for c in chunks if c.document_id == document_id), key=lambda c: c.ordinal)
            )
            if tuple(c.ordinal for c in ordered) != tuple(range(len(ordered))):
                raise CorpusError(f"chunk ordinals must be contiguous from zero: {document_id}")
            self.by_document[document_id] = ordered

    def validate_span(self, span: TextSpan, *, require_chunk: bool = False) -> None:
        document = self.documents.get(span.document_id)
        if document is None:
            raise CorpusError(f"unknown document: {span.document_id}")
        if document.text[span.start : span.end] != span.text:
            raise CorpusError(f"source text mismatch: {span.document_id}:{span.start}:{span.end}")
        if isinstance(span, EvidenceSpan):
            if span.chunk_id is None:
                if require_chunk:
                    raise CorpusError("evidence requires a source chunk")
                return
            chunk = self.chunks.get(span.chunk_id)
            if chunk is None:
                raise CorpusError(f"unknown chunk: {span.chunk_id}")
            if (
                chunk.document_id != span.document_id
                or span.start < chunk.start
                or span.end > chunk.end
            ):
                raise CorpusError(f"evidence lies outside its source chunk: {span.chunk_id}")

    def evidence(self, document_id: str, start: int, end: int) -> tuple[EvidenceSpan, ...]:
        """Return every chunk containing this entire source span, in stable order."""
        return tuple(
            EvidenceSpan(
                document_id=document_id,
                chunk_id=chunk.chunk_id,
                start=start,
                end=end,
                text=self.documents[document_id].text[start:end],
            )
            for chunk in sorted(self.by_document[document_id], key=lambda c: c.chunk_id)
            if chunk.start <= start < end <= chunk.end
        )
