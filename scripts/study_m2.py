"""Print the exact examples used by docs/ingestion.md; no files are written."""

import codecs

from graphrag_bench.config import ChunkingConfig
from graphrag_bench.ingestion.chunker import chunk_document
from graphrag_bench.ingestion.parser import parse_bytes


def main() -> None:
    text = "Alpha. Beta. Gamma. Delta."
    parsed = parse_bytes(text.encode("utf-8"), relative_path="sentences.txt")
    config = ChunkingConfig(max_chars=40, max_units=2, overlap_units=1)
    print("Example 1: overlapping sentence windows")
    print(f"Document: {parsed.document.text!r}")
    print(f"Config: {config.model_dump(exclude={'schema_version'})}")
    for chunk in chunk_document(parsed.document, config):
        print(f"chunk {chunk.ordinal}: [{chunk.start}, {chunk.end}) {chunk.text!r}")

    raw = codecs.BOM_UTF8 + "α.\r\nBeta.".encode()
    parsed = parse_bytes(raw, relative_path="unicode.txt")
    config = ChunkingConfig(max_units=1, overlap_units=0)
    print("\nExample 2: Unicode, BOM, and CRLF provenance")
    print(f"Raw bytes: {len(raw)}; stored characters: {len(parsed.document.text)}")
    for chunk in chunk_document(parsed.document, config):
        bom_bytes = len(codecs.BOM_UTF8) if parsed.source.had_utf8_bom else 0
        start = bom_bytes + len(parsed.document.text[: chunk.start].encode("utf-8"))
        end = bom_bytes + len(parsed.document.text[: chunk.end].encode("utf-8"))
        assert raw[start:end].decode("utf-8") == chunk.text
        print(
            f"characters [{chunk.start}, {chunk.end}); raw bytes [{start}, {end}): {chunk.text!r}"
        )


if __name__ == "__main__":
    main()
