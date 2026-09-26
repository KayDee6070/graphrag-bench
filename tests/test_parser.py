import codecs
from hashlib import sha256

import pytest

from graphrag_bench.ingestion.parser import parse_bytes, parse_file
from graphrag_bench.ingestion.types import IngestionError


def test_parser_preserves_text_and_tracks_raw_bytes():
    text = "Résumé: α uses β.\r\n\r\n  Indentation stays.  \r\n"
    raw = codecs.BOM_UTF8 + text.encode("utf-8")
    parsed = parse_bytes(raw, relative_path="notes/unicode.txt")
    assert parsed.document.text == text
    assert parsed.source.raw_sha256 == sha256(raw).hexdigest()
    assert parsed.document.content_sha256 == sha256(text.encode()).hexdigest()
    assert parsed.source.had_utf8_bom is True
    assert parsed.source.raw_bytes == len(raw)
    assert parsed.source.sections[0].end == len(text)
    assert parsed.document.source_uri == "corpus:///notes/unicode.txt"


def test_identity_tracks_relative_path_and_content_not_bom():
    original = parse_bytes(b"Alder uses Birch.", relative_path="a.txt")
    repeated = parse_bytes(b"Alder uses Birch.", relative_path="a.txt")
    bom = parse_bytes(codecs.BOM_UTF8 + b"Alder uses Birch.", relative_path="a.txt")
    renamed = parse_bytes(b"Alder uses Birch.", relative_path="b.txt")
    changed = parse_bytes(b"Alder uses Elm.", relative_path="a.txt")
    assert repeated == original
    assert bom.document == original.document
    assert bom.source.raw_sha256 != original.source.raw_sha256
    assert renamed.document.document_id != original.document.document_id
    assert changed.document.document_id != original.document.document_id


def test_markdown_sections_keep_headings_and_ignore_fenced_code():
    text = (
        "Preamble.\n# Paper #\nOverview.\n## Method\nMethod text.\n"
        "### Detail\n```python\n# Not a heading\n```\n"
        "## Results\nResult text.\n"
    )
    parsed = parse_bytes(text.encode(), relative_path="paper.md")
    sections = parsed.source.sections
    assert parsed.document.title == "Paper"
    assert [section.title for section in sections] == [
        None,
        "Paper",
        "Paper / Method",
        "Paper / Method / Detail",
        "Paper / Results",
    ]
    assert "".join(text[section.start : section.end] for section in sections) == text
    assert text[sections[-1].start :].startswith("## Results")
    assert parsed.document.text == text


@pytest.mark.parametrize("marker", ["````", "~~~~"])
def test_shorter_fence_does_not_end_code_block(marker):
    text = f"# Real\n{marker}\n### Code\n{marker[:3]}\n# Still code\n{marker}\n## End\n"
    parsed = parse_bytes(text.encode(), relative_path="fences.markdown")
    assert [section.title for section in parsed.source.sections] == ["Real", "Real / End"]


def test_plain_text_does_not_interpret_markdown_headings():
    parsed = parse_bytes(b"# Literal text\nContent.", relative_path="my note.TXT")
    assert parsed.document.title == "my note"
    assert parsed.document.source_uri == "corpus:///my%20note.TXT"
    assert parsed.source.sections[0].title is None


@pytest.mark.parametrize(
    ("raw", "name", "message"),
    [
        (b"", "empty.txt", "empty or whitespace-only"),
        (b" \r\n\t", "blank.txt", "empty or whitespace-only"),
        (codecs.BOM_UTF8, "bom.txt", "empty or whitespace-only"),
        (b"\xff", "encoding.txt", "valid UTF-8"),
        (b"binary\x00data", "binary.txt", "NUL bytes"),
        (b"text", "paper.pdf", "unsupported file format"),
        (b"text", "../outside.txt", "normalized path inside the corpus"),
        (b"text", "/absolute.txt", "normalized path inside the corpus"),
        (b"text", "a//b.txt", "normalized path inside the corpus"),
    ],
)
def test_invalid_sources_are_rejected(raw, name, message):
    with pytest.raises(IngestionError, match=message):
        parse_bytes(raw, relative_path=name)


def test_parse_file_checks_root_and_symlinks(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("Outside.")
    with pytest.raises(IngestionError, match="outside source_root"):
        parse_file(outside, source_root=source)
    link = source / "link.txt"
    link.symlink_to(outside)
    with pytest.raises(IngestionError, match="symlinked input"):
        parse_file(link, source_root=source)


def test_missing_file_has_context(tmp_path):
    with pytest.raises(IngestionError, match="cannot read"):
        parse_file(tmp_path / "missing.txt", source_root=tmp_path)


def test_empty_markdown_heading_has_a_nonempty_label():
    parsed = parse_bytes(b"# ###\nBody.", relative_path="empty-heading.md")
    assert parsed.document.title == "Untitled section"
    assert parsed.source.sections[0].title == "Untitled section"
