from itertools import pairwise

import pytest

from graphrag_bench.config import ChunkingConfig
from graphrag_bench.ingestion.chunker import chunk_document
from graphrag_bench.ingestion.parser import parse_bytes
from graphrag_bench.ingestion.types import IngestionError, Section


def chunks_for(text: str, **parameters):
    document = parse_bytes(text.encode(), relative_path="example.txt").document
    config = ChunkingConfig(**parameters)
    return chunk_document(document, config)


def test_overlap_windows_have_exact_offsets_and_no_redundant_tail():
    chunks = chunks_for("Alpha. Beta. Gamma. Delta.", max_chars=40, max_units=2, overlap_units=1)
    assert [(chunk.start, chunk.end, chunk.text) for chunk in chunks] == [
        (0, 12, "Alpha. Beta."),
        (7, 19, "Beta. Gamma."),
        (13, 26, "Gamma. Delta."),
    ]
    assert [chunk.ordinal for chunk in chunks] == [0, 1, 2]


def test_overlap_shrinks_when_it_would_prevent_new_content():
    chunks = chunks_for("Alpha. Beta. Gamma. Delta.", max_chars=12, max_units=3, overlap_units=2)
    assert [chunk.text for chunk in chunks] == ["Alpha. Beta.", "Beta. Gamma.", "Delta."]
    assert all(first.end < second.end for first, second in pairwise(chunks))


def test_sentences_avoid_common_abbreviations_and_decimal_splits():
    text = "Dr. Smith uses Fig. 2. Accuracy is 0.75. It works!"
    chunks = chunks_for(text, max_units=1, overlap_units=0)
    assert [chunk.text for chunk in chunks] == [
        "Dr. Smith uses Fig. 2.",
        "Accuracy is 0.75.",
        "It works!",
    ]


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
def test_wrapped_lines_stay_together_but_blank_lines_end_paragraphs(newline):
    first = f"One paragraph wraps{newline}onto another line."
    text = f"{first}{newline} \t{newline}Another paragraph."
    chunks = chunks_for(text)
    assert [chunk.text for chunk in chunks] == [first, "Another paragraph."]


def test_unbroken_long_sentence_uses_hard_character_cuts():
    chunks = chunks_for("αβγδεζηθικλ", max_chars=4, max_units=2, overlap_units=1)
    assert [chunk.text for chunk in chunks] == ["αβγδ", "εζηθ", "ικλ"]


def test_long_sentence_prefers_word_boundaries():
    chunks = chunks_for("alpha beta gamma", max_chars=10)
    assert [chunk.text for chunk in chunks] == ["alpha beta", "gamma"]


def test_chunks_never_cross_markdown_sections():
    parsed = parse_bytes(b"# Paper\nIntro.\n## Method\nDetails.\n", relative_path="paper.md")
    chunks = chunk_document(parsed.document, sections=parsed.source.sections)
    assert [chunk.section for chunk in chunks] == ["Paper", "Paper / Method"]
    assert [chunk.text for chunk in chunks] == ["# Paper\nIntro.", "## Method\nDetails."]
    assert all(chunk.page is None for chunk in chunks)


@pytest.mark.parametrize(
    "sections",
    [
        (),
        (Section(start=1, end=6),),
        (Section(start=0, end=4),),
        (Section(start=0, end=5), Section(start=4, end=6)),
    ],
)
def test_invalid_section_partitions_are_rejected(sections):
    document = parse_bytes(b"Alpha.", relative_path="a.txt").document
    with pytest.raises(IngestionError, match="sections must"):
        chunk_document(document, sections=sections)


def test_chunk_ids_bind_document_content_and_configuration():
    document = parse_bytes(b"Alpha. Beta.", relative_path="a.txt").document
    first = chunk_document(document)
    assert first == chunk_document(document)
    changed_config = chunk_document(document, ChunkingConfig(max_chars=801))
    assert first[0].text == changed_config[0].text
    assert first[0].chunk_id != changed_config[0].chunk_id
    other = parse_bytes(b"Alpha. Beta.", relative_path="b.txt").document
    assert first[0].chunk_id != chunk_document(other)[0].chunk_id


@pytest.mark.parametrize("max_chars", [1, 7, 20, 80])
@pytest.mark.parametrize("overlap_units", [0, 1, 2])
def test_chunk_invariants_across_boundaries(max_chars, overlap_units):
    text = "  Dr. Lin measures 0.75.\r\nA veryveryverylongword follows!\n\nαβγ? More text.  "
    parsed = parse_bytes(text.encode(), relative_path="a.txt")
    config = ChunkingConfig(max_chars=max_chars, max_units=3, overlap_units=overlap_units)
    chunks = chunk_document(parsed.document, config)
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    assert all(chunk.text == text[chunk.start : chunk.end] for chunk in chunks)
    assert all(0 < len(chunk.text) <= max_chars for chunk in chunks)
    assert all(
        first.start < second.start and first.end < second.end for first, second in pairwise(chunks)
    )
    covered = {index for chunk in chunks for index in range(chunk.start, chunk.end)}
    assert {index for index, char in enumerate(text) if not char.isspace()} <= covered


def test_sentence_configuration_reproduces_all_m1_reference_spans(corpus):
    config = ChunkingConfig(max_units=1, overlap_units=0)
    actual = [chunk for document in corpus.documents for chunk in chunk_document(document, config)]

    # M1 IDs and the manually assigned Overview label are not parser outputs.
    def source_fields(chunk):
        return chunk.document_id, chunk.ordinal, chunk.start, chunk.end, chunk.text

    assert list(map(source_fields, actual)) == list(map(source_fields, corpus.chunks))
    assert len(actual) == 25
