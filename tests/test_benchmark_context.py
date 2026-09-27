import pytest

from graphrag_bench.benchmark.context import assemble_context, render_context
from graphrag_bench.benchmark.dataset import BenchmarkError
from graphrag_bench.corpus import CorpusIndex
from graphrag_bench.models import Chunk, Document, RetrievalHit, RetrievalResult, text_sha256


class CharacterCounter:
    """Test-only counter makes budget arithmetic exactly inspectable."""

    def count_tokens(self, text):
        return len(text)


def sources(intervals=((0, 6), (3, 9), (0, 9)), text="abcdefghi"):
    doc = Document(
        document_id="d",
        title="Source",
        text=text,
        source_uri="test",
        content_sha256=text_sha256(text),
    )
    chunks = tuple(
        Chunk(chunk_id=f"c{i}", document_id="d", start=a, end=b, text=text[a:b], ordinal=i)
        for i, (a, b) in enumerate(intervals)
    )
    return CorpusIndex((doc,), chunks)


def result(*ids):
    return RetrievalResult(
        query="Question",
        strategy="vector",
        hits=tuple(RetrievalHit(chunk_id=i, score=1.0) for i in ids),
        elapsed_ms=0.0,
    )


def test_overlap_removed_by_coordinates_and_context_keeps_original_slices():
    corpus = sources()
    selected = assemble_context(
        result("c0", "c1", "c2"), corpus, CharacterCounter(), max_tokens=100
    )
    assert [(p.start, p.end, p.text) for p in selected.pieces] == [(0, 6, "abcdef"), (6, 9, "ghi")]
    assert selected.selected_chunk_ids == ("c0", "c1")
    assert selected.skipped_overlap == ("c2",)
    assert selected.text == "[d:0:6]\nabcdef\n\n[d:6:9]\nghi"
    assert selected.token_count == len(selected.text)
    for piece in selected.pieces:
        assert corpus.documents[piece.document_id].text[piece.start : piece.end] == piece.text


def test_exact_boundary_includes_source_headers_and_separators():
    corpus = sources()
    exact = len("[d:0:6]\nabcdef")
    accepted = assemble_context(result("c0"), corpus, CharacterCounter(), max_tokens=exact)
    rejected = assemble_context(result("c0"), corpus, CharacterCounter(), max_tokens=exact - 1)
    assert accepted.token_count == exact
    assert rejected.pieces == () and rejected.text == "" and rejected.token_count == 0
    assert rejected.skipped_budget == ("c0",)


def test_oversized_chunk_skipped_without_clipping_and_later_chunk_can_fit():
    corpus = sources(((0, 9), (0, 1)))
    selected = assemble_context(result("c0", "c1"), corpus, CharacterCounter(), max_tokens=9)
    assert selected.skipped_budget == ("c0",)
    assert selected.selected_chunk_ids == ("c1",)
    assert selected.pieces[0].text == "a"


def test_multiple_uncovered_fragments_of_one_chunk_are_admitted_together():
    corpus = sources(((3, 6), (0, 9)))
    selected = assemble_context(result("c0", "c1"), corpus, CharacterCounter(), max_tokens=100)
    assert [(p.start, p.end) for p in selected.pieces] == [(3, 6), (0, 3), (6, 9)]
    tighter = assemble_context(
        result("c0", "c1"), corpus, CharacterCounter(), max_tokens=selected.token_count - 1
    )
    assert tighter.selected_chunk_ids == ("c0",)
    assert tighter.skipped_budget == ("c1",)


def test_budget_counts_whole_rendered_context_not_sum_of_individual_costs():
    class ContextSensitiveCounter:
        def count_tokens(self, text):
            return 1 if "\n\n" not in text else 10

    selected = assemble_context(
        result("c0", "c1"), sources(), ContextSensitiveCounter(), max_tokens=5
    )
    assert selected.selected_chunk_ids == ("c0",)
    assert selected.skipped_budget == ("c1",)


def test_same_text_in_different_documents_preserves_both_source_records():
    documents = tuple(
        Document(
            document_id=i, title=i, text="same", source_uri=i, content_sha256=text_sha256("same")
        )
        for i in ("a", "b")
    )
    chunks = tuple(
        Chunk(
            document_id=d.document_id,
            chunk_id=d.document_id,
            ordinal=0,
            start=0,
            end=4,
            text=d.text,
        )
        for d in documents
    )
    selected = assemble_context(
        result("a", "b"), CorpusIndex(documents, chunks), CharacterCounter(), max_tokens=100
    )
    assert selected.selected_chunk_ids == ("a", "b")
    assert selected.skipped_overlap == ()


def test_unknown_hit_and_invalid_budget_fail():
    with pytest.raises(BenchmarkError, match="unknown retrieved chunk"):
        assemble_context(result("missing"), sources(), CharacterCounter(), max_tokens=10)
    with pytest.raises(BenchmarkError, match="positive integer"):
        assemble_context(result(), sources(), CharacterCounter(), max_tokens=True)
    assert render_context(()) == ""


@pytest.mark.parametrize("count", [-1, 0, True, 1.5])
def test_broken_token_counter_rejected(count):
    class BrokenCounter:
        def count_tokens(self, text):
            return count

    with pytest.raises(BenchmarkError, match="token counter"):
        assemble_context(result("c0"), sources(), BrokenCounter(), max_tokens=10)
