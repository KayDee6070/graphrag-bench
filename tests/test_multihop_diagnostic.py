"""Coverage arithmetic behind the multi-hop diagnostic; no retrieval or model involved."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "diagnose_multihop", ROOT / "scripts" / "diagnose_multihop.py"
)
diagnose = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnose)


def chunk(chunk_id, start, end, document_id="doc-1"):
    return SimpleNamespace(chunk_id=chunk_id, document_id=document_id, start=start, end=end)


def fact(*spans, document_id="doc-1"):
    return SimpleNamespace(
        spans=tuple(
            SimpleNamespace(document_id=document_id, start=start, end=end) for start, end in spans
        )
    )


def test_single_chunk_containing_the_whole_span_is_found():
    chunks = [chunk("a", 0, 100), chunk("b", 90, 200)]

    assert diagnose.single_chunks(fact((10, 50)), chunks) == ["a"]
    assert diagnose.covering_group(fact((10, 50)), chunks) == ["a"]


def test_span_straddling_a_boundary_needs_both_chunks():
    chunks = [chunk("a", 0, 100), chunk("b", 95, 200)]

    assert diagnose.single_chunks(fact((80, 150)), chunks) == []
    assert diagnose.covering_group(fact((80, 150)), chunks) == ["a", "b"]


def test_a_gap_between_chunks_makes_a_span_uncoverable():
    chunks = [chunk("a", 0, 100), chunk("b", 120, 200)]

    assert diagnose.covering_group(fact((80, 150)), chunks) is None


def test_chunks_in_another_document_never_cover_a_span():
    chunks = [chunk("a", 0, 100, document_id="doc-2")]

    assert diagnose.single_chunks(fact((10, 50)), chunks) == []
    assert diagnose.covering_group(fact((10, 50)), chunks) is None


def test_a_fact_with_several_spans_needs_all_of_them():
    chunks = [chunk("a", 0, 100), chunk("b", 100, 200)]

    assert diagnose.covering_group(fact((10, 50), (120, 150)), chunks) == ["a", "b"]
    assert diagnose.covering_group(fact((10, 50), (220, 250)), chunks) is None


def test_group_rank_waits_for_the_deepest_required_chunk():
    hits = [SimpleNamespace(chunk_id=name) for name in ("x", "a", "y", "b")]

    assert diagnose.group_rank(["a", "b"], hits) == 4
    assert diagnose.group_rank(["a"], hits) == 2
    assert diagnose.rank_of(["a", "b"], hits) == 2


@pytest.mark.parametrize("group", [None, []])
def test_group_rank_of_nothing_is_unknown(group):
    assert diagnose.group_rank(group, [SimpleNamespace(chunk_id="a")]) is None


def test_group_rank_is_none_when_one_member_is_never_retrieved():
    hits = [SimpleNamespace(chunk_id="a")]

    assert diagnose.group_rank(["a", "b"], hits) is None
