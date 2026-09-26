from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from graphrag_bench.models import (
    Document,
    Entity,
    EvidenceSpan,
    RelationAssertion,
    RetrievalHit,
    RetrievalResult,
    RunManifest,
    TraversalPath,
    text_sha256,
)


def document_fields() -> dict:
    text = "Résumé: α uses β.\n"
    return {
        "document_id": "doc-unicode",
        "title": "Unicode example",
        "text": text,
        "content_sha256": text_sha256(text),
        "source_uri": "fixture://unicode",
    }


def test_document_roundtrip_preserves_unicode_and_whitespace():
    document = Document(**document_fields())
    restored = Document.model_validate_json(document.model_dump_json())
    assert restored == document
    assert restored.text == "Résumé: α uses β.\n"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("content_sha256", "0" * 64, "does not match document text"),
        ("content_sha256", "not-a-hash", "String should match pattern"),
        ("title", " \n", "String should match pattern"),
        ("document_id", "contains spaces", "String should match pattern"),
        ("schema_version", "2.0", "Input should be '1.0'"),
        ("unexpected", "field", "Extra inputs are not permitted"),
    ],
)
def test_invalid_document_is_rejected(field, value, message):
    with pytest.raises(ValidationError, match=message):
        Document(**{**document_fields(), field: value})


@pytest.mark.parametrize(
    ("start", "end", "text"),
    [(3, 2, "x"), (0, 2, "x"), (-1, 1, "xx"), (False, 1, "x"), (0.0, 1, "x")],
)
def test_invalid_span_bounds_are_rejected(start, end, text):
    with pytest.raises(ValidationError):
        EvidenceSpan(document_id="doc-a", start=start, end=end, text=text)


def test_span_offsets_count_unicode_characters_not_bytes():
    span = EvidenceSpan(document_id="doc-a", start=0, end=2, text="αβ")
    assert span.end == len(span.text)
    assert span.end != len(span.text.encode("utf-8"))


@pytest.mark.parametrize("aliases", [("Birch",), ("BASE", "Base")])
def test_redundant_aliases_are_rejected(aliases):
    with pytest.raises(ValidationError):
        Entity(entity_id="birch", name="Birch", entity_type="Model", aliases=aliases)


def test_entity_types_are_extensible():
    entity = Entity(entity_id="a", name="A", entity_type="ExperimentalHardware")
    assert entity.entity_type == "ExperimentalHardware"


def test_relations_require_source_chunk_provenance():
    with pytest.raises(ValidationError, match="relation evidence requires chunk_id"):
        RelationAssertion(
            assertion_id="a",
            subject_id="one",
            predicate="CUSTOM_RELATION",
            object_id="two",
            evidence=(EvidenceSpan(document_id="doc-a", start=0, end=1, text="x"),),
            extraction_method="manual",
            extraction_version="1",
        )


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_retrieval_scores_are_rejected(score):
    with pytest.raises(ValidationError):
        RetrievalHit(chunk_id="chunk-a", score=score)


def test_result_order_is_explicit_and_duplicate_chunks_are_rejected():
    hit = RetrievalHit(chunk_id="chunk-a", score=0.5)
    with pytest.raises(ValidationError, match="duplicate retrieved chunk IDs"):
        RetrievalResult(query="question", strategy="vector", elapsed_ms=1.0, hits=(hit, hit))
    miss = RetrievalResult(query="unknown entity", strategy="graph", elapsed_ms=0.0)
    assert miss.hits == ()


def test_traversal_path_rejects_missing_edges():
    with pytest.raises(ValidationError, match="one more entity than assertions"):
        TraversalPath(entity_ids=("a", "b", "c"), assertion_ids=("edge-ab",))
    path = TraversalPath(entity_ids=("a", "b", "c"), assertion_ids=("edge-ab", "edge-bc"))
    assert len(path.assertion_ids) == 2


def test_manifest_requires_timezone_and_reproducibility_fields():
    fields = {
        "run_id": "run-1",
        "created_at": datetime(2026, 1, 1, tzinfo=UTC),
        "seed": 42,
        "python_version": "3.12.3",
        "package_version": "0.1.0",
        "corpus_sha256": "a" * 64,
        "benchmark_sha256": "b" * 64,
    }
    manifest = RunManifest(**fields)
    assert RunManifest.model_validate_json(manifest.model_dump_json()) == manifest
    with pytest.raises(ValidationError, match="timezone info"):
        RunManifest(**{**fields, "created_at": datetime(2026, 1, 1)})
