import json
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from graphrag_bench.fixtures import (
    FixtureDataset,
    FixtureError,
    FixtureManifest,
    load_fixture,
    read_jsonl,
)
from graphrag_bench.models import Document


def test_fixture_is_complete_and_reproducible(corpus, fixture_root):
    assert load_fixture(fixture_root) == corpus
    assert FixtureDataset.model_validate_json(corpus.model_dump_json()) == corpus
    assert (len(corpus.documents), len(corpus.chunks), len(corpus.entities)) == (8, 25, 15)
    assert (len(corpus.relations), len(corpus.questions)) == (23, 20)
    assert {question.split for question in corpus.questions} == {"fixture"}
    assert {question.question_type for question in corpus.questions} == {
        "factual",
        "relationship",
        "comparison",
    }


@pytest.mark.parametrize(
    "collection", ["documents", "chunks", "entities", "relations", "questions"]
)
def test_duplicate_record_ids_are_rejected(corpus, collection):
    data = corpus.model_dump()
    data[collection] += (data[collection][0],)
    with pytest.raises(ValidationError, match="duplicate .* IDs"):
        FixtureDataset.model_validate(data)


@pytest.mark.parametrize(
    ("collection", "field", "value", "message"),
    [
        ("chunks", "document_id", "missing", "unknown document"),
        ("entities", "entity_type", "missing", "unknown entity type"),
        ("relations", "subject_id", "missing", "unknown relation endpoint"),
        ("relations", "object_id", "missing", "unknown relation endpoint"),
        ("relations", "predicate", "missing", "unknown relation type"),
        ("questions", "relevant_entity_ids", ("missing",), "unknown relevant entity"),
        ("questions", "required_document_count", 2, "minimum across evidence sets"),
        ("questions", "relevant_document_ids", ("doc-birch",), "union of gold evidence"),
    ],
)
def test_invalid_references_and_labels_are_rejected(corpus, collection, field, value, message):
    data = corpus.model_dump()
    data[collection][0][field] = value
    with pytest.raises(ValidationError, match=message):
        FixtureDataset.model_validate(data)


def test_chunk_text_must_match_its_document(corpus):
    data = corpus.model_dump()
    chunk = data["chunks"][0]
    chunk["text"] = "X" + chunk["text"][1:]
    with pytest.raises(ValidationError, match="source text mismatch"):
        FixtureDataset.model_validate(data)


def test_chunk_ordinals_are_unique_within_each_document(corpus):
    data = corpus.model_dump()
    data["chunks"][1]["ordinal"] = data["chunks"][0]["ordinal"]
    with pytest.raises(ValidationError, match="duplicate chunk ordinals"):
        FixtureDataset.model_validate(data)


@pytest.mark.parametrize(
    ("chunk_id", "message"),
    [
        ("missing", "unknown chunk"),
        ("chunk-birch-1", "chunk/document mismatch"),
        ("chunk-alder-2", "evidence outside chunk"),
    ],
)
def test_relation_evidence_must_resolve_inside_its_chunk(corpus, chunk_id, message):
    data = corpus.model_dump()
    data["relations"][0]["evidence"][0]["chunk_id"] = chunk_id
    with pytest.raises(ValidationError, match=message):
        FixtureDataset.model_validate(data)


def test_alternative_evidence_and_hops_are_independent_of_document_count(corpus):
    questions = {question.question_id: question for question in corpus.questions}
    comparison = questions["q17"]
    assert (comparison.required_document_count, comparison.reasoning_hops) == (2, 1)
    alternate = questions["q19"]
    assert (alternate.required_document_count, alternate.reasoning_hops) == (1, 2)
    assert len(alternate.sufficient_evidence_sets) == 2
    assert all(
        span.chunk_id is None
        for question in corpus.questions
        for evidence in question.sufficient_evidence_sets
        for fact in evidence.facts
        for span in fact.spans
    )


def test_fixture_retains_ambiguous_aliases_without_merging_entities(corpus):
    owners = {entity.entity_id for entity in corpus.entities if "Base" in entity.aliases}
    assert owners == {"entity-birch", "entity-elm"}


def test_bridge_question_has_a_provenanced_two_document_chain(corpus):
    question = next(question for question in corpus.questions if question.question_id == "q09")
    relations = {relation.assertion_id: relation for relation in corpus.relations}
    first, second = relations["assertion-alder-1"], relations["assertion-birch-1"]
    assert first.object_id == second.subject_id == "entity-birch"
    assert "birch" not in question.question.casefold()
    assert first.evidence[0].document_id != second.evidence[0].document_id
    gold = [fact.spans[0].text for fact in question.sufficient_evidence_sets[0].facts]
    assert gold == [first.evidence[0].text, second.evidence[0].text]


def test_duplicate_assertions_from_different_sources_are_preserved(corpus):
    repeated = [
        relation
        for relation in corpus.relations
        if (relation.subject_id, relation.predicate, relation.object_id)
        == ("entity-cedar", "REPORTS", "entity-accuracy")
    ]
    assert {relation.evidence[0].document_id for relation in repeated} == {"doc-cedar", "doc-lumen"}


def test_question_groups_cannot_cross_splits(corpus):
    data = corpus.model_dump()
    data["questions"][0]["split"] = "test"
    with pytest.raises(ValidationError, match="question group crosses splits"):
        FixtureDataset.model_validate(data)


def test_fixture_detects_changed_bytes(fixture_root, tmp_path):
    root = tmp_path / "fixture"
    shutil.copytree(fixture_root, root)
    path = root / "corpus/documents.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(FixtureError, match="checksum mismatch: corpus/documents.jsonl"):
        load_fixture(root)


def test_manifest_rejects_unexpected_paths(fixture_root):
    data = json.loads((fixture_root / "manifest.json").read_text())
    data["files"]["../../outside"] = "a" * 64
    with pytest.raises(ValidationError, match="exactly the six fixture data files"):
        FixtureManifest.model_validate(data)


@pytest.mark.parametrize("bad_line", ["\n", "{invalid json}\n", '{"document_id": "incomplete"}\n'])
def test_jsonl_errors_include_filename_and_line(corpus, tmp_path, bad_line):
    path = tmp_path / "broken.jsonl"
    path.write_text(corpus.documents[0].model_dump_json() + "\n" + bad_line)
    with pytest.raises(FixtureError, match=r"broken.jsonl:2:"):
        read_jsonl(path, Document)


@pytest.mark.parametrize("contents", [b"", b"\xff"])
def test_empty_or_invalid_utf8_jsonl_is_rejected(tmp_path, contents):
    path = tmp_path / "broken.jsonl"
    path.write_bytes(contents)
    with pytest.raises(FixtureError):
        read_jsonl(path, Document)


def test_missing_fixture_reports_a_useful_error(tmp_path: Path):
    with pytest.raises(FixtureError, match="manifest.json"):
        load_fixture(tmp_path)
