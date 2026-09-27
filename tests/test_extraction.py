from pathlib import Path

import pytest
from pydantic import ValidationError

from graphrag_bench.config import ChunkingConfig
from graphrag_bench.extraction import extract, load_rules
from graphrag_bench.extraction.config import ExtractionError, ExtractionRule, ExtractionRules
from graphrag_bench.extraction.registry import EntityRegistry
from graphrag_bench.ingestion.chunker import chunk_document
from graphrag_bench.models import Document, text_sha256

RULES_PATH = Path(__file__).resolve().parents[1] / "configs" / "extraction.toml"


@pytest.fixture
def rules():
    return load_rules(RULES_PATH)


def source(text, identifier="document"):
    return Document(
        document_id=identifier,
        title=identifier,
        text=text,
        content_sha256=text_sha256(text),
        source_uri=f"{identifier}.txt",
    )


def run(text, rules, config=None):
    document = source(text)
    chunks = chunk_document(document, config)
    return extract((document,), chunks, rules)


def named_relations(result):
    names = {e.entity_id: e.name for e in result.entities}
    return {(names[r.subject_id], r.predicate, names[r.object_id]) for r in result.relations}


def test_extract_matches_manual_fixture_from_source_documents_only(corpus, rules):
    chunks = tuple(
        chunk
        for doc in corpus.documents
        for chunk in chunk_document(doc, ChunkingConfig(max_units=1, overlap_units=0))
    )
    # Only documents, generated chunks, and syntax rules reach the extractor.
    result = extract(corpus.documents, chunks, rules)
    assert {(e.name, e.entity_type, e.aliases) for e in result.entities} == {
        (e.name, e.entity_type, e.aliases) for e in corpus.entities
    }
    predicted_names = {e.entity_id: e.name for e in result.entities}
    gold_names = {e.entity_id: e.name for e in corpus.entities}

    def assertions(relations, names):
        return sorted(
            (
                names[r.subject_id],
                r.predicate,
                names[r.object_id],
                tuple((s.document_id, s.start, s.end, s.text) for s in r.evidence),
            )
            for r in relations
        )

    assert assertions(result.relations, predicted_names) == assertions(corpus.relations, gold_names)
    assert len(result.entities) == 15
    assert len(result.relations) == 23
    assert [i.span.text for i in result.issues] == [
        "Cedar contains 1200 examples.",
        "Fern contains 800 examples.",
    ]
    for entity in result.entities:
        assert entity.mentions
        if "Base" in entity.aliases:
            assert sum(m.text == "Base" for m in entity.mentions) == 1
    cedar = next(e for e in result.entities if e.name == "Cedar")
    assert any(m.document_id == "doc-cedar" and m.start > 39 for m in cedar.mentions)


def test_predictions_do_not_depend_on_input_or_rule_order(corpus, rules):
    chunks = tuple(c for d in corpus.documents for c in chunk_document(d))
    first = extract(corpus.documents, chunks, rules)
    reversed_rules = ExtractionRules(rules=tuple(reversed(rules.rules)))
    second = extract(reversed(corpus.documents), reversed(chunks), reversed_rules)
    assert first == second
    assert rules.fingerprint == reversed_rules.fingerprint


def test_alias_ambiguity_and_unknown_subject_never_guess(rules):
    result = run(
        "North (also called Base) is a language model.\n"
        "South (also called Base) is a language model.\n"
        "Base uses the Light reranker.\n"
        "Missing uses the Light reranker.\n",
        rules,
    )
    assert result.relations == ()
    assert {e.name for e in result.entities} == {"North", "South", "Light"}
    assert {i.code for i in result.issues} == {
        "ambiguous_entity",
        "ambiguous_mention",
        "unknown_entity",
    }


def test_unique_alias_resolves_without_creating_a_duplicate(rules):
    result = run(
        "Zephyr (ZP) is an encoder evaluated on the Ocean dataset.\n"
        "Orbit is a language model.\n"
        "Orbit uses the ZP encoder.\n"
        "ZP is evaluated using the Rank metric.\n",
        rules,
    )
    assert named_relations(result) == {
        ("Zephyr", "EVALUATED_ON", "Ocean"),
        ("Orbit", "USES", "Zephyr"),
        ("Zephyr", "MEASURED_BY", "Rank"),
    }
    assert "ZP" not in {e.name for e in result.entities}
    assert result.issues == ()


@pytest.mark.parametrize(
    "statement",
    [
        "Orbit does not use the Light reranker.",
        "If Orbit uses the Light reranker, it may improve.",
        "Orbit may use the Light reranker.",
        "Someone says Orbit uses the Light reranker.",
        "Orbit uses the Light reranker. Another sentence.",
    ],
)
def test_nonmatching_statements_do_not_create_positive_assertions(statement, rules):
    result = run("Orbit is a language model.\n" + statement, rules)
    assert result.relations == ()
    assert result.issues[0].code == "unmatched_statement"


def test_overlapping_chunks_share_one_assertion_but_keep_all_chunk_references(rules):
    document = source(
        "Orbit is a language model.\nOrbit uses the Light reranker.\nOrbit targets search."
    )
    overlapping = chunk_document(document, ChunkingConfig(max_units=2, overlap_units=1))
    separate = chunk_document(document, ChunkingConfig(max_units=1, overlap_units=0))
    first = extract((document,), overlapping, rules)
    second = extract((document,), separate, rules)
    assert len(first.relations) == len(second.relations) == 2
    assert {r.assertion_id for r in first.relations} == {r.assertion_id for r in second.relations}
    uses = next(r for r in first.relations if r.predicate == "USES")
    assert len(uses.evidence) == 2
    assert len({e.chunk_id for e in uses.evidence}) == 2


def test_repeated_claims_across_documents_remain_distinct(rules):
    documents = (
        source("Ocean reports the Rank metric.", "a"),
        source("Ocean reports the Rank metric.", "b"),
    )
    chunks = tuple(c for d in documents for c in chunk_document(d))
    result = extract(documents, chunks, rules)
    assert len(result.entities) == 2
    assert len(result.relations) == 2
    assert len({r.assertion_id for r in result.relations}) == 2


def test_same_spelling_different_types_do_not_merge(rules):
    result = run("Echo is a language model.\nEcho is an encoder.\nEcho targets search.\n", rules)
    assert {(e.name, e.entity_type) for e in result.entities if e.name == "Echo"} == {
        ("Echo", "Model"),
        ("Echo", "Encoder"),
    }
    assert result.relations == ()
    assert any(i.code == "ambiguous_entity" for i in result.issues)


def test_fragmented_statement_is_reported_without_extraction(rules):
    result = run(
        "Orbit is a language model evaluated on the Ocean dataset.",
        rules,
        ChunkingConfig(max_chars=15, max_units=1, overlap_units=0),
    )
    assert result.entities == result.relations == ()
    assert [i.code for i in result.issues] == ["uncovered_statement"]


def test_headings_fenced_examples_and_word_substrings_are_not_extracted(rules):
    result = run(
        "# Orbit is a language model.\n"
        "```text\nFake is a language model.\n```\n"
        "Real is a language model.\nSurreal is prose.\n",
        rules,
    )
    assert [e.name for e in result.entities] == ["Real"]
    assert [m.text for m in result.entities[0].mentions] == ["Real"]


def test_normalization_merges_case_spacing_and_unicode_but_preserves_types():
    registry = EntityRegistry()
    first = registry.declare("Café Model", "Model")
    assert registry.declare("CAFE\u0301   model", "Model") == first
    assert registry.declare("Café Model", "Dataset") != first
    assert registry.candidates("café model", "Model") == (first,)


def test_custom_ontology_works_without_changing_extractor():
    rules = ExtractionRules(
        rules=(
            ExtractionRule(
                rule_id="author",
                pattern=r"(?P<subject>\w+) wrote (?P<object>\w+)\.",
                subject_type="Author",
                object_type="Paper",
                predicate="AUTHORED",
                declares_subject=True,
            ),
        )
    )
    result = run("Ada wrote Notebook.", rules)
    assert named_relations(result) == {("Ada", "AUTHORED", "Notebook")}


def test_multiple_matching_rules_are_reported(rules):
    duplicate = rules.rules[0].model_copy(update={"rule_id": "duplicate"})
    result = run(
        "Orbit is a retrieval method built on the North model.",
        ExtractionRules(rules=(*rules.rules, duplicate)),
    )
    assert result.entities == result.relations == ()
    assert result.issues[0].code == "multiple_rules"


@pytest.mark.parametrize(
    "updates",
    [
        {"pattern": "("},
        {"pattern": "plain"},
        {"object_type": None},
        {"declares_subject": True, "subject_type": None},
    ],
)
def test_invalid_rule_rejected(rules, updates):
    values = rules.rules[0].model_dump() | updates
    with pytest.raises(ValidationError):
        ExtractionRule.model_validate(values)


def test_malformed_rule_file_reports_path(tmp_path):
    path = tmp_path / "rules.toml"
    path.write_text("[[rules", encoding="utf-8")
    with pytest.raises(ExtractionError, match="rules.toml"):
        load_rules(path)
