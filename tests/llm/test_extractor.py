import copy
import json

import pytest

from graphrag_bench.corpus import CorpusIndex
from graphrag_bench.extraction.llm.config import LLMError
from graphrag_bench.extraction.llm.contracts import Completion
from graphrag_bench.extraction.llm.extractor import extract_with_llm, process_responses
from graphrag_bench.graph import build_graph
from graphrag_bench.models import Chunk, Document, text_sha256


def test_extracts_source_checked_multidocument_graph(llm_config, source_pair, make_provider):
    documents, chunks = source_pair
    provider = make_provider()
    result = extract_with_llm(documents, chunks, llm_config, provider)
    assert (len(result.entities), len(result.relations), len(result.issues)) == (3, 2, 0)
    assert not result.rejected_chunks
    graph = build_graph(result.entities, result.relations, documents, chunks)
    nova = graph.lookup("Nova")[0]
    assert len(nova.mentions) == 2
    assert {r.predicate for r in graph.outgoing(nova.entity_id)} == {"EVALUATED_ON"}
    assert len(provider.requests) == len(chunks)
    for relation in result.relations:
        assert relation.extraction_method == "llm-assisted"
        assert relation.qualifiers["review_status"] == "unreviewed"
        for span in relation.evidence:
            CorpusIndex(documents, chunks).validate_span(span, require_chunk=True)
    assert (
        process_responses(CorpusIndex(documents, chunks), result.records, llm_config, provider.spec)
        == result
    )


@pytest.mark.parametrize(
    "response",
    [
        "not json",
        '```json\n{"entities": [], "relations": []}\n```',
        '{"entities": [], "entities": [], "relations": []}',
        '{"entities": [], "relations": [], "confidence": 1}',
        '{"entities": [], "relations": NaN}',
        '{"entities": []}',
        '{"entities": [{"id": "e", "name": "Orion", "entity_type": "Model"},'
        ' {"id": "e", "name": "Nova", "entity_type": "Model"}], "relations": []}',
    ],
)
def test_malformed_response_rejects_chunk_and_preserves_raw_text(
    response, source_pair, llm_config, make_provider
):
    result = extract_with_llm(*source_pair, llm_config, make_provider(response))
    assert not result.entities and not result.relations
    assert len(result.rejected_chunks) == 2
    assert {i.code for i in result.issues} == {"invalid_json"}
    assert all(r.completion.text == response for r in result.records)


def test_empty_response_is_valid_abstention(source_pair, llm_config, make_provider):
    result = extract_with_llm(
        *source_pair, llm_config, make_provider({"entities": [], "relations": []})
    )
    assert not result.entities and not result.relations and not result.issues
    assert not result.rejected_chunks


def test_length_limited_response_is_rejected_even_if_json_is_complete(
    source_pair, llm_config, make_provider
):
    response = Completion(
        text='{"entities": [], "relations": []}',
        input_tokens=1,
        output_tokens=1,
        finish_reason="length",
        elapsed_ms=0.0,
    )
    result = extract_with_llm(*source_pair, llm_config, make_provider(response))
    assert len(result.rejected_chunks) == 2
    assert {i.code for i in result.issues} == {"output_limit"}


@pytest.mark.parametrize(
    "mutation,code",
    [
        ("missing_name", "invalid_entity"),
        ("wrong_type", "invalid_entity"),
        ("unknown_endpoint", "invalid_relation"),
        ("unknown_predicate", "invalid_relation"),
        ("wrong_endpoint_types", "invalid_relation"),
        ("self_relation", "invalid_relation"),
        ("invented_quote", "invalid_quote"),
        ("missing_endpoint_quote", "invalid_quote"),
    ],
)
def test_invalid_proposals_are_excluded(
    mutation, code, proposals, source_pair, llm_config, make_provider
):
    payload = proposals["Orion uses Nova."]
    if mutation == "missing_name":
        payload["entities"][0]["name"] = "Invented"
    elif mutation == "wrong_type":
        payload["entities"][0]["entity_type"] = "Unknown"
    elif mutation == "unknown_endpoint":
        payload["relations"][0]["subject"] = "missing"
    elif mutation == "unknown_predicate":
        payload["relations"][0]["predicate"] = "INVENTED"
    elif mutation == "wrong_endpoint_types":
        payload["relations"][0]["predicate"] = "EVALUATED_ON"
    elif mutation == "self_relation":
        payload["relations"][0]["object"] = "e1"
    elif mutation == "invented_quote":
        payload["relations"][0]["quote"] = "Orion is based on Nova."
    else:
        payload["relations"][0]["quote"] = "Orion"
    result = extract_with_llm(
        (source_pair[0][0],), (source_pair[1][0],), llm_config, make_provider(payload)
    )
    assert not result.relations
    assert code in {i.code for i in result.issues}


def test_unicode_offsets_are_computed_not_guessed(llm_config, make_provider, proposals):
    prefix = "π😀 Header\n"
    text = prefix + "Orion uses Nova."
    document = Document(
        document_id="unicode",
        title="Unicode",
        text=text,
        content_sha256=text_sha256(text),
        source_uri="fixture:unicode",
    )
    chunk = Chunk(
        document_id=document.document_id,
        chunk_id="chunk",
        ordinal=0,
        start=len(prefix),
        end=len(text),
        text=text[len(prefix) :],
    )
    result = extract_with_llm(
        (document,), (chunk,), llm_config, make_provider(proposals[chunk.text])
    )
    evidence = result.relations[0].evidence[0]
    assert evidence.start == len(prefix)
    assert evidence.end == len(text)


def test_repeated_quote_in_one_chunk_is_ambiguous(
    llm_config, make_provider, source_pair, proposals
):
    document = source_pair[0][0]
    text = document.text + " " + document.text
    document = document.model_copy(update={"text": text, "content_sha256": text_sha256(text)})
    chunk = Chunk(
        document_id=document.document_id,
        chunk_id="repeat",
        ordinal=0,
        start=0,
        end=len(text),
        text=text,
    )
    result = extract_with_llm(
        (document,), (chunk,), llm_config, make_provider(proposals["Orion uses Nova."])
    )
    assert not result.relations
    assert result.issues[0].code == "invalid_quote"


def test_overlap_and_duplicate_proposals_do_not_multiply_assertions(
    llm_config, make_provider, source_pair, proposals
):
    document = source_pair[0][0]
    first = source_pair[1][0]
    second = first.model_copy(update={"chunk_id": "second", "ordinal": 1})
    payload = proposals[document.text]
    payload["relations"].append(copy.deepcopy(payload["relations"][0]))
    result = extract_with_llm((document,), (first, second), llm_config, make_provider(payload))
    assert len(result.relations) == 1
    assert len(result.relations[0].evidence) == 2


def test_normalized_name_and_type_merge_but_distinct_types_remain_separate(
    llm_config, make_provider, source_pair, proposals
):
    payload = proposals["Orion uses Nova."]
    payload["entities"].append({"id": "e3", "name": "Orion", "entity_type": "Encoder"})
    result = extract_with_llm(
        (source_pair[0][0],), (source_pair[1][0],), llm_config, make_provider(payload)
    )
    graph = build_graph(
        result.entities, result.relations, (source_pair[0][0],), (source_pair[1][0],)
    )
    assert len(graph.lookup("Orion")) == 2
    assert len(graph.lookup("Orion", "Model")) == 1


def test_exact_quote_does_not_prove_direction_or_semantic_truth(
    llm_config, make_provider, source_pair, proposals
):
    # Intentional limitation: reversed semantics pass mechanical source checks.
    # The output remains explicitly unreviewed; no confidence/entailment claim is made.
    payload = proposals["Orion uses Nova."]
    payload["relations"][0].update(subject="e2", object="e1")
    result = extract_with_llm(
        (source_pair[0][0],), (source_pair[1][0],), llm_config, make_provider(payload)
    )
    assert result.relations[0].qualifiers["review_status"] == "unreviewed"


def test_word_fragments_do_not_become_entity_mentions(source_pair, llm_config, make_provider):
    payload = {"entities": [{"id": "x", "name": "Ori", "entity_type": "Model"}], "relations": []}
    result = extract_with_llm(*source_pair, llm_config, make_provider(payload))
    assert not result.entities
    assert {i.code for i in result.issues} == {"invalid_entity"}


def test_provider_mismatch_fails_before_inference(source_pair, llm_config, make_provider):
    provider = make_provider()
    provider.spec = provider.spec.model_copy(update={"revision": "f" * 40})
    with pytest.raises(LLMError, match="model differs"):
        extract_with_llm(*source_pair, llm_config, provider)
    assert not provider.requests


def test_provider_failure_is_not_silently_replaced_with_rules(
    source_pair, llm_config, make_provider
):
    with pytest.raises(LLMError, match="unavailable"):
        extract_with_llm(*source_pair, llm_config, make_provider(LLMError("unavailable")))


def test_request_contains_sources_and_schema_without_benchmark_hints(
    source_pair, llm_config, make_provider
):
    provider = make_provider()
    extract_with_llm(*source_pair, llm_config, provider)
    task = json.loads(provider.requests[0].messages[-1].content)
    assert set(task) == {
        "allowed_entity_types",
        "allowed_relations",
        "response_format",
        "source_passage",
    }
    assert task["source_passage"] in {document.text for document in source_pair[0]}
    assert "untrusted data" in provider.requests[0].messages[0].content


def test_replay_rejects_missing_duplicate_or_altered_requests(
    source_pair, llm_config, make_provider
):
    provider = make_provider()
    result = extract_with_llm(*source_pair, llm_config, provider)
    corpus = CorpusIndex(*source_pair)
    with pytest.raises(LLMError, match="every source chunk"):
        process_responses(corpus, result.records[:1], llm_config, provider.spec)
    with pytest.raises(LLMError, match="duplicate"):
        process_responses(corpus, result.records * 2, llm_config, provider.spec)
    changed = result.records[0].model_copy(update={"request_sha256": "0" * 64})
    with pytest.raises(LLMError, match="fingerprint"):
        process_responses(corpus, (changed, result.records[1]), llm_config, provider.spec)
