import json
import re
import runpy
from pathlib import Path

import pytest

from graphrag_bench.corpus import CorpusIndex
from graphrag_bench.extraction.llm import prompt
from graphrag_bench.extraction.llm.config import LLMError
from graphrag_bench.extraction.llm.contracts import ResponseRecord
from graphrag_bench.extraction.llm.extractor import extract_with_llm, process_responses
from graphrag_bench.models import Chunk, text_sha256
from graphrag_bench.serialization import json_bytes


@pytest.fixture
def cued(llm_config):
    return llm_config.model_copy(update={"passage_selection": "relation-cues-v1"})


@pytest.fixture
def single_source(source_pair):
    def create(text):
        document = source_pair[0][0].model_copy(
            update={"text": text, "content_sha256": text_sha256(text)}
        )
        chunk = Chunk(
            document_id=document.document_id,
            chunk_id="selected-source",
            ordinal=0,
            start=0,
            end=len(text),
            text=text,
        )
        return (document,), (chunk,)

    return create


@pytest.mark.parametrize(
    "field,value",
    [
        ("text", '{"entities":[],"relations":[],"extra":true}'),
        ("input_tokens", 1),
        ("output_tokens", 1),
        ("elapsed_ms", 1.0),
        ("finish_reason", "length"),
    ],
)
def test_abstention_receipt_cannot_be_altered(field, value, cued, single_source, make_provider):
    source = single_source("No cue here.")
    provider = make_provider(config=cued)
    result = extract_with_llm(*source, cued, provider)
    record = result.records[0]
    altered = record.model_copy(
        update={"completion": record.completion.model_copy(update={field: value})}
    )
    with pytest.raises(LLMError, match="exact empty, zero-cost"):
        process_responses(CorpusIndex(*source), (altered,), cued, provider.spec)


@pytest.mark.parametrize("text", ["No cue here.", "Orion uses Nova."])
def test_replay_requires_origin_matching_selection(text, cued, single_source, make_provider):
    source = single_source(text)
    provider = make_provider(config=cued)
    result = extract_with_llm(*source, cued, provider)
    record = result.records[0]
    wrong = (
        "provider" if record.origin == "deterministic-abstention" else "deterministic-abstention"
    )
    with pytest.raises(LLMError, match="origin"):
        process_responses(
            CorpusIndex(*source),
            (record.model_copy(update={"origin": wrong}),),
            cued,
            provider.spec,
        )


def test_mentions_are_limited_to_selected_spans(cued, single_source, make_provider, proposals):
    text = "Orion and Nova appear with Comet. Orion uses Nova. Orion returns."
    source = single_source(text)
    payload = proposals["Orion uses Nova."]
    payload["entities"].append({"id": "e3", "name": "Comet", "entity_type": "Model"})
    result = extract_with_llm(*source, cued, make_provider(payload, config=cued))
    assert len(result.relations) == 1
    assert {e.name for e in result.entities} == {"Orion", "Nova"}
    for entity in result.entities:
        assert all(span.start >= text.index("Orion uses") for span in entity.mentions)
    assert [issue.code for issue in result.issues] == ["invalid_entity"]


@pytest.mark.parametrize(
    "quote",
    [
        "Orion greets Nova.",
        "Orion uses Nova. Orion greets Nova. Nova uses Orion.",
        "Orion introduces Nova.",
    ],
)
def test_quote_cannot_escape_one_selected_span(
    quote, cued, single_source, make_provider, proposals
):
    text = "Orion uses Nova. Orion greets Nova. Nova uses Orion. Orion introduces Nova."
    source = single_source(text)
    payload = proposals["Orion uses Nova."]
    payload["relations"][0]["quote"] = quote
    result = extract_with_llm(*source, cued, make_provider(payload, config=cued))
    assert not result.relations
    assert [issue.code for issue in result.issues] == ["invalid_quote"]


def test_selector_changes_invalidate_only_cued_recipe(cued, llm_config, make_provider, monkeypatch):
    spec = make_provider().spec
    original = prompt.extraction_fingerprint(cued, spec)
    legacy = prompt.extraction_fingerprint(llm_config, spec)
    monkeypatch.setattr(prompt, "_RELATION_CUE", re.compile(r"\buses\b"))
    assert prompt.extraction_fingerprint(cued, spec) != original
    assert prompt.extraction_fingerprint(llm_config, spec) == legacy


def test_provider_record_preserves_legacy_json_shape(source_pair, llm_config, make_provider):
    result = extract_with_llm(*source_pair, llm_config, make_provider())
    record = result.records[0]
    raw = json_bytes(record)
    assert "origin" not in json.loads(raw)
    restored = ResponseRecord.model_validate_json(raw)
    assert restored.origin == "provider"
    assert json_bytes(restored) == raw
    assert "origin" not in result.model_dump(mode="json")["records"][0]


def test_model_cache_rejects_deterministic_origin(source_pair, cued, make_provider, tmp_path):
    provider = make_provider(config=cued)
    result = extract_with_llm(*source_pair, cued, provider, response_cache=tmp_path)
    path = tmp_path / f"{result.records[0].request_sha256}.json"
    entry = json.loads(path.read_text())
    entry["record"]["origin"] = "deterministic-abstention"
    path.write_text(json.dumps(entry))
    with pytest.raises(LLMError, match="invalid response cache"):
        extract_with_llm(*source_pair, cued, provider, response_cache=tmp_path)


@pytest.mark.parametrize("provider_calls,expected_hours", [(0, None), (1, 1.0), (2, 1.0)])
def test_preflight_estimate_does_not_treat_skips_as_model_calls(provider_calls, expected_hours):
    summarize = runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "scripts/preflight_paper_extraction.py")
    )["summarize_sample"]
    rows = [
        {
            "origin": "provider" if i < provider_calls else "deterministic-abstention",
            "completion_seconds": 36 if i < provider_calls else 0,
            "finish_reason": "eos",
            "schema_valid": True,
            "accepted_entities": 0,
            "accepted_assertions": 0,
            "rejected": False,
        }
        for i in range(8)
    ]
    summary = summarize(rows, 100)
    assert summary["estimated_full_hours"] == expected_hours
    assert summary["provider_schema_valid_count"] == provider_calls
    assert summary["deterministic_abstention_count"] == 8 - provider_calls
    if not provider_calls:
        assert summarize(rows, 0)["estimated_full_hours"] == 0
