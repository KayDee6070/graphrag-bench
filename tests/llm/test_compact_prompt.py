import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from graphrag_bench.corpus import CorpusIndex
from graphrag_bench.extraction.llm.config import LLMError, LLMExtractionConfig, load_llm_config
from graphrag_bench.extraction.llm.extractor import (
    extract_with_llm,
    process_responses,
    validate_sample_responses,
)
from graphrag_bench.extraction.llm.prompt import extraction_fingerprint, make_request
from graphrag_bench.serialization import json_bytes

ROOT = Path(__file__).resolve().parents[2]


def test_standard_recipe_fingerprint_remains_backward_compatible(llm_config, make_provider):
    # Captured before adding opt-in compact prompts; existing M8 graphs must replay.
    assert extraction_fingerprint(llm_config, make_provider().spec) == (
        "2592e02d1456b7ea56974e52c886e8f5f353a841ff39f26b0dd46d66468db3fd"
    )


def test_compact_prompt_is_source_bound_and_example_free(llm_config, source_pair, make_provider):
    compact = load_llm_config(ROOT / "configs/llm-extraction-compact.toml")
    standard_request = make_request(source_pair[1][0], llm_config)
    request = make_request(source_pair[1][0], compact)
    assert request.prompt_version == "compact-chunk-relations-v1"
    assert len(request.messages) == 2
    assert request.messages[-1].content.endswith(source_pair[1][0].text)
    assert "Comet" not in request.messages[-1].content
    assert sum(len(m.content) for m in request.messages) < sum(
        len(m.content) for m in standard_request.messages
    )
    assert {r.predicate for r in compact.relations} == {r.predicate for r in llm_config.relations}
    provider = make_provider(config=compact)
    result = extract_with_llm(*source_pair, compact, provider)
    assert len(result.relations) == 2
    assert (
        process_responses(CorpusIndex(*source_pair), result.records, compact, provider.spec)
        == result
    )


def test_compact_configuration_rejects_hidden_examples(llm_config):
    with pytest.raises(ValidationError, match="example-free"):
        LLMExtractionConfig.model_validate(llm_config.model_dump() | {"prompt_style": "compact-v1"})


def test_focused_prompt_keeps_ontology_out_of_final_source_message(source_pair, make_provider):
    config = load_llm_config(ROOT / "configs/llm-extraction-focused.toml")
    request = make_request(source_pair[1][0], config)
    assert request.prompt_version == "focused-chunk-relations-v1"
    assert json.loads(request.messages[-1].content) == {"source_passage": source_pair[1][0].text}
    assert "Entity types:" in request.messages[0].content
    assert "EVALUATED_ON" in request.messages[0].content
    assert len(config.examples) == 4
    provider = make_provider(config=config)
    result = extract_with_llm(*source_pair, config, provider)
    assert (
        process_responses(CorpusIndex(*source_pair), result.records, config, provider.spec)
        == result
    )


@pytest.mark.parametrize("style", ["compact", "focused"])
def test_prompt_survives_canonical_config_serialization(style, source_pair, make_provider):
    config = load_llm_config(ROOT / f"configs/llm-extraction-{style}.toml")
    restored = LLMExtractionConfig.model_validate_json(json_bytes(config))
    spec = make_provider(config=config).spec
    assert make_request(source_pair[1][0], config) == make_request(source_pair[1][0], restored)
    assert extraction_fingerprint(config, spec) == extraction_fingerprint(restored, spec)


def test_sample_diagnostics_do_not_relax_complete_graph_validation(
    source_pair, llm_config, make_provider
):
    provider = make_provider()
    result = extract_with_llm(*source_pair, llm_config, provider)
    corpus = CorpusIndex(*source_pair)
    sample = result.records[:1]
    checked = validate_sample_responses(corpus, sample, llm_config, provider.spec)
    assert len(checked.relations) == 1
    with pytest.raises(LLMError, match="every source chunk"):
        process_responses(corpus, sample, llm_config, provider.spec)
    with pytest.raises(LLMError, match="duplicate"):
        validate_sample_responses(corpus, sample * 2, llm_config, provider.spec)
    damaged = sample[0].model_copy(update={"request_sha256": "0" * 64})
    with pytest.raises(LLMError, match="fingerprint"):
        validate_sample_responses(corpus, (damaged,), llm_config, provider.spec)


@pytest.mark.parametrize("name,accepted", [("  Nova\n", True), ("No va", False), ("Comet", False)])
def test_opt_in_name_trimming_still_requires_exact_source(
    name, accepted, llm_config, make_provider, source_pair, proposals
):
    payload = proposals["Orion uses Nova."]
    payload["entities"][1]["name"] = name
    config = llm_config.model_copy(update={"strip_entity_whitespace": True})
    provider = make_provider(payload, config=config)
    result = extract_with_llm((source_pair[0][0],), (source_pair[1][0],), config, provider)
    assert bool(result.relations) is accepted
    assert json.loads(result.records[0].completion.text)["entities"][1]["name"] == name
    if accepted:
        assert {e.name for e in result.entities} == {"Orion", "Nova"}
        assert result.relations[0].evidence[0].text == "Orion uses Nova."
        assert {issue.code for issue in result.issues} == {"normalized_entity"}
        strict = extract_with_llm(
            (source_pair[0][0],), (source_pair[1][0],), llm_config, make_provider(payload)
        )
        assert not strict.relations
    assert extraction_fingerprint(config, provider.spec) != extraction_fingerprint(
        llm_config, provider.spec
    )
