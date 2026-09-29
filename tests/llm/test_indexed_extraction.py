import json
import runpy
from pathlib import Path

import pytest

from graphrag_bench.corpus import CorpusIndex
from graphrag_bench.extraction.llm.config import load_llm_config
from graphrag_bench.extraction.llm.extractor import extract_with_llm, process_responses
from graphrag_bench.extraction.llm.pipeline import build_llm_graph_to_directory, replay_llm_graph
from graphrag_bench.extraction.llm.prompt import (
    extraction_fingerprint,
    make_request,
    numbered_spans,
)
from graphrag_bench.ingestion.reader import load_ingestion
from graphrag_bench.models import Chunk, text_sha256
from graphrag_bench.serialization import json_bytes

ROOT = Path(__file__).resolve().parents[2]
ROW = ["Orion", "Model", "USES", "Nova", "Model", 1]


@pytest.fixture
def indexed():
    return load_llm_config(ROOT / "configs/llm-extraction-indexed.toml")


@pytest.fixture
def source(source_pair):
    def make(text):
        document = source_pair[0][0].model_copy(
            update={"text": text, "content_sha256": text_sha256(text)}
        )
        chunk = Chunk(
            document_id=document.document_id,
            chunk_id="source",
            ordinal=0,
            start=0,
            end=len(text),
            text=text,
        )
        return (document,), (chunk,)

    return make


def test_numbered_prompt_preserves_exact_sentences(indexed, source):
    pair = source("  Orion (Ada et al., 2020) uses Nova.\n\nNova returns. ")
    chunk = pair[1][0]
    request = make_request(chunk, indexed)
    assert json.loads(request.messages[-1].content) == {
        "sentences": [
            {"id": 1, "text": "Orion (Ada et al., 2020) uses Nova."},
            {"id": 2, "text": "Nova returns."},
        ]
    }
    assert request.chunk == chunk
    assert request.prompt_version == "indexed-chunk-relations-v1"
    assert "triples" not in request.messages[0].content
    for message in request.messages:
        if message.role == "assistant":
            assert set(json.loads(message.content)) == {"relations"}


@pytest.mark.parametrize("named", [False, True])
def test_rows_expand_to_original_quotes_and_replay(named, indexed, source, make_provider):
    pair = source("Background. Orion uses Nova. Other discussion.")
    payload = {"relations": [ROW[:-1] + [2]]}
    if named:
        payload = {
            "relations": [
                dict(
                    zip(
                        (
                            "subject",
                            "subject_type",
                            "predicate",
                            "object",
                            "object_type",
                            "sentence_id",
                        ),
                        ROW[:-1] + [2],
                        strict=True,
                    )
                )
            ]
        }
    provider = make_provider(payload, config=indexed)
    result = extract_with_llm(*pair, indexed, provider)
    assert len(result.relations) == 1 and not result.issues
    quote = result.relations[0].evidence[0]
    assert (quote.start, quote.text) == (12, "Orion uses Nova.")
    assert pair[0][0].text[quote.start : quote.end] == quote.text
    assert json.loads(result.records[0].completion.text) == payload
    assert process_responses(CorpusIndex(*pair), result.records, indexed, provider.spec) == result


@pytest.mark.parametrize("sentence", [True, 1.0, "1", 0, -1])
def test_sentence_ids_are_positive_strict_integers(sentence, indexed, source, make_provider):
    pair = source("Orion uses Nova.")
    result = extract_with_llm(
        *pair, indexed, make_provider({"relations": [ROW[:-1] + [sentence]]}, config=indexed)
    )
    assert result.rejected_chunks == ("source",)
    assert not result.relations


@pytest.mark.parametrize(
    "updates", [{"sentence_id": True}, {"sentence_id": 1.0}, {"quote": "made up"}]
)
def test_named_rows_reject_loose_ids_and_extra_fields(updates, indexed, source, make_provider):
    row = (
        dict(
            zip(
                ("subject", "subject_type", "predicate", "object", "object_type", "sentence_id"),
                ROW,
                strict=True,
            )
        )
        | updates
    )
    result = extract_with_llm(
        *source("Orion uses Nova."), indexed, make_provider({"relations": [row]}, config=indexed)
    )
    assert result.rejected_chunks and not result.relations


@pytest.mark.parametrize(
    "payload",
    [
        {"relations": [ROW[:-1]]},
        {"relations": [ROW + ["extra"]]},
        {"relations": [ROW] * 5},
        {"relations": [], "entities": []},
        '{"relations":[],"relations":[]}',
        '{"relations":NaN}',
    ],
)
def test_invalid_rows_are_not_repaired(payload, indexed, source, make_provider):
    result = extract_with_llm(
        *source("Orion uses Nova."), indexed, make_provider(payload, config=indexed)
    )
    assert result.rejected_chunks and not result.relations


@pytest.mark.parametrize(
    "text,sentence",
    [
        ("Orion uses Nova.", 2),
        ("Orion returns. Nova returns.", 1),
        ("Orion uses Nova. Unrelated sentence.", 2),
        ("Orion uses Nova. Orion uses Nova.", 1),
    ],
)
def test_quote_checks_survive_indexed_conversion(text, sentence, indexed, source, make_provider):
    result = extract_with_llm(
        *source(text),
        indexed,
        make_provider({"relations": [ROW[:-1] + [sentence]]}, config=indexed),
    )
    assert not result.relations and not result.rejected_chunks
    assert "invalid_quote" in {i.code for i in result.issues}


def test_bad_sentence_does_not_remove_valid_row(indexed, source, make_provider):
    result = extract_with_llm(
        *source("Orion uses Nova."),
        indexed,
        make_provider({"relations": [ROW[:-1] + [9], ROW]}, config=indexed),
    )
    assert len(result.relations) == 1
    assert [i.code for i in result.issues] == ["invalid_quote"]


@pytest.mark.parametrize(
    "field,value,issue",
    [
        (1, "InventedType", "invalid_entity"),
        (2, "INVENTED", "invalid_relation"),
        (4, "Dataset", "invalid_relation"),
    ],
)
def test_ontology_checks_survive_conversion(field, value, issue, indexed, source, make_provider):
    row = ROW.copy()
    row[field] = value
    result = extract_with_llm(
        *source("Orion uses Nova."), indexed, make_provider({"relations": [row]}, config=indexed)
    )
    assert not result.relations
    assert issue in {i.code for i in result.issues}


def test_indexed_empty_skip_is_explicit_and_replayable(indexed, source, make_provider):
    config = indexed.model_copy(update={"passage_selection": "relation-cues-v1"})
    pair = source("No cue here.")
    provider = make_provider(config=config)
    result = extract_with_llm(*pair, config, provider)
    assert not provider.requests and not result.issues
    raw = json_bytes(result.records[0])
    assert json.loads(raw)["origin"] == "deterministic-abstention"
    assert process_responses(CorpusIndex(*pair), result.records, config, provider.spec) == result


def test_cue_selection_numbering_has_original_offsets(indexed, source):
    config = indexed.model_copy(update={"passage_selection": "relation-cues-v1"})
    text = "Background. Orion uses Nova. Other discussion. Nova uses Orion."
    content = json.loads(make_request(source(text)[1][0], config).messages[-1].content)
    spans = numbered_spans(text, config)
    assert [s["text"] for s in content["sentences"]] == [text[a:b] for a, b in spans]
    assert content["sentences"][0]["text"] == "Orion uses Nova."
    assert spans[0][0] == 12


def test_indexed_graph_roundtrip_and_legacy_hash(
    indexed, llm_config, llm_ingestion, make_provider, tmp_path
):
    def response(request):
        if "Orion" in request.chunk.text:
            return {"relations": [ROW]}
        return {"relations": [["Nova", "Model", "EVALUATED_ON", "Harbor", "Dataset", 1]]}

    provider = make_provider(response, config=indexed)
    run, replay = tmp_path / "run", tmp_path / "replay"
    original = build_llm_graph_to_directory(llm_ingestion, run, indexed, provider)
    restored = replay_llm_graph(run, llm_ingestion, replay)
    assert original.assertion_count == 2
    assert original.artifact_hashes == restored.artifact_hashes
    assert extraction_fingerprint(llm_config, make_provider().spec) == (
        "2592e02d1456b7ea56974e52c886e8f5f353a841ff39f26b0dd46d66468db3fd"
    )


def test_draft_checks_do_not_reach_provider_and_reject_wrong_source(
    llm_ingestion, indexed, make_provider, tmp_path
):
    diagnostic = runpy.run_path(str(ROOT / "scripts/check_extraction_sample.py"))
    run_checks = diagnostic["run_checks"]
    batch, hashes = load_ingestion(llm_ingestion)
    chunk = next(c for c in batch.chunks if "Orion" in c.text)
    checks = {
        "source_hashes": hashes,
        "review_status": "draft",
        "selection": "test",
        "cases": [
            {
                "id": "SECRET-LABEL",
                "chunk_id": chunk.chunk_id,
                "expected": [{"subjects": ["Orion"], "predicate": "USES", "objects": ["Nova"]}],
            }
        ],
    }
    path = tmp_path / "checks.json"
    path.write_text(json.dumps(checks))
    provider = make_provider({"relations": [ROW]}, config=indexed)
    report = run_checks(llm_ingestion, path, indexed, provider, tmp_path / "cache")
    assert report["summary"]["matched_facts"] == 1
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))
    assert diagnostic["verify_checks"](llm_ingestion, path, report_path) == report["summary"]
    report["summary"]["matched_facts"] = 9
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="differs from replayed"):
        diagnostic["verify_checks"](llm_ingestion, path, report_path)
    assert all("SECRET-LABEL" not in m.content for r in provider.requests for m in r.messages)
    checks["source_hashes"] = {}
    path.write_text(json.dumps(checks))
    provider.requests.clear()
    with pytest.raises(ValueError, match="different source"):
        run_checks(llm_ingestion, path, indexed, provider, tmp_path / "cache")
    assert not provider.requests
