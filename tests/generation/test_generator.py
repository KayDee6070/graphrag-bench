import json

import pytest

from graphrag_bench.corpus import CorpusIndex
from graphrag_bench.extraction.llm.contracts import Completion
from graphrag_bench.generation.config import GenerationError
from graphrag_bench.generation.contracts import RetrievalReceipt
from graphrag_bench.generation.generator import generate_answer, render_answer, replay_answer
from graphrag_bench.models import Chunk, Document, RetrievalHit, RetrievalResult, text_sha256


def test_cites_both_sources_with_exact_coordinates_and_replays(
    retrieval_receipt, answer_corpus, generation_config, answer_provider
):
    provider = answer_provider()
    request, receipt, answer = generate_answer(
        retrieval_receipt, answer_corpus, generation_config, provider
    )
    assert answer.status == "answered" and len(answer.claims[0].citations) == 2
    assert answer.review_status == "unreviewed"
    for citation in answer.claims[0].citations:
        answer_corpus.validate_span(citation.span, require_chunk=True)
        assert citation.source_uri == answer_corpus.documents[citation.span.document_id].source_uri
    assert (
        replay_answer(
            retrieval_receipt, request, receipt, answer_corpus, generation_config, provider.spec
        )
        == answer
    )
    task = json.loads(request.messages[-1].content)
    assert set(task) == {"question", "sources"}
    assert len(task["sources"]) == 2
    assert request.context.text in request.messages[-1].content
    assert request.context.token_count == len(request.context.text)
    assert "[S1] [S2]" in render_answer(answer)


@pytest.mark.parametrize(
    "raw",
    [
        "not JSON",
        "```json\n{}\n```",
        '{"status":"answered","claims":[]}',
        '{"status":"insufficient_evidence","claims":[],"extra":1}',
        '{"status":"insufficient_evidence","status":"answered","claims":[]}',
        '{"status":"answered","claims":NaN}',
        '{"status":"answered","claims":[{"text":"Harbor","citations":[]}]}',
        '{"status":"insufficient_evidence","claims":[{"text":"Harbor","citations":[{"source_id":"S1","quote":"x"}]}]}',
    ],
)
def test_invalid_schema_never_yields_an_answer(
    raw, retrieval_receipt, answer_corpus, generation_config, answer_provider
):
    _, receipt, answer = generate_answer(
        retrieval_receipt, answer_corpus, generation_config, answer_provider(raw)
    )
    assert answer.status == "invalid_output" and not answer.claims
    assert answer.issues[0].code == "invalid_response"
    assert receipt.completion.text == raw


@pytest.mark.parametrize("kind", ["unknown_source", "invented_quote", "wrong_source", "duplicate"])
def test_bad_citation_rejects_entire_answer(
    kind, retrieval_receipt, answer_corpus, generation_config, answer_provider
):
    def response(request):
        reference = {"source_id": "S1", "quote": request.context.pieces[0].text}
        citations = [reference]
        if kind == "unknown_source":
            reference["source_id"] = "S99"
        elif kind == "invented_quote":
            reference["quote"] = "A fabricated supporting quote."
        elif kind == "wrong_source":
            reference["quote"] = request.context.pieces[1].text
        else:
            citations.append(reference.copy())
        return {
            "status": "answered",
            "claims": [
                {
                    "text": "This first claim has a real reference.",
                    "citations": [{"source_id": "S2", "quote": request.context.pieces[1].text}],
                },
                {"text": "Harbor.", "citations": citations},
            ],
        }

    _, _, answer = generate_answer(
        retrieval_receipt, answer_corpus, generation_config, answer_provider(response)
    )
    assert answer.status == "invalid_output" and not answer.claims
    assert answer.issues[0].code == "invalid_citation"


def test_model_abstention_is_distinct_from_invalid_output(
    retrieval_receipt, answer_corpus, generation_config, answer_provider
):
    _, _, answer = generate_answer(
        retrieval_receipt,
        answer_corpus,
        generation_config,
        answer_provider({"status": "insufficient_evidence", "claims": []}),
    )
    assert answer.status == "insufficient_evidence" and not answer.issues


@pytest.mark.parametrize("mode", ["no_hits", "no_budget"])
def test_no_selected_evidence_skips_generation(
    mode, retrieval_receipt, answer_corpus, generation_config, answer_provider
):
    if mode == "no_hits":
        retrieval_receipt = retrieval_receipt.model_copy(
            update={"result": retrieval_receipt.result.model_copy(update={"hits": ()})}
        )
    else:
        generation_config = generation_config.model_copy(update={"context_tokens": 1})
    provider = answer_provider(AssertionError("model must not be called"), config=generation_config)
    request, receipt, answer = generate_answer(
        retrieval_receipt, answer_corpus, generation_config, provider
    )
    assert not provider.requests and receipt.completion is None
    assert answer.status == "insufficient_evidence"
    assert request.context.token_count == 0


def test_output_limit_is_rejected_even_for_valid_json(
    retrieval_receipt, answer_corpus, generation_config, answer_provider
):
    completion = Completion(
        text='{"status":"insufficient_evidence","claims":[]}',
        input_tokens=1,
        output_tokens=1,
        finish_reason="length",
        elapsed_ms=0,
    )
    _, _, answer = generate_answer(
        retrieval_receipt, answer_corpus, generation_config, answer_provider(completion)
    )
    assert answer.status == "invalid_output" and answer.issues[0].code == "output_limit"


def test_real_quote_can_still_support_wrong_answer_or_incomplete_chain(
    retrieval_receipt, answer_corpus, generation_config, answer_provider
):
    def wrong(request):
        return {
            "status": "answered",
            "claims": [
                {
                    "text": "An unrelated and wrong conclusion.",
                    "citations": [{"source_id": "S1", "quote": request.context.pieces[0].text}],
                }
            ],
        }

    _, _, answer = generate_answer(
        retrieval_receipt, answer_corpus, generation_config, answer_provider(wrong)
    )
    assert answer.status == "answered" and answer.review_status == "unreviewed"
    assert (
        len(answer.claims[0].citations) == 1
    )  # Mechanical checks cannot establish chain completeness.


def test_model_markup_is_escaped_in_human_readable_answer(
    retrieval_receipt, answer_corpus, generation_config, answer_provider
):
    def response(request):
        return {
            "status": "answered",
            "claims": [
                {
                    "text": "[S99](https://invented.invalid) <script>bad</script>",
                    "citations": [{"source_id": "S1", "quote": request.context.pieces[0].text}],
                }
            ],
        }

    _, _, answer = generate_answer(
        retrieval_receipt, answer_corpus, generation_config, answer_provider(response)
    )
    rendered = render_answer(answer)
    assert "[S99](" not in rendered and "<script>" not in rendered
    assert "[S1]" in rendered


def test_provider_failure_and_mismatch_do_not_silently_abstain(
    retrieval_receipt, answer_corpus, generation_config, answer_provider
):
    provider = answer_provider(GenerationError("unavailable"))
    with pytest.raises(GenerationError, match="unavailable"):
        generate_answer(retrieval_receipt, answer_corpus, generation_config, provider)
    provider = answer_provider()
    provider.spec = provider.spec.model_copy(update={"revision": "f" * 40})
    with pytest.raises(GenerationError, match="model differs"):
        generate_answer(retrieval_receipt, answer_corpus, generation_config, provider)
    assert not provider.requests


@pytest.mark.parametrize(
    "change,match",
    [
        ("fingerprint", "fingerprint"),
        ("prompt", "saved answer request"),
        ("counts", "token measurement"),
        ("extra_count", "unused token"),
        ("missing_completion", "requires an inference"),
        ("token_limit", "exceed"),
    ],
)
def test_replay_rejects_inconsistent_receipts(
    change, match, retrieval_receipt, answer_corpus, generation_config, answer_provider
):
    provider = answer_provider()
    request, receipt, _ = generate_answer(
        retrieval_receipt, answer_corpus, generation_config, provider
    )
    if change == "fingerprint":
        receipt = receipt.model_copy(update={"request_sha256": "0" * 64})
    elif change == "prompt":
        request = request.model_copy(update={"messages": request.messages[:-1]})
    elif change == "counts":
        request = request.model_copy(update={"token_measurements": request.token_measurements[1:]})
    elif change == "extra_count":
        request = request.model_copy(
            update={
                "token_measurements": request.token_measurements + request.token_measurements[:1]
            }
        )
    elif change == "missing_completion":
        receipt = receipt.model_copy(update={"completion": None})
    else:
        receipt = receipt.model_copy(
            update={"completion": receipt.completion.model_copy(update={"input_tokens": 100000})}
        )
    with pytest.raises(GenerationError, match=match):
        replay_answer(
            retrieval_receipt, request, receipt, answer_corpus, generation_config, provider.spec
        )


def test_unicode_and_overlap_citations_cannot_reach_omitted_text(
    generation_config, answer_provider
):
    text = "π😀 First. Second."
    document = Document(
        document_id="d",
        title="Unicode",
        text=text,
        content_sha256=text_sha256(text),
        source_uri="test:unicode",
    )
    chunks = (
        Chunk(document_id="d", chunk_id="c0", ordinal=0, start=0, end=9, text=text[:9]),
        Chunk(document_id="d", chunk_id="c1", ordinal=1, start=0, end=len(text), text=text),
    )
    corpus = CorpusIndex((document,), chunks)
    retrieval = RetrievalReceipt(
        config=generation_config.retrieval,
        result=RetrievalResult(
            query="Question?",
            strategy="graph",
            hits=tuple(RetrievalHit(chunk_id=c.chunk_id, score=1.0) for c in chunks),
            elapsed_ms=0.0,
        ),
    )
    provider = answer_provider()
    request, _, answer = generate_answer(retrieval, corpus, generation_config, provider)
    assert [(p.start, p.end) for p in request.context.pieces] == [(0, 9), (9, len(text))]
    assert answer.claims[0].citations[1].span.start == 9

    def omitted(_):
        return {
            "status": "answered",
            "claims": [{"text": "First.", "citations": [{"source_id": "S2", "quote": "First."}]}],
        }

    _, _, rejected = generate_answer(retrieval, corpus, generation_config, answer_provider(omitted))
    assert rejected.status == "invalid_output"


def test_repeated_quote_rejected_and_exact_budget_boundary(generation_config, answer_provider):
    text = "Repeat. Repeat."
    doc = Document(
        document_id="d",
        title="Repeated",
        text=text,
        source_uri="test:repeat",
        content_sha256=text_sha256(text),
    )
    chunk = Chunk(document_id="d", chunk_id="c", ordinal=0, start=0, end=len(text), text=text)
    corpus = CorpusIndex((doc,), (chunk,))
    retrieval = RetrievalReceipt(
        config=generation_config.retrieval,
        result=RetrievalResult(
            query="Question?",
            strategy="graph",
            hits=(RetrievalHit(chunk_id="c", score=1.0),),
            elapsed_ms=0.0,
        ),
    )
    bad = {
        "status": "answered",
        "claims": [{"text": "Repeat.", "citations": [{"source_id": "S1", "quote": "Repeat."}]}],
    }
    request, _, answer = generate_answer(retrieval, corpus, generation_config, answer_provider(bad))
    assert answer.status == "invalid_output"
    config = generation_config.model_copy(update={"context_tokens": request.context.token_count})
    assert (
        generate_answer(retrieval, corpus, config, answer_provider(config=config))[2].status
        == "answered"
    )
    config = config.model_copy(update={"context_tokens": config.context_tokens - 1})
    assert (
        generate_answer(retrieval, corpus, config, answer_provider(config=config))[2].status
        == "insufficient_evidence"
    )
