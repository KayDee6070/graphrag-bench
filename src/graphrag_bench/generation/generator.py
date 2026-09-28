"""Generate claims and resolve citations only within evidence actually shown to the model."""

import json
import re

from graphrag_bench.benchmark.context import assemble_context
from graphrag_bench.corpus import CorpusIndex
from graphrag_bench.extraction.llm.contracts import Completion, ModelSpec
from graphrag_bench.generation.config import GenerationConfig, GenerationError
from graphrag_bench.generation.context import RecordingCounter, ReplayCounter, render_sources
from graphrag_bench.generation.contracts import (
    Answer,
    AnswerIssue,
    AnswerProposal,
    AnswerProvider,
    AnswerReceipt,
    AnswerRequest,
    Citation,
    CitedClaim,
    RetrievalReceipt,
)
from graphrag_bench.generation.prompt import answer_fingerprint, make_answer_request
from graphrag_bench.models import EvidenceSpan


def validate_provider(config: GenerationConfig, spec: ModelSpec) -> None:
    if (spec.model_id, spec.revision) != (config.model.model_id, config.model.revision):
        raise GenerationError("answer provider model differs from configuration")
    if any(spec.settings.get(k) != v for k, v in config.model.model_dump(mode="json").items()):
        raise GenerationError("answer provider settings differ from configuration")


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid(code: str, message: str) -> Answer:
    return Answer(status="invalid_output", issues=(AnswerIssue(code=code, message=message),))


def validate_answer(
    request: AnswerRequest, completion: Completion | None, corpus: CorpusIndex
) -> Answer:
    if not request.context.pieces:
        if completion is not None:
            raise GenerationError("empty context must not have an inference completion")
        return Answer(status="insufficient_evidence")
    if completion is None:
        raise GenerationError("nonempty context requires an inference completion")
    if completion.finish_reason == "length":
        return _invalid("output_limit", "Generation hit its output limit; answer rejected.")
    try:

        def invalid_constant(value: str) -> None:
            raise ValueError(f"invalid JSON constant: {value}")

        proposal = AnswerProposal.model_validate(
            json.loads(
                completion.text, object_pairs_hook=_unique_object, parse_constant=invalid_constant
            )
        )
    except (ValueError, RecursionError) as error:
        return _invalid("invalid_response", f"Response rejected: {str(error)[:240]}")
    claims = []
    sources = {f"S{i}": p for i, p in enumerate(request.context.pieces, start=1)}
    for claim in proposal.claims:
        citations = []
        seen = set()
        for reference in claim.citations:
            piece = sources.get(reference.source_id)
            if piece is None:
                return _invalid("invalid_citation", f"Unknown source ID: {reference.source_id}")
            start = piece.text.find(reference.quote)
            if start < 0 or piece.text.find(reference.quote, start + 1) >= 0:
                return _invalid(
                    "invalid_citation", "Quote must occur exactly once in its shown source."
                )
            key = (reference.source_id, reference.quote)
            if key in seen:
                return _invalid("invalid_citation", "Duplicate citation within one claim.")
            seen.add(key)
            span = EvidenceSpan(
                document_id=piece.document_id,
                chunk_id=piece.chunk_id,
                start=piece.start + start,
                end=piece.start + start + len(reference.quote),
                text=reference.quote,
            )
            corpus.validate_span(span, require_chunk=True)
            document = corpus.documents[piece.document_id]
            citations.append(
                Citation(
                    source_id=reference.source_id,
                    span=span,
                    title=document.title,
                    source_uri=document.source_uri,
                )
            )
        claims.append(CitedClaim(text=claim.text, citations=tuple(citations)))
    return Answer(status=proposal.status, claims=tuple(claims))


def validate_retrieval(
    receipt: RetrievalReceipt, corpus: CorpusIndex, config: GenerationConfig
) -> None:
    if receipt.config != config.retrieval or receipt.result.strategy != config.retrieval.strategy:
        raise GenerationError("retrieval receipt differs from answer configuration")
    if len(receipt.result.hits) > config.retrieval.top_k:
        raise GenerationError("retrieval receipt exceeds top_k")
    if any(hit.chunk_id not in corpus.chunks for hit in receipt.result.hits):
        raise GenerationError("retrieval receipt contains an unknown source chunk")


def generate_answer(
    retrieval: RetrievalReceipt,
    corpus: CorpusIndex,
    config: GenerationConfig,
    provider: AnswerProvider,
) -> tuple[AnswerRequest, AnswerReceipt, Answer]:
    spec = provider.spec
    validate_provider(config, spec)
    validate_retrieval(retrieval, corpus, config)
    counter = RecordingCounter(provider)
    context = assemble_context(
        retrieval.result, corpus, counter, max_tokens=config.context_tokens, renderer=render_sources
    )
    request = make_answer_request(retrieval.result.query, context, tuple(counter.measurements))
    completion = provider.complete(request) if context.pieces else None
    if completion is not None and not isinstance(completion, Completion):
        raise GenerationError("provider must return a validated Completion")
    if provider.spec != spec:
        raise GenerationError("answer provider specification changed during generation")
    receipt = AnswerReceipt(
        request_sha256=answer_fingerprint(config, spec, request), completion=completion
    )
    return request, receipt, replay_answer(retrieval, request, receipt, corpus, config, spec)


def replay_answer(
    retrieval: RetrievalReceipt,
    request: AnswerRequest,
    receipt: AnswerReceipt,
    corpus: CorpusIndex,
    config: GenerationConfig,
    spec: ModelSpec,
) -> Answer:
    validate_provider(config, spec)
    validate_retrieval(retrieval, corpus, config)
    counter = ReplayCounter(request.token_measurements)
    context = assemble_context(
        retrieval.result, corpus, counter, max_tokens=config.context_tokens, renderer=render_sources
    )
    counter.finish()
    if request != make_answer_request(retrieval.result.query, context, request.token_measurements):
        raise GenerationError("saved answer request differs from source, selection, or prompt")
    if receipt.request_sha256 != answer_fingerprint(config, spec, request):
        raise GenerationError("answer request fingerprint mismatch")
    if receipt.completion is not None and (
        receipt.completion.input_tokens > config.model.max_input_tokens
        or receipt.completion.output_tokens > config.model.max_new_tokens
    ):
        raise GenerationError("completion token counts exceed configured limits")
    return validate_answer(request, receipt.completion, corpus)


def _escape(text: str) -> str:
    # Treat generated/source strings as text, preventing fabricated Markdown links/HTML.
    return re.sub(r"([\\`*_{}\[\]<>()#!|>])", r"\\\1", text)


def render_answer(answer: Answer) -> str:
    if answer.status == "insufficient_evidence":
        return "Insufficient evidence in the selected sources to answer the question.\n"
    if answer.status == "invalid_output":
        return "The model response failed format or citation checks. No answer was accepted.\n"
    lines = ["Answer proposal — semantic correctness has not been reviewed.", ""]
    for claim in answer.claims:
        labels = dict.fromkeys(c.source_id for c in claim.citations)
        lines.append(_escape(claim.text) + " " + " ".join(f"[{label}]" for label in labels))
        lines.append("")
    lines.extend(("Sources and exact quotes:", ""))
    seen = set()
    for claim in answer.claims:
        for citation in claim.citations:
            span = citation.span
            key = (citation.source_id, span.start, span.end)
            if key in seen:
                continue
            seen.add(key)
            lines.extend(
                (
                    f"- **[{citation.source_id}]** {_escape(citation.title)}; "
                    f"{_escape(citation.source_uri)}",
                    f"  Document `{span.document_id}`, chunk `{span.chunk_id}`, "
                    f"characters [{span.start}, {span.end}).",
                    "",
                    "  > " + _escape(span.text).replace("\n", "\n  > "),
                    "",
                )
            )
    return "\n".join(lines)
