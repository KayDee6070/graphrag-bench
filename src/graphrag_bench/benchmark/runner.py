"""Execute repeated retrieval queries; gold labels are used only after retrieval."""

import random
from collections.abc import Callable, Mapping
from hashlib import sha256
from time import perf_counter

from pydantic import Field

from graphrag_bench.benchmark.config import BenchmarkConfig, Strategy
from graphrag_bench.benchmark.context import SelectedContext, TokenCounter, assemble_context
from graphrag_bench.benchmark.dataset import BenchmarkDataset, BenchmarkError
from graphrag_bench.benchmark.metrics import EvidenceScore, score_evidence
from graphrag_bench.corpus import CorpusIndex
from graphrag_bench.models import (
    Identifier,
    JsonValue,
    NonNegativeInt,
    PositiveInt,
    Record,
    RetrievalResult,
)
from graphrag_bench.serialization import json_bytes

BENCHMARK_VERSION = "source-evidence-benchmark-v1"


class SearchObservation(Record):
    result: RetrievalResult
    diagnostics: dict[str, JsonValue] = Field(default_factory=dict)


Search = Callable[[str, int], SearchObservation]


class CutoffEvaluation(Record):
    k: PositiveInt
    raw: EvidenceScore
    context: SelectedContext
    budgeted: EvidenceScore
    context_ms: float = Field(ge=0)


class QuestionRun(Record):
    question_id: Identifier
    strategy: Strategy
    repeat: NonNegativeInt
    observation: SearchObservation
    retrieval_ms: float = Field(ge=0)
    evaluations: tuple[CutoffEvaluation, ...]


def deterministic_record(record: QuestionRun) -> dict:
    """Rankings, paths, selected text, and metric values; omit times and diagnostics."""
    return {
        "question_id": record.question_id,
        "strategy": record.strategy,
        "repeat": record.repeat,
        "hits": [hit.model_dump(mode="json") for hit in record.observation.result.hits],
        "evaluations": [
            item.model_dump(mode="json", exclude={"context_ms"}) for item in record.evaluations
        ],
    }


def evaluation_fingerprint(records: tuple[QuestionRun, ...]) -> str:
    class Results(Record):
        records: tuple[dict[str, JsonValue], ...]

    return sha256(
        json_bytes(Results(records=tuple(deterministic_record(r) for r in records)))
    ).hexdigest()


def run_queries(
    dataset: BenchmarkDataset,
    corpus: CorpusIndex,
    searches: Mapping[str, Search],
    counter: TokenCounter,
    config: BenchmarkConfig,
) -> tuple[QuestionRun, ...]:
    if set(searches) != set(config.strategies):
        raise BenchmarkError("search strategies must exactly match benchmark configuration")
    if tuple(corpus.documents.values()) != dataset.documents:
        raise BenchmarkError("benchmark and retrievers must use the same source documents")
    k = max(config.cutoffs)
    # One explicitly excluded warm-up per method. No gold fields enter these calls.
    for strategy in config.strategies:
        searches[strategy](dataset.questions[0].question, k)
    generator = random.Random(config.seed)
    records = []
    for repeat in range(config.repeats):
        for question in dataset.questions:
            order = list(config.strategies)
            generator.shuffle(order)
            for strategy in order:
                started = perf_counter()
                observation = searches[strategy](question.question, k)
                elapsed = (perf_counter() - started) * 1000
                result = observation.result
                if (
                    result.query != question.question
                    or result.strategy != strategy
                    or len(result.hits) > k
                ):
                    raise BenchmarkError(
                        "retriever returned a different query, strategy, or excessive hits"
                    )
                if any(hit.chunk_id not in corpus.chunks for hit in result.hits):
                    raise BenchmarkError("retriever returned an unknown source chunk")
                evaluations = []
                for cutoff in config.cutoffs:
                    prefix = RetrievalResult(
                        query=result.query,
                        strategy=strategy,
                        hits=result.hits[:cutoff],
                        elapsed_ms=result.elapsed_ms,
                    )
                    started = perf_counter()
                    context = assemble_context(
                        prefix, corpus, counter, max_tokens=config.max_context_tokens
                    )
                    context_ms = (perf_counter() - started) * 1000
                    evaluations.append(
                        CutoffEvaluation(
                            k=cutoff,
                            raw=score_evidence(
                                question, (corpus.chunks[h.chunk_id] for h in prefix.hits)
                            ),
                            context=context,
                            budgeted=score_evidence(question, context.pieces),
                            context_ms=context_ms,
                        )
                    )
                records.append(
                    QuestionRun(
                        question_id=question.question_id,
                        strategy=strategy,
                        repeat=repeat,
                        observation=observation,
                        retrieval_ms=elapsed,
                        evaluations=tuple(evaluations),
                    )
                )
    return tuple(sorted(records, key=lambda r: (r.question_id, r.strategy, r.repeat)))
