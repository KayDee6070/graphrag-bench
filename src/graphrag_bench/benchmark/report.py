"""Descriptive macro averages and paired question differences, without significance claims."""

from collections import defaultdict
from statistics import mean

import numpy as np

from graphrag_bench.benchmark.dataset import BenchmarkDataset
from graphrag_bench.benchmark.runner import (
    QuestionRun,
    deterministic_record,
    evaluation_fingerprint,
)
from graphrag_bench.models import JsonValue, NonNegativeInt, PositiveInt, Record, Sha256


class BenchmarkSummary(Record):
    question_count: PositiveInt
    group_count: PositiveInt
    record_count: PositiveInt
    split: str
    evaluation_sha256: Sha256
    repeatable: bool | None
    unstable_queries: tuple[str, ...]
    aggregates: tuple[dict[str, JsonValue], ...]
    paired: tuple[dict[str, JsonValue], ...]
    unreachable_gold_questions: tuple[str, ...]
    extraction_issue_count: NonNegativeInt


def summarize(
    dataset: BenchmarkDataset,
    records: tuple[QuestionRun, ...],
    *,
    unreachable: tuple[str, ...] = (),
    extraction_issue_count: int = 0,
) -> BenchmarkSummary:
    questions = {q.question_id: q for q in dataset.questions}
    buckets = defaultdict(list)
    signatures = defaultdict(list)
    pair_values = defaultdict(list)
    for record in records:
        signature = deterministic_record(record)
        signature.pop("repeat")
        signatures[(record.question_id, record.strategy)].append(signature)
        question = questions[record.question_id]
        groups = (
            "all",
            f"type:{question.question_type}",
            f"hops:{question.reasoning_hops}",
            f"documents:{question.required_document_count}",
        )
        for evaluation in record.evaluations:
            for group in groups:
                buckets[(record.strategy, evaluation.k, group)].append((record, evaluation))
            pair_values[(record.question_id, record.strategy, evaluation.k)].append(evaluation)
    aggregates = []
    for (strategy, k, group), values in sorted(buckets.items()):
        aggregates.append(
            {
                "strategy": strategy,
                "k": k,
                "group": group,
                "questions": len({record.question_id for record, _ in values}),
                "retrieval_samples": len(values),
                "fact_recall_at_k": mean(e.raw.evidence_coverage for _, e in values),
                "raw_complete_rate": mean(float(e.raw.complete_evidence) for _, e in values),
                "context_evidence_coverage": mean(e.budgeted.evidence_coverage for _, e in values),
                "complete_evidence_rate": mean(
                    float(e.budgeted.complete_evidence) for _, e in values
                ),
                "mean_context_tokens": mean(e.context.token_count for _, e in values),
                "mean_selected_chunks": mean(len(e.context.selected_chunk_ids) for _, e in values),
                "budget_skipped_chunks": sum(len(e.context.skipped_budget) for _, e in values),
                "retrieval_median_ms": float(np.median([r.retrieval_ms for r, _ in values])),
                "retrieval_p95_ms": float(np.percentile([r.retrieval_ms for r, _ in values], 95)),
                "context_mean_ms": mean(e.context_ms for _, e in values),
            }
        )
    paired = []
    cutoffs = sorted({evaluation.k for record in records for evaluation in record.evaluations})
    for k in cutoffs:
        comparisons = []
        for question in dataset.questions:
            vector = pair_values.get((question.question_id, "vector", k))
            hybrid = pair_values.get((question.question_id, "hybrid", k))
            if vector and hybrid:
                comparisons.append(
                    {
                        "question_id": question.question_id,
                        "group_id": question.group_id,
                        "complete_difference": mean(
                            float(e.budgeted.complete_evidence) for e in hybrid
                        )
                        - mean(float(e.budgeted.complete_evidence) for e in vector),
                        "coverage_difference": mean(e.budgeted.evidence_coverage for e in hybrid)
                        - mean(e.budgeted.evidence_coverage for e in vector),
                    }
                )
        if comparisons:
            deltas = [item["complete_difference"] for item in comparisons]
            paired.append(
                {
                    "k": k,
                    "comparison": "hybrid-minus-vector",
                    "questions": len(comparisons),
                    "complete_difference": mean(deltas),
                    "coverage_difference": mean(
                        item["coverage_difference"] for item in comparisons
                    ),
                    "wins": sum(d > 0 for d in deltas),
                    "ties": sum(d == 0 for d in deltas),
                    "losses": sum(d < 0 for d in deltas),
                    "per_question": comparisons,
                }
            )
    unstable = tuple(
        f"{question}/{strategy}"
        for (question, strategy), values in sorted(signatures.items())
        if any(value != values[0] for value in values[1:])
    )
    return BenchmarkSummary(
        question_count=len(questions),
        group_count=len({q.group_id for q in dataset.questions}),
        record_count=len(records),
        split=dataset.questions[0].split,
        evaluation_sha256=evaluation_fingerprint(records),
        repeatable=(not unstable) if all(len(v) > 1 for v in signatures.values()) else None,
        unstable_queries=unstable,
        aggregates=tuple(aggregates),
        paired=tuple(paired),
        unreachable_gold_questions=unreachable,
        extraction_issue_count=extraction_issue_count,
    )


def render_report(summary: BenchmarkSummary) -> str:
    stability = str(summary.repeatable) if summary.repeatable is not None else "not assessed"
    lines = [
        "# Retrieval benchmark report",
        "",
        f"Split: **{summary.split}**. Questions: {summary.question_count}; "
        f"question groups: {summary.group_count}; timed retrievals: {summary.record_count}.",
        "",
        "Fixture results are development diagnostics, not held-out research evidence. "
        "Other splits still require an independently controlled tuning protocol.",
        "",
        "Rows average question scores; repeats measure stability and latency, not new questions. "
        "Fact Recall@K scores annotated facts before the token budget. "
        "Coverage and complete rate score the selected context after the shared budget.",
        "",
        "| Strategy | K | Fact Recall@K | Context coverage | Complete evidence | "
        "Mean tokens | Retrieval p95 ms |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary.aggregates:
        if row["group"] == "all":
            lines.append(
                f"| {row['strategy']} | {row['k']} | {row['fact_recall_at_k']:.3f} | "
                f"{row['context_evidence_coverage']:.3f} | {row['complete_evidence_rate']:.3f} | "
                f"{row['mean_context_tokens']:.1f} | {row['retrieval_p95_ms']:.2f} |"
            )
    lines.extend(["", "## Paired hybrid versus vector", ""])
    for row in summary.paired:
        lines.append(
            f"- K={row['k']}: complete-evidence difference "
            f"{100 * row['complete_difference']:+.1f} percentage points; "
            f"wins/ties/losses {row['wins']}/{row['ties']}/{row['losses']}."
        )
    if not summary.paired:
        lines.append("Both vector and hybrid are required for a paired comparison.")
    lines.extend(
        [
            "",
            "## Audit",
            "",
            f"- Identical rankings, context, and scores across repeats: {stability}.",
            f"- Evaluation fingerprint: `{summary.evaluation_sha256}`.",
            "- Gold questions not fully coverable by all generated chunks: "
            f"{list(summary.unreachable_gold_questions)}.",
            f"- Extraction issues: {summary.extraction_issue_count}.",
            "- `summary.json` includes groups by question type, reasoning hops, "
            "and required document count.",
            "- `results.jsonl` records missing facts, selected text, skipped chunks, "
            "paths, and component diagnostics.",
            "- Warm retrieval excludes loading/indexing; context assembly is timed separately. "
            "The same max-K retrieval feeds each cutoff, so retrieval times repeat across K rows.",
            "- No answer-quality scores, API cost estimates, confidence intervals, "
            "or significance claims are produced.",
            "",
            "See the recorded configuration and M7 study guide for definitions and limitations.",
            "",
        ]
    )
    return "\n".join(lines)
