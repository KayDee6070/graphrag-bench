"""Source-only local extraction timing sample; this does not produce a usable graph.

Receipts use the same cache keys as build-llm-graph. The final JSON records
historical completion times even when receipts are reused on a later invocation.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from time import perf_counter

from graphrag_bench.corpus import CorpusIndex
from graphrag_bench.extraction.llm.config import load_llm_config
from graphrag_bench.extraction.llm.extractor import (
    _parse,
    complete_chunk,
    validate_sample_responses,
)
from graphrag_bench.extraction.llm.prompt import extraction_fingerprint, selected_passage
from graphrag_bench.extraction.llm.provider import LocalTransformersProvider
from graphrag_bench.ingestion.reader import load_ingestion


def summarize_sample(rows: list[dict], model_invocation_count: int) -> dict:
    """Skip receipts are neither model successes nor evidence of zero inference cost."""
    invoked = [row for row in rows if row["origin"] == "provider"]
    mean_seconds = (
        sum(row["completion_seconds"] for row in invoked) / len(invoked) if invoked else None
    )
    return {
        "finish_reasons": dict(Counter(r["finish_reason"] for r in invoked)),
        "origins": dict(Counter(r["origin"] for r in rows)),
        "provider_schema_valid_count": sum(r["schema_valid"] for r in invoked),
        "provider_sample_count": len(invoked),
        "deterministic_abstention_count": len(rows) - len(invoked),
        "accepted_entities": sum(r["accepted_entities"] for r in rows),
        "accepted_assertions": sum(r["accepted_assertions"] for r in rows),
        "rejected_chunks": sum(r["rejected"] for r in rows),
        "mean_provider_completion_seconds": mean_seconds,
        "estimated_full_hours": (
            mean_seconds * model_invocation_count / 3600
            if mean_seconds is not None
            else (0.0 if model_invocation_count == 0 else None)
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="ingestion directory, never question labels")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--response-cache", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument(
        "--sample-pool",
        choices=("all", "model-invocations"),
        default="all",
        help="sample all chunks or only chunks whose selected passage reaches the model",
    )
    args = parser.parse_args()
    batch, hashes = load_ingestion(args.source)
    config = load_llm_config(args.config)
    chunks = sorted(batch.chunks, key=lambda c: (c.document_id, c.ordinal))
    corpus = CorpusIndex(batch.documents, batch.chunks)
    corpus_count = len(chunks)
    model_invocation_count = sum(bool(selected_passage(c.text, config)) for c in chunks)
    if args.sample_pool == "model-invocations":
        chunks = [chunk for chunk in chunks if selected_passage(chunk.text, config)]
    if not 2 <= args.samples <= len(chunks):
        parser.error("samples must be between 2 and the source chunk count")
    indices = [round(i * (len(chunks) - 1) / (args.samples - 1)) for i in range(args.samples)]
    started = perf_counter()
    provider = LocalTransformersProvider(config.model)
    model_load_seconds = perf_counter() - started
    rows = []
    for position in indices:
        chunk = chunks[position]
        started = perf_counter()
        receipt = complete_chunk(chunk, config, provider, args.response_cache)
        validated = validate_sample_responses(corpus, (receipt,), config, provider.spec)
        try:
            _parse(receipt.completion.text, style=config.prompt_style)
            schema_valid = True
        except ValueError:
            schema_valid = False
        row = {
            "sorted_position": position,
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "page": chunk.page,
            "request_sha256": receipt.request_sha256,
            "wall_seconds": perf_counter() - started,
            "completion_seconds": receipt.completion.elapsed_ms / 1000,
            "input_tokens": receipt.completion.input_tokens,
            "output_tokens": receipt.completion.output_tokens,
            "finish_reason": receipt.completion.finish_reason,
            "origin": receipt.origin,
            "schema_valid": schema_valid,
            "accepted_entities": len(validated.entities),
            "accepted_assertions": len(validated.relations),
            "rejected": bool(validated.rejected_chunks),
            "issues": dict(Counter(issue.code for issue in validated.issues)),
        }
        rows.append(row)
        print(json.dumps({"sample": len(rows), **row}), flush=True)
    print(
        json.dumps(
            {
                "kind": "source-only-extraction-preflight-v2",
                "selection": args.sample_pool + "-evenly-spaced-over-document-id-and-ordinal",
                "source_hashes": hashes,
                "provider": provider.spec.model_dump(mode="json"),
                "config": config.model_dump(mode="json"),
                "extraction_sha256": extraction_fingerprint(config, provider.spec),
                "model_load_seconds": model_load_seconds,
                "corpus_chunks": corpus_count,
                "model_invocation_chunks": model_invocation_count,
                "samples": rows,
                **summarize_sample(rows, model_invocation_count),
                "limitations": [
                    "Small systematic source sample, not a representative performance estimate.",
                    "Completion times are receipt values; cache reads do not reset them.",
                    "Projection uses provider samples only; excludes loading and validation.",
                    "A deterministic skip is not evidence that the source has no relation.",
                    "Cue selection may drop valid relations; recall has not been measured.",
                    "Source checks are replayed; assertions remain semantically unreviewed.",
                    "No graph artifact or retrieval-quality estimate is produced.",
                ],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
