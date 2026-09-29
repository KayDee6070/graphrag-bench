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
from graphrag_bench.extraction.llm.cache import get_completion
from graphrag_bench.extraction.llm.config import load_llm_config
from graphrag_bench.extraction.llm.contracts import ProposedExtraction
from graphrag_bench.extraction.llm.extractor import validate_sample_responses
from graphrag_bench.extraction.llm.prompt import make_request
from graphrag_bench.extraction.llm.provider import LocalTransformersProvider
from graphrag_bench.ingestion.reader import load_ingestion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="ingestion directory, never question labels")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--response-cache", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()
    batch, hashes = load_ingestion(args.source)
    chunks = sorted(batch.chunks, key=lambda c: (c.document_id, c.ordinal))
    corpus = CorpusIndex(batch.documents, batch.chunks)
    if not 2 <= args.samples <= len(chunks):
        parser.error("samples must be between 2 and the source chunk count")
    indices = [round(i * (len(chunks) - 1) / (args.samples - 1)) for i in range(args.samples)]
    config = load_llm_config(args.config)
    started = perf_counter()
    provider = LocalTransformersProvider(config.model)
    model_load_seconds = perf_counter() - started
    rows = []
    for position in indices:
        chunk = chunks[position]
        started = perf_counter()
        receipt = get_completion(provider, make_request(chunk, config), args.response_cache)
        validated = validate_sample_responses(corpus, (receipt,), config, provider.spec)
        try:
            ProposedExtraction.model_validate_json(receipt.completion.text)
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
            "schema_valid": schema_valid,
            "accepted_entities": len(validated.entities),
            "accepted_assertions": len(validated.relations),
            "rejected": bool(validated.rejected_chunks),
            "issues": dict(Counter(issue.code for issue in validated.issues)),
        }
        rows.append(row)
        print(json.dumps({"sample": len(rows), **row}), flush=True)
    total_seconds = sum(r["completion_seconds"] for r in rows)
    print(
        json.dumps(
            {
                "kind": "source-only-extraction-preflight-v1",
                "selection": "evenly-spaced-round-over-document-id-and-ordinal",
                "source_hashes": hashes,
                "provider": provider.spec.model_dump(mode="json"),
                "config": config.model_dump(mode="json"),
                "model_load_seconds": model_load_seconds,
                "corpus_chunks": len(chunks),
                "samples": rows,
                "finish_reasons": dict(Counter(r["finish_reason"] for r in rows)),
                "schema_valid_count": sum(r["schema_valid"] for r in rows),
                "accepted_assertions": sum(r["accepted_assertions"] for r in rows),
                "rejected_chunks": sum(r["rejected"] for r in rows),
                "estimated_full_hours": total_seconds / len(rows) * len(chunks) / 3600,
                "limitations": [
                    "Small systematic source sample, not a representative performance estimate.",
                    "Completion times are receipt values; cache reads do not reset them.",
                    "Source checks are replayed; assertions remain semantically unreviewed.",
                    "No graph artifact or retrieval-quality estimate is produced.",
                ],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
