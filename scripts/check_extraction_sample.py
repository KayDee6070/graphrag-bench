"""Paired development diagnostic; draft reference facts never enter model requests.

Uses original complete chunks, then checks mechanically accepted assertions against
draft reference facts. No graph artifact, held-out score, or
full-corpus runtime projection is produced.
"""

import argparse
import json
from collections import Counter
from hashlib import sha256
from pathlib import Path

from graphrag_bench.corpus import CorpusIndex
from graphrag_bench.extraction.llm.config import LLMExtractionConfig, load_llm_config
from graphrag_bench.extraction.llm.contracts import Completion, ExtractionProvider, ModelSpec
from graphrag_bench.extraction.llm.extractor import (
    _parse,
    complete_chunk,
    validate_sample_responses,
)
from graphrag_bench.extraction.llm.prompt import extraction_fingerprint, request_fingerprint
from graphrag_bench.extraction.llm.provider import LocalTransformersProvider
from graphrag_bench.ingestion.reader import load_ingestion


def run_checks(
    source: Path,
    checks_path: Path,
    config: LLMExtractionConfig,
    provider: ExtractionProvider,
    cache: Path | None,
) -> dict:
    raw = checks_path.read_bytes()
    checks = json.loads(raw)
    batch, hashes = load_ingestion(source)
    if checks["source_hashes"] != hashes:
        raise ValueError("draft checks refer to different source artifacts")
    corpus = CorpusIndex(batch.documents, batch.chunks)
    cases = checks["cases"]
    ids = [case["chunk_id"] for case in cases]
    if len(set(ids)) != len(ids) or not set(ids) <= corpus.chunks.keys():
        raise ValueError("draft checks need unique, known source chunk IDs")
    spec = provider.spec
    rows = []
    for case in cases:
        # Only the original source chunk/config reach the inference interface.
        receipt = complete_chunk(corpus.chunks[case["chunk_id"]], config, provider, cache)
        result = validate_sample_responses(corpus, (receipt,), config, spec)
        entities = {e.entity_id: e for e in result.entities}
        assertions = [
            {
                "subject": entities[r.subject_id].name,
                "subject_type": entities[r.subject_id].entity_type,
                "predicate": r.predicate,
                "object": entities[r.object_id].name,
                "object_type": entities[r.object_id].entity_type,
                "evidence": [e.model_dump(mode="json") for e in r.evidence],
            }
            for r in result.relations
        ]
        expected = case["expected"]
        matches = [
            [
                i
                for i, fact in enumerate(expected)
                if a["subject"] in fact["subjects"]
                and a["object"] in fact["objects"]
                and a["predicate"] == fact["predicate"]
            ]
            for a in assertions
        ]
        try:
            _parse(receipt.completion.text, style=config.prompt_style)
            valid_schema = True
        except (ValueError, RecursionError):
            valid_schema = False
        rows.append(
            {
                "case_id": case["id"],
                "chunk_id": case["chunk_id"],
                "request_sha256": receipt.request_sha256,
                "completion": receipt.completion.model_dump(mode="json"),
                "origin": receipt.origin,
                "schema_valid": valid_schema,
                "rejected": bool(result.rejected_chunks),
                "issues": dict(Counter(i.code for i in result.issues)),
                "assertions": assertions,
                "expected_facts": len(expected),
                "matched_facts": len({i for found in matches for i in found}),
                "unmatched_assertions": sum(not found for found in matches),
            }
        )
        print(
            json.dumps(
                {
                    "case": case["id"],
                    "assertions": len(assertions),
                    "matched": rows[-1]["matched_facts"],
                    "issues": rows[-1]["issues"],
                }
            ),
            flush=True,
        )
    if provider.spec != spec:
        raise ValueError("provider specification changed during sample")
    return {
        "kind": "draft-extraction-source-check-v1",
        "review_status": checks["review_status"],
        "selection": checks["selection"],
        "checks_sha256": sha256(raw).hexdigest(),
        "source_hashes": hashes,
        "config": config.model_dump(mode="json"),
        "provider": spec.model_dump(mode="json"),
        "extraction_sha256": extraction_fingerprint(config, spec),
        "cases": rows,
        "summary": {
            "cases": len(rows),
            "expected_facts": sum(r["expected_facts"] for r in rows),
            "matched_facts": sum(r["matched_facts"] for r in rows),
            "unmatched_assertions": sum(r["unmatched_assertions"] for r in rows),
            "accepted_assertions": sum(len(r["assertions"]) for r in rows),
            "negative_cases_without_assertions": sum(
                not r["expected_facts"] and not r["assertions"] and not r["rejected"] for r in rows
            ),
            "schema_valid_responses": sum(r["schema_valid"] for r in rows),
            "rejected_responses": sum(r["rejected"] for r in rows),
            "output_limit_responses": sum(
                r["completion"]["finish_reason"] == "length" for r in rows
            ),
            "mean_recorded_completion_seconds": sum(r["completion"]["elapsed_ms"] for r in rows)
            / max(len(rows), 1)
            / 1000,
        },
        "limitations": [
            "Purposive development sample; draft labels await independent human review.",
            "Matches check names, direction and predicate; semantics and types need review.",
            "Unmatched assertions require review; they are not automatically false positives.",
            "No labels or reference facts are passed to inference; cached timings are historical.",
        ],
    }


def verify_checks(source: Path, checks_path: Path, report_path: Path) -> dict:
    """Rebuild requests and recompute source checks/metrics without loading a model."""
    saved = json.loads(report_path.read_bytes())
    config = LLMExtractionConfig.model_validate(saved["config"])
    spec = ModelSpec.model_validate(saved["provider"])
    receipts = {case["request_sha256"]: case for case in saved["cases"]}

    class ReceiptProvider:
        def __init__(self):
            self.spec = spec

        def complete(self, request):
            key = request_fingerprint(spec, request)
            if key not in receipts:
                raise ValueError("saved diagnostic request differs from current source/prompt")
            return Completion.model_validate(receipts[key]["completion"])

    replay = run_checks(source, checks_path, config, ReceiptProvider(), None)
    if replay != saved:
        raise ValueError("saved diagnostic differs from replayed source checks or draft metrics")
    return replay["summary"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--checks", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--response-cache", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replay", type=Path, help="verify a saved diagnostic without inference")
    args = parser.parse_args()
    if args.replay:
        if args.config or args.response_cache or args.output:
            parser.error("replay uses its saved recipe; omit config, cache, and output")
        print(json.dumps(verify_checks(args.source, args.checks, args.replay)))
        return
    if not args.config or not args.response_cache or not args.output:
        parser.error("inference requires config, response-cache, and output")
    if args.output.exists():
        parser.error("output already exists; choose a new file")
    if args.output.resolve().is_relative_to(args.source.resolve()):
        parser.error("diagnostic output must be outside source ingestion")
    config = load_llm_config(args.config)
    report = run_checks(
        args.source,
        args.checks,
        config,
        LocalTransformersProvider(config.model),
        args.response_cache,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
