"""Canonical UTF-8 JSON used by deterministic ingestion and graph artifacts."""

import json

from graphrag_bench.models import Record


def json_bytes(record: Record) -> bytes:
    serialized = json.dumps(
        record.model_dump(mode="json"),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return (serialized + "\n").encode("utf-8")
