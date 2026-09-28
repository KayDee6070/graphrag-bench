"""Content-addressed completion receipts; corrupt entries fail instead of regenerating."""

from hashlib import sha256
from pathlib import Path

from graphrag_bench.extraction.llm.config import LLMError
from graphrag_bench.extraction.llm.contracts import (
    Completion,
    ExtractionProvider,
    ExtractionRequest,
    ModelSpec,
    ResponseRecord,
)
from graphrag_bench.extraction.llm.prompt import request_fingerprint
from graphrag_bench.models import Record, Sha256
from graphrag_bench.serialization import json_bytes


class CacheEntry(Record):
    provider: ModelSpec
    record: ResponseRecord
    completion_sha256: Sha256


def get_completion(
    provider: ExtractionProvider, request: ExtractionRequest, cache: Path | None
) -> ResponseRecord:
    spec = provider.spec
    key = request_fingerprint(spec, request)
    path = cache / f"{key}.json" if cache is not None else None
    if path is not None and path.exists():
        try:
            entry = CacheEntry.model_validate_json(path.read_bytes())
            if (
                entry.provider != spec
                or entry.record.request != request
                or entry.record.request_sha256 != key
                or entry.completion_sha256
                != sha256(json_bytes(entry.record.completion)).hexdigest()
            ):
                raise ValueError("receipt contents differ from key or completion checksum")
            return entry.record
        except (OSError, UnicodeError, ValueError) as error:
            raise LLMError(f"invalid response cache entry {path}: {error}") from error
    # Provider failures abort the run. A model's invalid JSON is still a completed response
    # and is preserved so validation can explain it without paying for another inference.
    response = provider.complete(request)
    if not isinstance(response, Completion):
        raise LLMError("provider must return a validated Completion record")
    record = ResponseRecord(request_sha256=key, request=request, completion=response)
    if path is not None:
        entry = CacheEntry(
            provider=spec,
            record=record,
            completion_sha256=sha256(json_bytes(response)).hexdigest(),
        )
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(json_bytes(entry))
        except OSError as error:
            raise LLMError(f"cannot save response cache {path}: {error}") from error
    return record
