"""Save complete answer receipts and replay source selection/citation checks offline."""

from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from platform import python_version
from typing import Literal

from graphrag_bench import __version__
from graphrag_bench.corpus import CorpusIndex
from graphrag_bench.extraction.llm.contracts import ModelSpec
from graphrag_bench.generation.config import GenerationConfig, GenerationError
from graphrag_bench.generation.contracts import (
    Answer,
    AnswerProvider,
    AnswerReceipt,
    AnswerRequest,
    RetrievalReceipt,
)
from graphrag_bench.generation.generator import generate_answer, render_answer, replay_answer
from graphrag_bench.ingestion.reader import load_ingestion
from graphrag_bench.models import NonNegativeInt, Record, Sha256, Text
from graphrag_bench.serialization import json_bytes


class AnswerManifest(Record):
    generation_version: Literal["cited-source-claims-v1"] = "cited-source-claims-v1"
    package_version: Text
    python_version: Text
    pydantic_version: Text
    package_sources_sha256: Sha256
    execution_mode: Literal["generation", "replay"]
    config: GenerationConfig
    provider: ModelSpec
    input_hashes: dict[str, Sha256]
    artifact_hashes: dict[str, Sha256]
    answer_status: Literal["answered", "insufficient_evidence", "invalid_output"]
    claim_count: NonNegativeInt
    citation_count: NonNegativeInt


def validate_answer_output(source: Path, output: Path, *protected: Path | None) -> None:
    if output.exists() or output.is_symlink():
        raise GenerationError(f"output already exists; choose a new directory: {output}")
    for directory in (source, *protected):
        if directory is not None and output.resolve().is_relative_to(directory.resolve()):
            raise GenerationError("answer output must be outside its source and input artifacts")


def _source_hash() -> str:
    package = Path(__file__).resolve().parents[1]
    return sha256(
        b"".join(
            path.relative_to(package).as_posix().encode("utf-8") + b"\0" + path.read_bytes() + b"\0"
            for path in sorted(package.rglob("*.py"))
        )
    ).hexdigest()


def _write_answer(
    output: Path,
    retrieval: RetrievalReceipt,
    request: AnswerRequest,
    receipt: AnswerReceipt,
    answer: Answer,
    config: GenerationConfig,
    spec: ModelSpec,
    input_hashes: dict[str, str],
    *,
    mode: Literal["generation", "replay"],
) -> AnswerManifest:
    artifacts = {
        "retrieval.json": json_bytes(retrieval),
        "request.json": json_bytes(request),
        "receipt.json": json_bytes(receipt),
        "answer.json": json_bytes(answer),
        "answer.md": render_answer(answer).encode("utf-8"),
    }
    manifest = AnswerManifest(
        package_version=__version__,
        python_version=python_version(),
        pydantic_version=version("pydantic"),
        package_sources_sha256=_source_hash(),
        execution_mode=mode,
        config=config,
        provider=spec,
        input_hashes=input_hashes,
        artifact_hashes={name: sha256(raw).hexdigest() for name, raw in artifacts.items()},
        answer_status=answer.status,
        claim_count=len(answer.claims),
        citation_count=sum(len(claim.citations) for claim in answer.claims),
    )
    try:
        output.mkdir(parents=True, exist_ok=False)
        for name, raw in artifacts.items():
            with (output / name).open("xb") as stream:
                stream.write(raw)
        with (output / "manifest.json").open("xb") as stream:
            stream.write(json_bytes(manifest))
    except OSError as error:
        raise GenerationError(f"cannot write answer artifacts: {error}") from error
    return manifest


def build_answer_to_directory(
    source: Path,
    output: Path,
    retrieval: RetrievalReceipt,
    config: GenerationConfig,
    provider: AnswerProvider,
) -> AnswerManifest:
    validate_answer_output(source, output)
    batch, input_hashes = load_ingestion(source)
    request, receipt, answer = generate_answer(
        retrieval, CorpusIndex(batch.documents, batch.chunks), config, provider
    )
    return _write_answer(
        output,
        retrieval,
        request,
        receipt,
        answer,
        config,
        provider.spec,
        input_hashes,
        mode="generation",
    )


def read_answer_run(
    directory: Path, source: Path
) -> tuple[AnswerManifest, RetrievalReceipt, AnswerRequest, AnswerReceipt, Answer]:
    try:
        manifest = AnswerManifest.model_validate_json((directory / "manifest.json").read_bytes())
        names = {"retrieval.json", "request.json", "receipt.json", "answer.json", "answer.md"}
        if set(manifest.artifact_hashes) != names:
            raise GenerationError("unexpected answer artifact file list")
        raw = {name: (directory / name).read_bytes() for name in names}
        if {
            name: sha256(value).hexdigest() for name, value in raw.items()
        } != manifest.artifact_hashes:
            raise GenerationError("answer artifact checksum mismatch")
        batch, input_hashes = load_ingestion(source)
        if manifest.input_hashes != input_hashes:
            raise GenerationError("answer was generated from different ingestion artifacts")
        retrieval = RetrievalReceipt.model_validate_json(raw["retrieval.json"])
        request = AnswerRequest.model_validate_json(raw["request.json"])
        receipt = AnswerReceipt.model_validate_json(raw["receipt.json"])
        answer = replay_answer(
            retrieval,
            request,
            receipt,
            CorpusIndex(batch.documents, batch.chunks),
            manifest.config,
            manifest.provider,
        )
        if (
            answer != Answer.model_validate_json(raw["answer.json"])
            or render_answer(answer).encode("utf-8") != raw["answer.md"]
        ):
            raise GenerationError("saved answer differs from replayed citation checks")
        if (manifest.answer_status, manifest.claim_count, manifest.citation_count) != (
            answer.status,
            len(answer.claims),
            sum(len(c.citations) for c in answer.claims),
        ):
            raise GenerationError("answer manifest status or count mismatch")
        return manifest, retrieval, request, receipt, answer
    except (OSError, UnicodeError, ValueError, RecursionError) as error:
        raise GenerationError(f"cannot load answer artifacts: {error}") from error


def replay_answer_to_directory(directory: Path, source: Path, output: Path) -> AnswerManifest:
    validate_answer_output(source, output, directory)
    manifest, retrieval, request, receipt, answer = read_answer_run(directory, source)
    return _write_answer(
        output,
        retrieval,
        request,
        receipt,
        answer,
        manifest.config,
        manifest.provider,
        manifest.input_hashes,
        mode="replay",
    )
