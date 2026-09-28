import copy
import json
from pathlib import Path

import pytest

from graphrag_bench.config import ChunkingConfig
from graphrag_bench.extraction.llm.config import load_llm_config
from graphrag_bench.extraction.llm.contracts import Completion, ModelSpec
from graphrag_bench.ingestion.chunker import chunk_document
from graphrag_bench.ingestion.pipeline import ingest_to_directory
from graphrag_bench.models import Document, text_sha256

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def llm_config():
    return load_llm_config(ROOT / "configs" / "llm-extraction.toml")


@pytest.fixture
def proposals():
    return {
        "Orion uses Nova.": {
            "entities": [
                {"id": "e1", "name": "Orion", "entity_type": "Model"},
                {"id": "e2", "name": "Nova", "entity_type": "Model"},
            ],
            "relations": [
                {"subject": "e1", "predicate": "USES", "object": "e2", "quote": "Orion uses Nova."}
            ],
        },
        "Nova is evaluated on Harbor.": {
            "entities": [
                {"id": "e1", "name": "Nova", "entity_type": "Model"},
                {"id": "e2", "name": "Harbor", "entity_type": "Dataset"},
            ],
            "relations": [
                {
                    "subject": "e1",
                    "predicate": "EVALUATED_ON",
                    "object": "e2",
                    "quote": "Nova is evaluated on Harbor.",
                }
            ],
        },
    }


@pytest.fixture
def make_provider(llm_config, proposals):
    class FakeProvider:
        """Prewritten responses, deliberately not a language model or extraction-quality test."""

        def __init__(self, response=None, config=None):
            config = config or llm_config
            self.spec = ModelSpec(
                provider="test-double",
                model_id=config.model.model_id,
                revision=config.model.revision,
                settings=config.model.model_dump(mode="json"),
                versions={"test": "1"},
            )
            self.requests = []
            self.response = response

        def complete(self, request):
            self.requests.append(request)
            if isinstance(self.response, Exception):
                raise self.response
            value = self.response
            if value is None:
                value = next(
                    copy.deepcopy(proposal)
                    for sentence, proposal in proposals.items()
                    if sentence in request.chunk.text
                )
            elif callable(value):
                value = value(request)
            if isinstance(value, Completion):
                return value
            return Completion(
                text=value if isinstance(value, str) else json.dumps(value),
                input_tokens=100,
                output_tokens=50,
                finish_reason="eos",
                elapsed_ms=1.0,
            )

    return FakeProvider


@pytest.fixture
def source_pair(proposals):
    documents = tuple(
        Document(
            document_id=f"doc-{i}",
            title=f"Source {i}",
            text=text,
            source_uri=f"fixture:{i}",
            content_sha256=text_sha256(text),
        )
        for i, text in enumerate(proposals)
    )
    chunks = tuple(
        chunk
        for document in documents
        for chunk in chunk_document(document, ChunkingConfig(max_units=1, overlap_units=0))
    )
    return documents, chunks


@pytest.fixture
def llm_ingestion(tmp_path, proposals):
    source = tmp_path / "texts"
    source.mkdir()
    for number, text in enumerate(proposals):
        (source / f"{number}.txt").write_text(text, encoding="utf-8")
    output = tmp_path / "ingestion"
    ingest_to_directory(source, output, ChunkingConfig(max_units=1, overlap_units=0))
    return output
