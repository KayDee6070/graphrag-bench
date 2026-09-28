import json
from pathlib import Path

import pytest

from graphrag_bench.config import ChunkingConfig
from graphrag_bench.corpus import CorpusIndex
from graphrag_bench.extraction.llm.contracts import Completion, ModelSpec
from graphrag_bench.generation.config import load_generation_config
from graphrag_bench.generation.contracts import RetrievalReceipt
from graphrag_bench.ingestion.pipeline import ingest_to_directory
from graphrag_bench.ingestion.reader import load_ingestion
from graphrag_bench.models import RetrievalHit, RetrievalResult

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def generation_config():
    return load_generation_config(ROOT / "configs/generation.toml")


@pytest.fixture
def answer_source(tmp_path):
    texts = tmp_path / "texts"
    texts.mkdir()
    (texts / "orion.txt").write_text(
        "Orion is a retrieval method built on the Nova model.\n", encoding="utf-8"
    )
    (texts / "nova.txt").write_text(
        "Nova is a language model evaluated on the Harbor dataset.\n", encoding="utf-8"
    )
    output = tmp_path / "ingestion"
    ingest_to_directory(texts, output, ChunkingConfig(max_units=1, overlap_units=0))
    return output


@pytest.fixture
def answer_corpus(answer_source):
    batch, _ = load_ingestion(answer_source)
    return CorpusIndex(batch.documents, batch.chunks)


@pytest.fixture
def retrieval_receipt(answer_corpus, generation_config):
    return RetrievalReceipt(
        result=RetrievalResult(
            query="Which dataset evaluates the model used by Orion?",
            strategy="graph",
            hits=tuple(
                RetrievalHit(chunk_id=c.chunk_id, score=1.0) for c in answer_corpus.chunks.values()
            ),
            elapsed_ms=0.0,
        ),
        config=generation_config.retrieval,
    )


@pytest.fixture
def answer_provider(generation_config):
    class FakeProvider:
        """Handwritten responses and character counts, not a model-quality evaluation."""

        def __init__(self, response=None, config=None):
            config = config or generation_config
            self.spec = ModelSpec(
                provider="test-only-answer",
                model_id=config.model.model_id,
                revision=config.model.revision,
                settings=config.model.model_dump(mode="json"),
                versions={"test": "1"},
            )
            self.response = response
            self.requests = []

        def count_tokens(self, text):
            return len(text)

        def complete(self, request):
            self.requests.append(request)
            if isinstance(self.response, Exception):
                raise self.response
            value = self.response
            if value is None:
                value = {
                    "status": "answered",
                    "claims": [
                        {
                            "text": "Harbor.",
                            "citations": [
                                {"source_id": f"S{i}", "quote": piece.text}
                                for i, piece in enumerate(request.context.pieces, start=1)
                            ],
                        }
                    ],
                }
            if callable(value):
                value = value(request)
            if isinstance(value, Completion):
                return value
            return Completion(
                text=value if isinstance(value, str) else json.dumps(value),
                input_tokens=100,
                output_tokens=40,
                finish_reason="eos",
                elapsed_ms=1.0,
            )

    return FakeProvider
