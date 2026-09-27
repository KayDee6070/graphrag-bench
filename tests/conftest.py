from pathlib import Path

import numpy as np
import pytest

from graphrag_bench.embeddings.base import EmbeddingSpec
from graphrag_bench.embeddings.config import EmbeddingConfig
from graphrag_bench.fixtures import FixtureDataset, load_fixture


class FakeEmbeddingProvider:
    """Deliberately nonsemantic test double. Never selectable in the production CLI."""

    def __init__(self):
        self.document_calls = []
        self.query_calls = []
        config = EmbeddingConfig(model_id="test/encoder", revision="0" * 40)
        self.spec = EmbeddingSpec(
            provider="test-only",
            model_id=config.model_id,
            revision=config.revision,
            dimensions=3,
            settings=config.model_dump(mode="json"),
            versions={"test": "1"},
        )

    def embed_documents(self, texts):
        self.document_calls.append(texts)
        return np.array(
            [[len(t), t.casefold().count("a"), 1] for t in texts], dtype=np.float32
        ).reshape(len(texts), 3)

    def embed_query(self, text):
        self.query_calls.append(text)
        return np.array([1, 0, 0], dtype=np.float32)


@pytest.fixture
def embedding_provider():
    return FakeEmbeddingProvider()


@pytest.fixture(scope="session")
def fixture_root() -> Path:
    return Path(__file__).resolve().parents[1] / "datasets" / "fixtures" / "tiny"


@pytest.fixture(scope="session")
def corpus(fixture_root: Path) -> FixtureDataset:
    return load_fixture(fixture_root)
