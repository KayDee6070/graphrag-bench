from pathlib import Path

import pytest

from graphrag_bench.fixtures import FixtureDataset, load_fixture


@pytest.fixture(scope="session")
def fixture_root() -> Path:
    return Path(__file__).resolve().parents[1] / "datasets" / "fixtures" / "tiny"


@pytest.fixture(scope="session")
def corpus(fixture_root: Path) -> FixtureDataset:
    return load_fixture(fixture_root)
