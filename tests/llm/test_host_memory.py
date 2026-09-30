"""Host RAM gating; a threshold is headroom guidance, never a reservation."""

from pathlib import Path

import pytest

from graphrag_bench.cli import main
from graphrag_bench.extraction.llm.config import LLMError
from graphrag_bench.extraction.llm.host import (
    GIB,
    available_memory_bytes,
    memory_check,
    positive_gib,
    require_memory,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def meminfo(tmp_path, monkeypatch):
    def write(text):
        path = tmp_path / "meminfo"
        path.write_text(text, encoding="ascii")
        monkeypatch.setattr(
            "graphrag_bench.extraction.llm.host.available_memory_bytes",
            lambda: available_memory_bytes(path),
        )
        return path

    return write


def test_available_memory_reads_kilobytes(tmp_path):
    path = tmp_path / "meminfo"
    path.write_text("MemTotal:       32000000 kB\nMemAvailable:   18874368 kB\n", encoding="ascii")

    assert available_memory_bytes(path) == 18874368 * 1024


@pytest.mark.parametrize(
    "text",
    [
        "MemTotal: 32000000 kB\n",
        "MemAvailable:   18874368\n",
        "MemAvailable:   18874368 MB\n",
        "MemAvailable:   lots kB\n",
    ],
)
def test_unreadable_available_memory_is_unknown(tmp_path, text):
    path = tmp_path / "meminfo"
    path.write_text(text, encoding="ascii")

    assert available_memory_bytes(path) is None


def test_missing_meminfo_is_unknown(tmp_path):
    assert available_memory_bytes(tmp_path / "absent") is None


def test_no_threshold_reports_without_deciding(meminfo):
    meminfo("MemAvailable:   1024 kB\n")

    check = require_memory(None)

    assert check["min_available_gib"] is None
    assert check["sufficient"] is None
    assert check["available_bytes"] == 1024 * 1024


def test_sufficient_memory_passes_and_is_recorded(meminfo):
    meminfo("MemAvailable:   20971520 kB\n")

    check = require_memory(18)

    assert check == {
        "min_available_gib": 18,
        "available_bytes": 20 * GIB,
        "sufficient": True,
    }


def test_insufficient_memory_refuses_before_any_model(meminfo):
    meminfo("MemAvailable:   16777216 kB\n")

    with pytest.raises(LLMError, match="16.00 GiB is below the required 18 GiB"):
        require_memory(18)


def test_unknown_memory_is_never_treated_as_sufficient(meminfo):
    meminfo("MemTotal: 32000000 kB\n")

    assert memory_check(18)["sufficient"] is False
    with pytest.raises(LLMError, match="cannot read available RAM"):
        require_memory(18)


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "plenty", ""])
def test_thresholds_must_be_finite_and_positive(value):
    with pytest.raises(ValueError, match="finite positive GiB value"):
        positive_gib(value)


def test_threshold_accepts_fractional_gib():
    assert positive_gib("17.5") == 17.5


def test_build_llm_graph_refuses_short_memory_before_loading_a_model(
    llm_ingestion, tmp_path, make_provider, monkeypatch, capsys
):
    constructed = []

    def fail_if_constructed(*args, **kwargs):
        constructed.append(kwargs)
        return make_provider()

    monkeypatch.setattr("graphrag_bench.cli.LocalTransformersProvider", fail_if_constructed)
    monkeypatch.setattr(
        "graphrag_bench.extraction.llm.host.available_memory_bytes", lambda: 16 * GIB
    )
    output = tmp_path / "graph"

    code = main(
        [
            "build-llm-graph",
            str(llm_ingestion),
            "--config",
            str(ROOT / "configs/llm-extraction.toml"),
            "--output",
            str(output),
            "--min-available-gib",
            "18",
        ]
    )

    assert code == 1
    assert "below the required 18 GiB" in capsys.readouterr().err
    assert not constructed
    assert not output.exists()
