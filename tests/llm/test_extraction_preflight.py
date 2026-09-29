import json
import runpy
import sys
from pathlib import Path

import pytest

from graphrag_bench.extraction.llm.config import load_llm_config
from graphrag_bench.extraction.llm.prompt import (
    extraction_fingerprint,
    make_request,
    request_fingerprint,
)
from graphrag_bench.ingestion.reader import load_ingestion

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/llm-extraction-indexed-3b.toml"


@pytest.fixture
def diagnostic():
    # Patch the script functions' actual global namespace, not runpy's returned copy.
    return runpy.run_path(str(ROOT / "scripts/check_extraction_sample.py"))["main"].__globals__


@pytest.fixture
def checks(llm_ingestion, tmp_path):
    batch, hashes = load_ingestion(llm_ingestion)
    path = tmp_path / "checks.json"
    path.write_text(
        json.dumps(
            {
                "source_hashes": hashes,
                "review_status": "draft",
                "selection": "test",
                "cases": [{"id": "negative", "chunk_id": batch.chunks[0].chunk_id, "expected": []}],
            }
        )
    )
    return path


def forbid_model(*args, **kwargs):
    raise AssertionError("preflight must not load a model")


def set_cli(monkeypatch, source, checks, *args):
    monkeypatch.setattr(
        sys, "argv", ["check_extraction_sample.py", str(source), "--checks", str(checks), *args]
    )


@pytest.mark.parametrize(
    "available,passed", [(18 * 1024**3, True), (1024**3, False), (None, False)]
)
def test_preflight_is_read_only_and_checks_memory(
    available, passed, diagnostic, llm_ingestion, checks, tmp_path, monkeypatch, capsys
):
    monkeypatch.setitem(diagnostic, "LocalTransformersProvider", forbid_model)
    monkeypatch.setitem(diagnostic, "available_memory_bytes", lambda: available)
    set_cli(
        monkeypatch,
        llm_ingestion,
        checks,
        "--config",
        str(CONFIG),
        "--preflight",
        "--min-available-gib",
        "18",
    )
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    if passed:
        diagnostic["main"]()
    else:
        with pytest.raises(SystemExit) as error:
            diagnostic["main"]()
        assert error.value.code == 2
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["memory_gate_passed"] is passed
    assert report["available_memory_bytes"] == available
    assert report["model"]["revision"] == "aa8e72537993ba99e69dfaafa59ed015b17504d1"
    assert report["cases"] == 1 and report["inference_run"] is False
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == before
    if not passed:
        assert "18 GiB required before model loading" in captured.err


@pytest.mark.parametrize("wrong_source", [False, True])
def test_inference_checks_source_and_memory_before_model_loading(
    wrong_source, diagnostic, llm_ingestion, checks, tmp_path, monkeypatch
):
    monkeypatch.setitem(diagnostic, "LocalTransformersProvider", forbid_model)
    monkeypatch.setitem(diagnostic, "available_memory_bytes", lambda: 1024**3)
    output, cache = tmp_path / "result.json", tmp_path / "cache"
    set_cli(
        monkeypatch,
        llm_ingestion,
        checks,
        "--config",
        str(CONFIG),
        "--response-cache",
        str(cache),
        "--output",
        str(output),
        "--min-available-gib",
        "18",
    )
    if wrong_source:
        payload = json.loads(checks.read_bytes())
        payload["source_hashes"] = {}
        checks.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="different source artifacts"):
            diagnostic["main"]()
    else:
        with pytest.raises(SystemExit) as error:
            diagnostic["main"]()
        assert error.value.code == 2
    assert not output.exists() and not cache.exists()


def test_sufficient_memory_allows_inference_and_replay_stays_model_free(
    diagnostic, llm_ingestion, checks, tmp_path, monkeypatch, make_provider, capsys
):
    config = load_llm_config(CONFIG)
    provider = make_provider({"relations": []}, config=config)
    monkeypatch.setitem(diagnostic, "LocalTransformersProvider", lambda model: provider)
    monkeypatch.setitem(diagnostic, "available_memory_bytes", lambda: 18 * 1024**3)
    output, cache = tmp_path / "result.json", tmp_path / "cache"
    set_cli(
        monkeypatch,
        llm_ingestion,
        checks,
        "--config",
        str(CONFIG),
        "--response-cache",
        str(cache),
        "--output",
        str(output),
        "--min-available-gib",
        "18",
    )
    diagnostic["main"]()
    assert len(provider.requests) == 1
    assert json.loads(output.read_bytes())["summary"]["negative_cases_without_assertions"] == 1
    monkeypatch.setitem(diagnostic, "LocalTransformersProvider", forbid_model)
    monkeypatch.setitem(diagnostic, "available_memory_bytes", forbid_model)
    set_cli(monkeypatch, llm_ingestion, checks, "--replay", str(output))
    diagnostic["main"]()
    assert "negative_cases_without_assertions" in capsys.readouterr().out


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "1e999", "invalid"])
def test_invalid_memory_threshold_fails_before_source_or_model(
    value, diagnostic, llm_ingestion, checks, monkeypatch
):
    monkeypatch.setitem(diagnostic, "LocalTransformersProvider", forbid_model)
    monkeypatch.setitem(diagnostic, "load_checks", forbid_model)
    set_cli(
        monkeypatch,
        llm_ingestion,
        checks,
        "--config",
        str(CONFIG),
        "--preflight",
        f"--min-available-gib={value}",
    )
    with pytest.raises(SystemExit) as error:
        diagnostic["main"]()
    assert error.value.code == 2


@pytest.mark.parametrize(
    "extra",
    [
        ["--preflight", "--config", str(CONFIG), "--output", "unused"],
        ["--preflight", "--config", str(CONFIG), "--response-cache", "unused"],
        ["--replay", "unused", "--preflight"],
        ["--replay", "unused", "--min-available-gib", "18"],
    ],
)
def test_ambiguous_modes_fail_before_loading(diagnostic, llm_ingestion, checks, monkeypatch, extra):
    monkeypatch.setitem(diagnostic, "load_checks", forbid_model)
    monkeypatch.setitem(diagnostic, "verify_checks", forbid_model)
    set_cli(monkeypatch, llm_ingestion, checks, *extra)
    with pytest.raises(SystemExit) as error:
        diagnostic["main"]()
    assert error.value.code == 2


@pytest.mark.parametrize(
    "contents,expected",
    [
        ("MemFree: 100 kB\nMemAvailable: 2048 kB\n", 2048 * 1024),
        ("MemAvailable: 0 kB\n", 0),
        ("MemFree: 2048 kB\n", None),
        ("MemAvailable: -1 kB\n", None),
        ("MemAvailable: 2048 MB\n", None),
        ("MemAvailable:\n", None),
        (None, None),
    ],
)
def test_memory_reader_distinguishes_unavailable_from_sufficient(
    diagnostic, tmp_path, contents, expected
):
    path = tmp_path / "meminfo"
    if contents is not None:
        path.write_text(contents)
    assert diagnostic["available_memory_bytes"](path) == expected


def test_capacity_trial_preserves_requests_but_separates_cache_keys(source_pair, make_provider):
    small = load_llm_config(ROOT / "configs/llm-extraction-indexed.toml")
    larger = load_llm_config(CONFIG)
    restored = larger.model_copy(
        update={
            "model": larger.model.model_copy(
                update={"model_id": small.model.model_id, "revision": small.model.revision}
            )
        }
    )
    assert restored == small
    chunk = source_pair[1][0]
    assert make_request(chunk, small) == make_request(chunk, larger)
    small_spec, larger_spec = make_provider(config=small).spec, make_provider(config=larger).spec
    assert extraction_fingerprint(small, small_spec) != extraction_fingerprint(larger, larger_spec)
    assert request_fingerprint(small_spec, make_request(chunk, small)) != request_fingerprint(
        larger_spec, make_request(chunk, larger)
    )
