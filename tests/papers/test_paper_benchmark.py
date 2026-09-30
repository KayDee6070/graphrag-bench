"""Paper comparison contracts; fake models test wiring, never retrieval quality."""

import json
import shutil
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

from graphrag_bench.benchmark.dataset import BenchmarkError
from graphrag_bench.benchmark.papers import (
    load_paper_inputs,
    paper_searches,
    run_paper_benchmark,
    verify_paper_benchmark,
)
from graphrag_bench.cli import main
from graphrag_bench.config import ChunkingConfig

ROOT = Path(__file__).resolve().parents[2]


def run(frozen_inputs, output):
    bundle, index, graph, config, provider = frozen_inputs
    return run_paper_benchmark(bundle, index, output, config, provider, graph_directory=graph)


def rehash(output, name):
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["artifact_hashes"][name] = sha256((output / name).read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))


def test_four_strategies_use_frozen_sources_and_verify_without_originals(frozen_inputs, tmp_path):
    bundle, index, graph, _, provider = frozen_inputs
    calls_before = len(provider.document_calls)
    output = tmp_path / "run"
    summary = run(frozen_inputs, output)
    assert summary.record_count == 8
    assert summary.split == "dev" and summary.repeatable
    assert len(provider.document_calls) == calls_before  # no re-embedding
    assert set(provider.query_calls) == {"Which gate does Eren's squad guard?"}
    for source in (bundle, index, graph):
        shutil.rmtree(source)
    assert verify_paper_benchmark(output) == summary
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["annotation_status"] == "pending-independent-review"
    assert manifest["graph_status"] == "unreviewed"
    assert main(["verify-paper-benchmark", str(output)]) == 0


def test_reference_answers_do_not_change_search(frozen_inputs):
    bundle, index, graph, config, provider = frozen_inputs
    inputs = load_paper_inputs(bundle, index, config, graph)
    altered = replace(inputs, dataset=replace(inputs.dataset, questions=()))
    original = paper_searches(inputs, provider, config)
    changed = paper_searches(altered, provider, config)
    for strategy in config.strategies:
        first, second = original[strategy]("Eren", 3), changed[strategy]("Eren", 3)
        assert first.result.hits == second.result.hits
        if strategy == "graph":
            assert any(hit.paths for hit in first.result.hits)


def test_baselines_require_no_graph(frozen_inputs, tmp_path):
    bundle, index, _, config, provider = frozen_inputs
    config = config.model_copy(update={"strategies": ("vector", "bm25")})
    output = tmp_path / "baseline"
    summary = run_paper_benchmark(bundle, index, output, config, provider)
    assert summary.record_count == 4
    assert not (output / "graph").exists()
    assert verify_paper_benchmark(output) == summary


@pytest.mark.parametrize("mismatch", ["chunking", "missing-graph", "provider", "existing-output"])
def test_mismatches_fail_before_queries(frozen_inputs, tmp_path, mismatch):
    bundle, index, graph, config, provider = frozen_inputs
    output = tmp_path / "run"
    if mismatch == "chunking":
        config = config.model_copy(update={"chunking": ChunkingConfig()})
    elif mismatch == "missing-graph":
        graph = None
    elif mismatch == "provider":
        provider.spec = provider.spec.model_copy(update={"revision": "1" * 40})
    else:
        output.mkdir()
    with pytest.raises(BenchmarkError):
        run_paper_benchmark(bundle, index, output, config, provider, graph_directory=graph)
    assert provider.query_calls == []


@pytest.mark.parametrize("damage", ["checksum", "score", "inventory", "context", "report", "path"])
def test_offline_verifier_detects_tampering(frozen_inputs, tmp_path, damage):
    output = tmp_path / "run"
    run(frozen_inputs, output)
    if damage == "path":
        path = output / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["artifact_hashes"]["../escape"] = "0" * 64
        path.write_text(json.dumps(manifest))
    elif damage == "report":
        (output / "report.md").write_text("All questions independently reviewed.")
        rehash(output, "report.md")
    else:
        path = output / "results.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if damage == "inventory":
            rows.pop()
        elif damage == "score":
            rows[0]["evaluations"][0]["raw"]["complete_evidence"] = not rows[0]["evaluations"][0][
                "raw"
            ]["complete_evidence"]
        elif damage == "context":
            rows[0]["evaluations"][0]["context"]["selected_chunk_ids"] = ["invented"]
        else:
            rows.pop()
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        if damage != "checksum":
            rehash(output, "results.jsonl")
    with pytest.raises(BenchmarkError):
        verify_paper_benchmark(output)


def test_cli_validates_missing_graph_before_model_load(frozen_inputs, tmp_path, monkeypatch):
    bundle, index, _, _, _ = frozen_inputs
    monkeypatch.setattr(
        "graphrag_bench.cli.SentenceTransformerProvider",
        lambda *a, **kw: pytest.fail("model must not load"),
    )
    assert (
        main(
            [
                "benchmark-papers",
                str(bundle),
                "--index",
                str(index),
                "--config",
                str(ROOT / "configs/benchmark-papers.toml"),
                "--output",
                str(tmp_path / "run"),
            ]
        )
        == 1
    )
