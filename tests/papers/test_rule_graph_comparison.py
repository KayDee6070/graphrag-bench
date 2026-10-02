"""Deterministic rule graphs as a comparison arm; wiring only, never retrieval quality."""

import json
import shutil
from hashlib import sha256

import pytest

from graphrag_bench.benchmark.config import BenchmarkConfig
from graphrag_bench.benchmark.dataset import BenchmarkError
from graphrag_bench.benchmark.papers import (
    RULE_GRAPH_FILES,
    load_paper_inputs,
    run_paper_benchmark,
    verify_paper_benchmark,
)
from graphrag_bench.graph.pipeline import build_graph_to_directory

RULES = """
[[rules]]
rule_id = "squad-membership"
pattern = '(?P<subject>[\\w-]+) belongs to (?P<object>[\\w-]+(?: [\\w-]+)*)\\.'
subject_type = "Person"
object_type = "Squad"
predicate = "BELONGS_TO"
declares_subject = true

[[rules]]
rule_id = "squad-assignment"
pattern = '(?P<subject>[\\w-]+(?: [\\w-]+)*) guards the (?P<object>[\\w-]+(?: [\\w-]+)*)\\.'
subject_type = "Squad"
object_type = "Place"
predicate = "GUARDS"
declares_subject = true
"""


@pytest.fixture
def rule_graph(frozen_inputs, tmp_path):
    bundle = frozen_inputs[0]
    rules_path = tmp_path / "rules.toml"
    rules_path.write_text(RULES, encoding="utf-8")
    output = tmp_path / "rule-graph"
    manifest = build_graph_to_directory(bundle / "ingestion", output, rules_path)
    assert manifest.assertion_count == 2
    return output


def test_rule_graph_runs_four_strategies_and_verifies_offline(frozen_inputs, tmp_path, rule_graph):
    bundle, index, llm_graph, config, provider = frozen_inputs
    output = tmp_path / "run"

    summary = run_paper_benchmark(
        bundle, index, output, config, provider, graph_directory=rule_graph
    )

    assert summary.record_count == 8
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["graph_status"] == "rule-based-unreviewed"
    copied = {
        name.removeprefix("graph/")
        for name in manifest["artifact_hashes"]
        if name.startswith("graph/")
    }
    assert copied == RULE_GRAPH_FILES
    assert {path.name for path in (output / "graph").iterdir()} == RULE_GRAPH_FILES
    report = (output / "report.md").read_text()
    assert "Graph source: deterministic rule graph" in report
    assert "it is a floor, not a graph-method ceiling" in report
    for source in (bundle, index, llm_graph, rule_graph):
        shutil.rmtree(source)
    assert verify_paper_benchmark(output) == summary


def test_rule_graph_issue_count_comes_from_its_own_receipt(frozen_inputs, rule_graph):
    bundle, index, _, config, _ = frozen_inputs

    inputs = load_paper_inputs(bundle, index, config, rule_graph)

    receipt = json.loads((rule_graph / "manifest.json").read_text())
    assert inputs.graph_status == "rule-based-unreviewed"
    assert inputs.issue_count == receipt["issue_count"]
    assert inputs.graph is not None and inputs.graph.edge_count == receipt["assertion_count"]


def test_relabelled_graph_kind_is_rejected(frozen_inputs, tmp_path, rule_graph):
    bundle, index, _, config, provider = frozen_inputs
    output = tmp_path / "run"
    run_paper_benchmark(bundle, index, output, config, provider, graph_directory=rule_graph)
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["graph_status"] = "unreviewed"
    path.write_text(json.dumps(manifest))

    with pytest.raises(BenchmarkError, match="unexpected paper experiment artifact file list"):
        verify_paper_benchmark(output)


def test_recorded_graph_status_must_match_compared_strategies(frozen_inputs, tmp_path, rule_graph):
    bundle, index, _, config, provider = frozen_inputs
    output = tmp_path / "run"
    run_paper_benchmark(bundle, index, output, config, provider, graph_directory=rule_graph)
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["graph_status"] = "not-used"
    path.write_text(json.dumps(manifest))

    with pytest.raises(BenchmarkError, match="disagrees with the compared strategies"):
        verify_paper_benchmark(output)


def test_graph_manifest_without_an_extractor_version_is_rejected(
    frozen_inputs, tmp_path, rule_graph
):
    bundle, index, _, config, _ = frozen_inputs
    path = rule_graph / "manifest.json"
    manifest = json.loads(path.read_text())
    del manifest["extractor_version"]
    path.write_text(json.dumps(manifest))

    with pytest.raises(BenchmarkError, match="must record an extractor version"):
        load_paper_inputs(bundle, index, config, rule_graph)


def test_vector_only_comparison_still_refuses_a_graph_directory(frozen_inputs, rule_graph):
    bundle, index, _, config, _ = frozen_inputs
    vector_only = BenchmarkConfig(
        chunking=config.chunking, cutoffs=config.cutoffs, repeats=1, strategies=("vector", "bm25")
    )

    with pytest.raises(BenchmarkError, match="--graph is required exactly when"):
        load_paper_inputs(bundle, index, vector_only, rule_graph)


def test_rule_graph_artifacts_are_copied_verbatim(frozen_inputs, tmp_path, rule_graph):
    bundle, index, _, config, provider = frozen_inputs
    output = tmp_path / "run"
    run_paper_benchmark(bundle, index, output, config, provider, graph_directory=rule_graph)

    for name in RULE_GRAPH_FILES:
        original = (rule_graph / name).read_bytes()
        assert (output / "graph" / name).read_bytes() == original
        manifest = json.loads((output / "manifest.json").read_text())
        assert manifest["artifact_hashes"][f"graph/{name}"] == sha256(original).hexdigest()


def test_a_dev_report_still_states_zero_held_out_questions(frozen_inputs, tmp_path, rule_graph):
    """Regression: dropping this line silently invalidated every previously saved run."""
    bundle, index, _, config, provider = frozen_inputs
    output = tmp_path / "run"
    run_paper_benchmark(bundle, index, output, config, provider, graph_directory=rule_graph)

    report = (output / "report.md").read_text()

    assert "Split: dev. Held-out questions: 0." in report
    assert "# Frozen real-paper development comparison" in report
    assert "This is a development diagnostic" in report
