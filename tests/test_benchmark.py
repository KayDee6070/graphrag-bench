import json
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

from graphrag_bench.benchmark.config import BenchmarkConfig, load_benchmark_config
from graphrag_bench.benchmark.dataset import BenchmarkError, load_benchmark
from graphrag_bench.benchmark.pipeline import prepare_retrieval, run_benchmark, verify_benchmark_run
from graphrag_bench.benchmark.report import summarize
from graphrag_bench.benchmark.runner import SearchObservation, run_queries
from graphrag_bench.cli import main
from graphrag_bench.extraction import load_rules
from graphrag_bench.models import BenchmarkQuestion, RetrievalHit, RetrievalResult, RunManifest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def dataset(fixture_root):
    return load_benchmark(
        fixture_root / "corpus/documents.jsonl",
        fixture_root / "gold/questions.jsonl",
        split="fixture",
    )


@pytest.fixture
def benchmark_provider(embedding_provider):
    embedding_provider.count_tokens = lambda text: len(text.split())
    return embedding_provider


@pytest.fixture
def rules():
    return load_rules(ROOT / "configs/extraction.toml")


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("duplicate", "duplicate question IDs"),
        ("bad_quote", "gold source text mismatch"),
        ("bad_document", "gold source text mismatch"),
        ("mixed_group", "group crosses splits"),
        ("blank", "blank"),
        ("empty", "empty benchmark input"),
    ],
)
def test_annotation_validation_precedes_split_selection(fixture_root, tmp_path, mutation, match):
    original = (fixture_root / "gold/questions.jsonl").read_text()
    data = [json.loads(line) for line in original.splitlines()]
    if mutation == "duplicate":
        data.append(data[0])
    elif mutation == "bad_quote":
        target = data[0]["sufficient_evidence_sets"][0]["facts"][0]["spans"][0]
        target["text"] = "X" * len(target["text"])
    elif mutation == "bad_document":
        data[0]["relevant_document_ids"] = ["missing"]
        data[0]["sufficient_evidence_sets"][0]["facts"][0]["spans"][0]["document_id"] = "missing"
    elif mutation == "mixed_group":
        data[1]["group_id"], data[1]["split"] = data[0]["group_id"], "test"
    text = "\n".join(json.dumps(row) for row in data) + "\n"
    if mutation == "blank":
        text += "\n"
        match = "Invalid JSON"
    elif mutation == "empty":
        text = ""
    path = tmp_path / "questions.jsonl"
    path.write_text(text)
    with pytest.raises(BenchmarkError, match=match):
        load_benchmark(fixture_root / "corpus/documents.jsonl", path, split="fixture")


def test_fixture_split_is_never_silently_promoted_to_held_out(fixture_root, dataset):
    assert len(dataset.questions) == 20
    with pytest.raises(BenchmarkError, match="no questions"):
        load_benchmark(
            fixture_root / "corpus/documents.jsonl",
            fixture_root / "gold/questions.jsonl",
            split="test",
        )


def test_explicit_split_selection_preserves_whole_file_hash_and_checks_gold_spans(
    fixture_root, tmp_path
):
    data = [
        json.loads(line)
        for line in (fixture_root / "gold/questions.jsonl").read_text().splitlines()
    ]
    for row in data:
        row["split"] = "dev" if row["group_id"] == "alder-base" else "test"
    path = tmp_path / "questions.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in data) + "\n")
    dev = load_benchmark(fixture_root / "corpus/documents.jsonl", path, split="dev")
    test = load_benchmark(fixture_root / "corpus/documents.jsonl", path, split="test")
    assert len(dev.questions) + len(test.questions) == 20
    assert dev.benchmark_sha256 == test.benchmark_sha256
    assert {q.group_id for q in dev.questions}.isdisjoint(q.group_id for q in test.questions)


@pytest.mark.parametrize(
    "settings",
    [
        {"strategies": []},
        {"strategies": ["graph", "graph"]},
        {"strategies": ["unknown"]},
        {"cutoffs": []},
        {"cutoffs": [10, 5]},
        {"cutoffs": [5, 5]},
        {"cutoffs": [True]},
        {"repeats": 0},
        {"max_context_tokens": 0},
        {"seed": -1},
        {"unknown": 1},
    ],
)
def test_invalid_experiment_policies_rejected(settings):
    with pytest.raises(ValidationError):
        BenchmarkConfig.model_validate(settings)


def test_configuration_matches_documented_defaults(tmp_path):
    assert load_benchmark_config(ROOT / "configs/benchmark.toml") == BenchmarkConfig()
    path = tmp_path / "bad.toml"
    for content in ("[unknown]\nseed=0", "benchmark=4", "[benchmark"):
        path.write_text(content)
        with pytest.raises(BenchmarkError):
            load_benchmark_config(path)


def test_complete_pipeline_reproducibility_artifacts_and_no_gold_graph(
    dataset, benchmark_provider, rules, tmp_path, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("benchmark must not load the oracle fixture graph")

    monkeypatch.setattr("graphrag_bench.fixtures.load_fixture", forbidden)
    config = BenchmarkConfig(repeats=2)
    first = run_benchmark(dataset, tmp_path / "one", config, rules, benchmark_provider)
    second = run_benchmark(dataset, tmp_path / "two", config, rules, benchmark_provider)
    assert first.record_count == 20 * 4 * 2 and first.question_count == 20
    assert first.repeatable and second.repeatable
    assert first.evaluation_sha256 == second.evaluation_sha256
    assert first.unreachable_gold_questions == ()
    assert first.extraction_issue_count == 2
    assert first == verify_benchmark_run(tmp_path / "one")
    assert len(benchmark_provider.document_calls) == 2
    manifest = RunManifest.model_validate_json((tmp_path / "one/manifest.json").read_bytes())
    assert manifest.corpus_sha256 == dataset.corpus_sha256
    assert manifest.benchmark_sha256 == dataset.benchmark_sha256
    assert manifest.config["tokenizer"]["truncation"] is False
    assert manifest.config["code"]["package_sources_sha256"]
    rows = [json.loads(line) for line in (tmp_path / "one/results.jsonl").read_text().splitlines()]
    assert all(e["context"]["token_count"] <= 2000 for row in rows for e in row["evaluations"])
    assert any(
        row["observation"]["diagnostics"].get("graph_trace")
        for row in rows
        if row["strategy"] == "hybrid"
    )
    for row in first.aggregates:
        if row["group"] == "all":
            assert row["questions"] == 20 and row["retrieval_samples"] == 40
    assert all(p["wins"] + p["ties"] + p["losses"] == 20 for p in first.paired)
    before = len(benchmark_provider.document_calls)
    with pytest.raises(BenchmarkError, match="output already exists"):
        run_benchmark(dataset, tmp_path / "one", config, rules, benchmark_provider)
    assert len(benchmark_provider.document_calls) == before
    (tmp_path / "one/report.md").write_text("corrupted")
    with pytest.raises(BenchmarkError, match="checksum mismatch"):
        verify_benchmark_run(tmp_path / "one")


def test_gold_answers_and_entity_labels_never_enter_retrieval(dataset, benchmark_provider, rules):
    config = BenchmarkConfig(strategies=("graph", "bm25"), repeats=1, cutoffs=(5,))
    prepared = prepare_retrieval(dataset.documents, config, rules, benchmark_provider)
    seen = []
    searches = {}
    for strategy, search in prepared.searches.items():

        def audited(query, k, search=search):
            assert isinstance(query, str) and isinstance(k, int)
            seen.append(query)
            return search(query, k)

        searches[strategy] = audited
    original = run_queries(dataset, prepared.corpus, searches, benchmark_provider, config)
    changed = replace(
        dataset,
        questions=tuple(
            BenchmarkQuestion.model_validate(
                q.model_dump()
                | {"expected_answer": "SECRET-GOLD", "relevant_entity_ids": ("SECRET-GOLD",)}
            )
            for q in dataset.questions
        ),
    )
    repeated = run_queries(changed, prepared.corpus, searches, benchmark_provider, config)
    assert all("SECRET-GOLD" not in query for query in seen)
    assert [r.observation.result.hits for r in original] == [
        r.observation.result.hits for r in repeated
    ]
    assert benchmark_provider.document_calls == benchmark_provider.query_calls == []


def test_budget_loss_is_separate_from_raw_fact_recall(dataset, benchmark_provider, rules):
    config = BenchmarkConfig(strategies=("bm25",), cutoffs=(25,), max_context_tokens=1, repeats=1)
    prepared = prepare_retrieval(dataset.documents, config, rules, benchmark_provider)
    records = run_queries(dataset, prepared.corpus, prepared.searches, benchmark_provider, config)
    assert any(r.evaluations[0].raw.complete_evidence for r in records)
    assert all(not r.evaluations[0].budgeted.complete_evidence for r in records)
    assert all(r.evaluations[0].context.pieces == () for r in records)
    assert summarize(dataset, records).repeatable is None


def test_macro_summary_and_paired_differences_have_hand_checked_denominators(
    dataset, benchmark_provider, rules
):
    dataset = replace(dataset, questions=dataset.questions[:2])
    config = BenchmarkConfig(strategies=("vector", "hybrid"), cutoffs=(25,), repeats=3)
    prepared = prepare_retrieval(dataset.documents, config, rules, benchmark_provider)

    def search(strategy):
        def retrieve(query, k):
            successful = strategy == "hybrid" or query == dataset.questions[0].question
            return SearchObservation(
                result=RetrievalResult(
                    query=query,
                    strategy=strategy,
                    elapsed_ms=0.0,
                    hits=tuple(RetrievalHit(chunk_id=i, score=1.0) for i in prepared.corpus.chunks)
                    if successful
                    else (),
                )
            )

        return retrieve

    records = run_queries(
        dataset,
        prepared.corpus,
        {s: search(s) for s in config.strategies},
        benchmark_provider,
        config,
    )
    summary = summarize(dataset, records)
    overall = {r["strategy"]: r for r in summary.aggregates if r["group"] == "all"}
    assert overall["vector"]["complete_evidence_rate"] == 0.5
    assert overall["hybrid"]["complete_evidence_rate"] == 1
    assert overall["vector"]["questions"] == 2 and overall["vector"]["retrieval_samples"] == 6
    assert summary.paired[0]["complete_difference"] == 0.5
    assert (summary.paired[0]["wins"], summary.paired[0]["ties"], summary.paired[0]["losses"]) == (
        1,
        1,
        0,
    )


def test_manifest_file_names_cannot_point_outside_run(dataset, benchmark_provider, rules, tmp_path):
    output = tmp_path / "run"
    run_benchmark(
        dataset, output, BenchmarkConfig(strategies=("bm25",), repeats=1), rules, benchmark_provider
    )
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["artifact_hashes"]["../outside"] = "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(BenchmarkError, match="unexpected benchmark artifact file list"):
        verify_benchmark_run(output)


def test_runner_detects_unstable_rankings_without_counting_repeats_as_questions(
    dataset, benchmark_provider, rules
):
    dataset = replace(dataset, questions=dataset.questions[:1])
    config = BenchmarkConfig(strategies=("vector",), cutoffs=(1,), repeats=2)
    prepared = prepare_retrieval(dataset.documents, config, rules, benchmark_provider)
    ids = list(prepared.corpus.chunks)
    calls = []

    def unstable(query, k):
        calls.append(query)
        return SearchObservation(
            result=RetrievalResult(
                query=query,
                strategy="vector",
                elapsed_ms=0.0,
                hits=(RetrievalHit(chunk_id=ids[len(calls) % 2], score=1.0),),
            )
        )

    records = run_queries(
        dataset, prepared.corpus, {"vector": unstable}, benchmark_provider, config
    )
    summary = summarize(dataset, records)
    assert len(calls) == 3 and summary.record_count == 2 and summary.question_count == 1
    assert not summary.repeatable and summary.unstable_queries == ("q01/vector",)


@pytest.mark.parametrize("invalid", ["query", "strategy", "count", "unknown_chunk"])
def test_retriever_contract_violations_stop_the_experiment(
    dataset, benchmark_provider, rules, invalid
):
    dataset = replace(dataset, questions=dataset.questions[:1])
    config = BenchmarkConfig(strategies=("vector",), cutoffs=(1,), repeats=1)
    prepared = prepare_retrieval(dataset.documents, config, rules, benchmark_provider)
    ids = list(prepared.corpus.chunks)

    def broken(query, k):
        return SearchObservation(
            result=RetrievalResult(
                query="wrong" if invalid == "query" else query,
                strategy="graph" if invalid == "strategy" else "vector",
                elapsed_ms=0.0,
                hits=tuple(
                    RetrievalHit(chunk_id=i, score=1.0)
                    for i in (
                        ids[:2]
                        if invalid == "count"
                        else ["missing" if invalid == "unknown_chunk" else ids[0]]
                    )
                ),
            )
        )

    with pytest.raises(BenchmarkError):
        run_queries(dataset, prepared.corpus, {"vector": broken}, benchmark_provider, config)


def test_cli_runs_and_verifies_without_gold_graph_or_real_neural_dependencies(
    fixture_root, tmp_path, benchmark_provider, monkeypatch, capsys
):
    def forbidden(*args, **kwargs):
        raise AssertionError("must not load oracle fixture graph")

    monkeypatch.setattr("graphrag_bench.cli.load_fixture", forbidden)
    monkeypatch.setattr(
        "graphrag_bench.cli.SentenceTransformerProvider", lambda *a, **k: benchmark_provider
    )
    output = tmp_path / "run"
    args = [
        "benchmark",
        str(fixture_root / "corpus/documents.jsonl"),
        "--questions",
        str(fixture_root / "gold/questions.jsonl"),
        "--split",
        "fixture",
        "--output",
        str(output),
        "--config",
        str(ROOT / "configs/benchmark.toml"),
        "--rules",
        str(ROOT / "configs/extraction.toml"),
        "--embedding-config",
        str(ROOT / "configs/embedding.toml"),
    ]
    assert main(args) == 0
    response = json.loads(capsys.readouterr().out)
    assert response["records"] == 240 and response["repeatable"]
    monkeypatch.setattr("graphrag_bench.cli.SentenceTransformerProvider", forbidden)
    assert main(["verify-benchmark", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "valid"
    assert main(args) == 1
    assert "output already exists" in capsys.readouterr().err
    args[args.index("fixture")] = "test"
    args[args.index(str(output))] = str(tmp_path / "second")
    assert main(args) == 1
    error = capsys.readouterr().err
    assert "no questions" in error and "Traceback" not in error
    assert not (tmp_path / "second").exists()
