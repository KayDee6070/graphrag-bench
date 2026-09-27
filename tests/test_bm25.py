import math

import pytest
from pydantic import ValidationError

from graphrag_bench.cli import main
from graphrag_bench.ingestion.chunker import chunk_document
from graphrag_bench.ingestion.parser import parse_bytes
from graphrag_bench.ingestion.pipeline import ingest_to_directory
from graphrag_bench.retrieval.bm25 import BM25Config, BM25Retriever, tokenize


def retriever(texts, config=None):
    docs = tuple(
        parse_bytes(text.encode(), relative_path=f"{i}.txt").document
        for i, text in enumerate(texts)
    )
    chunks = tuple(c for doc in docs for c in chunk_document(doc))
    return BM25Retriever(docs, chunks, config)


def test_hand_calculated_bm25_score_and_length_normalization():
    search = retriever(("apple apple banana", "banana"))
    result = search.retrieve("apple")
    assert len(result.hits) == 1
    # N=2, df=1, tf=2, document length=3, average length=2, k1=1.5, b=.75.
    expected = math.log(2) * (2 * 2.5) / (2 + 1.5 * (0.25 + 0.75 * 3 / 2))
    assert result.hits[0].score == pytest.approx(expected)
    assert search.retrieve("apple apple").hits == result.hits


def test_no_overlap_and_empty_corpus_return_no_hits():
    assert retriever(("apple",)).retrieve("orange").hits == ()
    assert retriever(("!!!",)).retrieve("apple").hits == ()
    assert retriever(()).retrieve("apple").hits == ()


def test_unicode_casefold_tokenization_and_stable_ties():
    assert tokenize("CAFÉ, Straße!") == ("café", "strasse")
    search = retriever(("Same words.", "Same words."))
    hits = search.retrieve("SAME", top_k=99).hits
    assert len(hits) == 2
    assert [h.chunk_id for h in hits] == sorted(h.chunk_id for h in hits)


@pytest.mark.parametrize("values", [{"k1": 0}, {"k1": float("nan")}, {"b": -0.1}, {"b": 1.1}])
def test_invalid_parameters_rejected(values):
    with pytest.raises(ValidationError):
        BM25Config(**values)


def test_cli_bm25_is_available_without_model(tmp_path, monkeypatch, capsys):
    def no_model(*args, **kwargs):
        raise AssertionError("BM25 must not initialize embeddings")

    monkeypatch.setattr("graphrag_bench.cli.SentenceTransformerProvider", no_model)
    source = tmp_path / "source"
    source.mkdir()
    (source / "data.txt").write_text("Apples are fruit.")
    output = tmp_path / "input"
    ingest_to_directory(source, output)
    assert main(["query-bm25", str(output), "--query", "apples"]) == 0
    assert '"strategy": "bm25"' in capsys.readouterr().out
