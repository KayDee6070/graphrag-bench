"""Teach fact coverage, alternative evidence, and budget loss without a neural model."""

from pathlib import Path

from graphrag_bench.benchmark.context import assemble_context
from graphrag_bench.benchmark.dataset import load_benchmark
from graphrag_bench.benchmark.metrics import score_evidence
from graphrag_bench.config import ChunkingConfig
from graphrag_bench.corpus import CorpusIndex
from graphrag_bench.extraction import extract, load_rules
from graphrag_bench.graph import build_graph
from graphrag_bench.ingestion.chunker import chunk_document
from graphrag_bench.retrieval.bm25 import BM25Retriever
from graphrag_bench.retrieval.graph import GraphRetriever


class TeachingCharacterCounter:
    """Only for this budget arithmetic example; production uses the pinned model tokenizer."""

    def count_tokens(self, text: str) -> int:
        return len(text)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    fixture = root / "datasets/fixtures/tiny"
    dataset = load_benchmark(
        fixture / "corpus/documents.jsonl", fixture / "gold/questions.jsonl", split="fixture"
    )
    questions = {q.question_id: q for q in dataset.questions}
    question = questions["q09"]
    print("1. Finding one fact is different from finding everything needed.")
    print(f"  {question.question}")
    facts = question.sufficient_evidence_sets[0].facts
    for fact in facts:
        print(f"  Required fact: {fact.statement}")
    partial = score_evidence(question, facts[0].spans)
    complete = score_evidence(question, (span for fact in facts for span in fact.spans))
    assert partial.evidence_coverage == 0.5 and not partial.complete_evidence
    assert complete.evidence_coverage == 1 and complete.complete_evidence
    print("  First fact only: 50% evidence coverage, incomplete evidence.")
    print("  Both facts: 100% evidence coverage, complete evidence.")
    print("  These are retrieval judgments; no answer has been generated or graded.")

    print("\n2. Alternative evidence sets are independent routes to a sufficient answer.")
    question = questions["q19"]
    print(f"  {question.question}")
    for number, evidence_set in enumerate(question.sufficient_evidence_sets, 1):
        spans = tuple(span for fact in evidence_set.facts for span in fact.spans)
        assert score_evidence(question, spans).complete_evidence
        print(f"  Route {number}: {len({s.document_id for s in spans})} document(s); complete.")
    print("  Two reasoning hops do not always imply two required documents.")

    print("\n3. Build retrievers from sources; consult gold only after retrieval.")
    chunks = tuple(
        chunk
        for document in dataset.documents
        for chunk in chunk_document(document, ChunkingConfig(max_units=1, overlap_units=0))
    )
    corpus = CorpusIndex(dataset.documents, chunks)
    extraction = extract(dataset.documents, chunks, load_rules(root / "configs/extraction.toml"))
    graph = GraphRetriever(
        build_graph(extraction.entities, extraction.relations, dataset.documents, chunks),
        dataset.documents,
        chunks,
    )
    lexical = BM25Retriever(dataset.documents, chunks)
    question = questions["q09"]
    for retriever in (lexical, graph):
        result = retriever.retrieve(question.question, top_k=5)
        score = score_evidence(question, (corpus.chunks[hit.chunk_id] for hit in result.hits))
        print(
            f"  {result.strategy} at K=5, before context budget: "
            f"Fact Recall={score.evidence_coverage:.0%}, complete={score.complete_evidence}"
        )
        print(f"    Missing from first sufficient set: {score.sets[0].missing_fact_ids}")

    print("\n4. A retrieved fact can be lost during context selection.")
    result = lexical.retrieve(questions["q01"].question, top_k=5)
    raw = score_evidence(questions["q01"], (corpus.chunks[h.chunk_id] for h in result.hits))
    assert raw.complete_evidence
    context = assemble_context(result, corpus, TeachingCharacterCounter(), max_tokens=1)
    after = score_evidence(questions["q01"], context.pieces)
    assert not after.complete_evidence and context.token_count == 0
    print("  Teaching-only budget: one character, with source headers included. Nothing fits.")
    print("  Retrieval found the fact; the budgeted context contains none of it.")
    print("  Actual benchmark runs use 2,000 pinned model-tokenizer tokens, not characters.")
    print("\n5. Twenty questions run three times are still twenty questions, not sixty.")
    print("  Repeats measure ranking stability and warm query timings.")
    print("  All twenty remain fixture examples, not an unseen test set.")
    print("  Use the benchmark CLI in docs/benchmark.md for all four methods and saved results.")


if __name__ == "__main__":
    main()
