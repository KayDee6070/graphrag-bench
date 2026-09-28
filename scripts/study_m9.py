"""An offline Scout lesson: answer proposals, missing reports, and citation checks."""

import json
from pathlib import Path

from graphrag_bench.corpus import CorpusIndex
from graphrag_bench.extraction.llm.contracts import Completion, ModelSpec
from graphrag_bench.generation.config import load_generation_config
from graphrag_bench.generation.contracts import RetrievalReceipt
from graphrag_bench.generation.generator import generate_answer
from graphrag_bench.models import Chunk, Document, RetrievalHit, RetrievalResult, text_sha256


def main() -> None:
    config = load_generation_config(Path(__file__).resolve().parents[1] / "configs/generation.toml")
    config = config.model_copy(
        update={
            "model": config.model.model_copy(
                update={"model_id": "teaching/no-model", "revision": "0" * 40}
            )
        }
    )
    texts = ("Eren belongs to Lantern Squad.", "Lantern Squad guards the east gate.")
    documents = tuple(
        Document(
            document_id=f"report-{i}",
            title=f"Report {i}",
            text=text,
            content_sha256=text_sha256(text),
            source_uri=f"original:report-{i}",
        )
        for i, text in enumerate(texts, start=1)
    )
    chunks = tuple(
        Chunk(
            document_id=d.document_id,
            chunk_id=f"chunk-{i}",
            ordinal=0,
            start=0,
            end=len(d.text),
            text=d.text,
        )
        for i, d in enumerate(documents, start=1)
    )
    corpus = CorpusIndex(documents, chunks)

    class HandwrittenProvider:
        def __init__(self, response):
            self.response = response
            self.calls = 0
            self.spec = ModelSpec(
                provider="handwritten-lesson-no-inference",
                model_id=config.model.model_id,
                revision=config.model.revision,
                settings=config.model.model_dump(mode="json"),
                versions={"lesson": "1"},
            )

        def count_tokens(self, text):
            # Deliberately a character counter for teaching, not the real model tokenizer.
            return len(text)

        def complete(self, request):
            self.calls += 1
            return Completion(
                text=json.dumps(self.response),
                input_tokens=0,
                output_tokens=0,
                finish_reason="eos",
                elapsed_ms=0,
            )

    def proposed(text, citations):
        return {"status": "answered", "claims": [{"text": text, "citations": citations}]}

    both = [{"source_id": f"S{i}", "quote": text} for i, text in enumerate(texts, start=1)]
    cases = (
        (
            "Both reports, both citations",
            2,
            proposed("Eren's squad guards the east gate.", both),
            "answered",
        ),
        (
            "Report B missing: handwritten abstention",
            1,
            {"status": "insufficient_evidence", "claims": []},
            "insufficient_evidence",
        ),
        (
            "Invented source S99",
            2,
            proposed("East gate.", [{"source_id": "S99", "quote": texts[1]}]),
            "invalid_output",
        ),
        (
            "Invented quote",
            2,
            proposed(
                "West gate.", [{"source_id": "S2", "quote": "Lantern Squad guards the west gate."}]
            ),
            "invalid_output",
        ),
        (
            "Wrong answer with real quotes",
            2,
            proposed("Eren's squad guards the west gate.", both),
            "answered",
        ),
        (
            "Only final fact cited: incomplete chain",
            2,
            proposed("Eren's squad guards the east gate.", both[1:]),
            "answered",
        ),
        (
            "No reports: software abstains",
            0,
            proposed("This response must never be used.", both),
            "insufficient_evidence",
        ),
    )
    print("M9 SCOUT LESSON — handwritten responses, no model inference.")
    print("Question: Which gate does Eren's squad guard?")
    for i, text in enumerate(texts, start=1):
        print(f"Report {i}: {text}")
    for label, count, response, expected in cases:
        retrieval = RetrievalReceipt(
            config=config.retrieval,
            result=RetrievalResult(
                query="Which gate does Eren's squad guard?",
                strategy="graph",
                hits=tuple(RetrievalHit(chunk_id=c.chunk_id, score=1.0) for c in chunks[:count]),
                elapsed_ms=0.0,
            ),
        )
        provider = HandwrittenProvider(response)
        _, _, answer = generate_answer(retrieval, corpus, config, provider)
        assert answer.status == expected
        assert provider.calls == (1 if count else 0)
        print(f"\n{label}: {answer.status}")
        for claim in answer.claims:
            print("  " + claim.text)
            print("  Citations: " + ", ".join(c.source_id for c in claim.citations))
        if answer.issues:
            print("  Rejected: " + answer.issues[0].message)
    print("\nA citation check asks: does this receipt exist in the shown report?")
    print("A meaning check asks: does the receipt support the claim and the complete connection?")
    print("M9 checks receipts. Accepted answers remain unreviewed proposals.")
    print("The missing-report response above was handwritten; it does not prove model abstention.")


if __name__ == "__main__":
    main()
