"""Inspect model-proposal validation using explicitly handwritten responses, offline."""

import copy
import json
from pathlib import Path

from graphrag_bench.corpus import CorpusIndex
from graphrag_bench.extraction.llm.config import load_llm_config
from graphrag_bench.extraction.llm.contracts import Completion, ModelSpec, ResponseRecord
from graphrag_bench.extraction.llm.extractor import process_responses
from graphrag_bench.extraction.llm.prompt import make_request, request_fingerprint
from graphrag_bench.models import Chunk, Document, text_sha256


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_llm_config(root / "configs/llm-extraction.toml")
    config = config.model_copy(
        update={
            "model": config.model.model_copy(
                update={
                    "model_id": "teaching/no-model",
                    "revision": "0" * 40,
                }
            ),
        }
    )
    spec = ModelSpec(
        provider="handwritten-study-no-inference",
        model_id=config.model.model_id,
        revision=config.model.revision,
        settings=config.model.model_dump(mode="json"),
        versions={"lesson": "1"},
    )
    text = "Orion uses Nova."
    document = Document(
        document_id="lesson",
        title="Source report",
        text=text,
        content_sha256=text_sha256(text),
        source_uri="original:study-m8",
    )
    chunk = Chunk(
        document_id="lesson", chunk_id="lesson-chunk", ordinal=0, start=0, end=len(text), text=text
    )
    corpus = CorpusIndex((document,), (chunk,))
    request = make_request(chunk, config)
    good = {
        "entities": [
            {"id": "e1", "name": "Orion", "entity_type": "Model"},
            {"id": "e2", "name": "Nova", "entity_type": "Model"},
        ],
        "relations": [{"subject": "e1", "predicate": "USES", "object": "e2", "quote": text}],
    }
    invented = copy.deepcopy(good)
    invented["relations"][0]["quote"] = "Orion is based on Nova."
    reversed_claim = copy.deepcopy(good)
    reversed_claim["relations"][0].update(subject="e2", object="e1")
    cases = (
        ("Exact quote and allowed relation", json.dumps(good), 1),
        ("Invented quote", json.dumps(invented), 0),
        ("Malformed output", "Here is the answer!", 0),
        ("Reversed meaning with a real quote", json.dumps(reversed_claim), 1),
    )
    print("M8 lesson: HANDWRITTEN RESPONSES. No model is loaded or evaluated.")
    print(f"Source: {text}\n")
    for label, raw, expected in cases:
        record = ResponseRecord(
            request_sha256=request_fingerprint(spec, request),
            request=request,
            completion=Completion(
                text=raw, input_tokens=0, output_tokens=0, finish_reason="eos", elapsed_ms=0.0
            ),
        )
        result = process_responses(corpus, (record,), config, spec)
        assert len(result.relations) == expected
        print(
            f"{label}: {len(result.relations)} source-checked assertions; "
            f"issues={[i.code for i in result.issues]}"
        )
        if result.relations:
            names = {e.entity_id: e.name for e in result.entities}
            relation = result.relations[0]
            print(
                f"  {names[relation.subject_id]} {relation.predicate} {names[relation.object_id]}"
            )
            print(f"  review_status={relation.qualifiers['review_status']}")
    print("\nA genuine quote proves the source location, not the relationship's meaning.")
    print(
        "Reviewers must check direction, negation, type, and whether the quote supports the claim."
    )
    print("Replay repeats validation of saved proposals; it does not ask the model again.")


if __name__ == "__main__":
    main()
