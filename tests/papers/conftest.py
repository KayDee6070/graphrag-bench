import json
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from graphrag_bench.benchmark.config import BenchmarkConfig
from graphrag_bench.config import ChunkingConfig
from graphrag_bench.extraction.llm.config import LLMExtractionConfig, RelationType, load_llm_config
from graphrag_bench.extraction.llm.contracts import Completion, ModelSpec
from graphrag_bench.extraction.llm.pipeline import build_llm_graph_to_directory
from graphrag_bench.papers.catalog import PaperCatalog
from graphrag_bench.papers.pipeline import prepare_papers
from graphrag_bench.retrieval.artifacts import build_vector_to_directory

ROOT = Path(__file__).resolve().parents[2]


def make_pdf(pages):
    """Original tiny PDF sources; no paper downloads, model weights, or binary fixtures."""
    writer = PdfWriter()
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    for text in pages:
        page = writer.add_blank_page(width=600, height=800)
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}
        )
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 50 750 Td ({text}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


@pytest.fixture
def paper_inputs(tmp_path):
    raw_directory = tmp_path / "raw"
    raw_directory.mkdir()
    papers = []
    for paper_id, version, pages in [
        ("scout-a", "2401.12345v1", ("Eren belongs to Lantern Squad.", "", "Repeated. Repeated.")),
        ("scout-b", "2402.12345v2", ("Lantern Squad guards the east gate.",)),
    ]:
        raw = make_pdf(pages)
        (raw_directory / f"{version}.pdf").write_bytes(raw)
        papers.append(
            {
                "paper_id": paper_id,
                "arxiv_version": version,
                "title": paper_id,
                "authors": ["Original test author"],
                "license_id": "CC-BY-4.0",
                "metadata_checked_on": "2026-09-28",
                "pdf_sha256": sha256(raw).hexdigest(),
                "pdf_bytes": len(raw),
            }
        )
    catalog = {"catalog_id": "scout-test", "papers": papers}
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog))
    annotation = {
        "catalog_id": "scout-test",
        "questions": [
            {
                "question_id": "gate",
                "question": "Which gate does Eren's squad guard?",
                "question_type": "relationship",
                "expected_answer": "The east gate.",
                "reasoning_hops": 2,
                "difficulty": "medium",
                "group_id": "scouts",
                "rationale": "Both original fictional reports are needed for this bridge.",
                "sufficient_evidence_sets": [
                    [
                        {
                            "fact_id": "membership",
                            "statement": "Eren belongs to Lantern Squad.",
                            "spans": [
                                {
                                    "paper_id": "scout-a",
                                    "page": 1,
                                    "quote": "Eren belongs to Lantern Squad.",
                                }
                            ],
                        },
                        {
                            "fact_id": "assignment",
                            "statement": "Lantern Squad guards the east gate.",
                            "spans": [
                                {
                                    "paper_id": "scout-b",
                                    "page": 1,
                                    "quote": "Lantern Squad guards the east gate.",
                                }
                            ],
                        },
                    ]
                ],
            }
        ],
    }
    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(json.dumps(annotation))
    return (catalog_path, annotation_path, raw_directory, PaperCatalog.model_validate(catalog))


@pytest.fixture
def frozen_inputs(paper_inputs, tmp_path, embedding_provider):
    bundle, index, graph = (tmp_path / name for name in ("bundle", "index", "graph"))
    chunking = ChunkingConfig(max_chars=512, max_units=5, overlap_units=1)
    prepare_papers(*paper_inputs[:3], bundle, chunking)
    build_vector_to_directory(bundle / "ingestion", index, embedding_provider)
    extraction = LLMExtractionConfig(
        model=load_llm_config(ROOT / "configs/llm-extraction.toml").model,
        entity_types={"Person": "A scout.", "Squad": "A squad.", "Place": "A location."},
        relations=(
            RelationType(
                predicate="BELONGS_TO",
                description="Membership.",
                subject_types=("Person",),
                object_types=("Squad",),
            ),
            RelationType(
                predicate="GUARDS",
                description="Assignment.",
                subject_types=("Squad",),
                object_types=("Place",),
            ),
        ),
    )

    class ScriptedGraphProvider:
        spec = ModelSpec(
            provider="test-only",
            model_id=extraction.model.model_id,
            revision=extraction.model.revision,
            settings=extraction.model.model_dump(mode="json"),
            versions={"test": "1"},
        )

        def complete(self, request):
            proposal = {"entities": [], "relations": []}
            for quote, subject, stype, predicate, target, ttype in (
                (
                    "Eren belongs to Lantern Squad.",
                    "Eren",
                    "Person",
                    "BELONGS_TO",
                    "Lantern Squad",
                    "Squad",
                ),
                (
                    "Lantern Squad guards the east gate.",
                    "Lantern Squad",
                    "Squad",
                    "GUARDS",
                    "east gate",
                    "Place",
                ),
            ):
                if quote in request.chunk.text:
                    proposal = {
                        "entities": [
                            {"id": "s", "name": subject, "entity_type": stype},
                            {"id": "o", "name": target, "entity_type": ttype},
                        ],
                        "relations": [
                            {"subject": "s", "predicate": predicate, "object": "o", "quote": quote}
                        ],
                    }
            return Completion(
                text=json.dumps(proposal),
                input_tokens=1,
                output_tokens=1,
                finish_reason="eos",
                elapsed_ms=0,
            )

    build_llm_graph_to_directory(bundle / "ingestion", graph, extraction, ScriptedGraphProvider())
    embedding_provider.count_tokens = lambda text: len(text.split())
    config = BenchmarkConfig(chunking=chunking, cutoffs=(1, 3), repeats=2)
    return bundle, index, graph, config, embedding_provider
