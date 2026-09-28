import builtins
import json
import shutil
from hashlib import sha256
from io import BytesIO
from urllib.error import URLError

import pytest
from pydantic import ValidationError

from graphrag_bench.benchmark.dataset import load_benchmark
from graphrag_bench.cli import main
from graphrag_bench.config import ChunkingConfig
from graphrag_bench.ingestion.reader import load_ingestion
from graphrag_bench.ingestion.types import IngestionError
from graphrag_bench.papers import acquire, pdf
from graphrag_bench.papers.acquire import checked_pdf, fetch_papers
from graphrag_bench.papers.catalog import PaperCatalog, PaperError, load_catalog
from graphrag_bench.papers.pipeline import BUNDLE_FILES, prepare_papers, verify_papers


def prepare(inputs, output):
    return prepare_papers(*inputs[:3], output)


def mutate(path, change):
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))


def rehash_bundle(output, name):
    mutate(
        output / "manifest.json",
        lambda data: data["artifact_hashes"].update(
            {name: sha256((output / name).read_bytes()).hexdigest()}
        ),
    )


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d["papers"][0].update(arxiv_version="2401.12345"),
        lambda d: d["papers"][0].update(arxiv_version="../escape"),
        lambda d: d["papers"][0].update(license_id="assumed-open"),
        lambda d: d["papers"][0].update(pdf_bytes=16_000_001),
        lambda d: d["papers"].append(d["papers"][0]),
        lambda d: d["papers"][1].update(arxiv_version="2401.12345v9"),
    ],
)
def test_catalog_rejects_unpinned_ambiguous_or_unbounded_sources(paper_inputs, change):
    path = paper_inputs[0]
    mutate(path, change)
    with pytest.raises(PaperError):
        load_catalog(path)


def test_catalog_total_download_budget(paper_inputs):
    data = paper_inputs[3].model_dump()
    for p in data["papers"]:
        p["pdf_bytes"] = 16_000_000
    extra = {**data["papers"][0], "paper_id": "third", "arxiv_version": "2403.12345v1"}
    data["papers"] += (extra,)
    with pytest.raises(ValidationError, match="exceeds 32 MB"):
        PaperCatalog.model_validate(data)


def test_existing_pdfs_are_reused_offline(paper_inputs, monkeypatch):
    monkeypatch.setattr(acquire, "urlopen", lambda *a, **k: pytest.fail("network forbidden"))
    assert fetch_papers(paper_inputs[3], paper_inputs[2]) == {"downloaded": 0, "reused": 2}


def test_download_is_bounded_and_pinned(paper_inputs, tmp_path, monkeypatch):
    catalog = paper_inputs[3]
    payloads = {p.pdf_url: (paper_inputs[2] / p.filename).read_bytes() for p in catalog.papers}
    reads = []

    class Response(BytesIO):
        def read(self, size=-1):
            reads.append(size)
            return super().read(size)

    monkeypatch.setattr(acquire, "urlopen", lambda req, **kw: Response(payloads[req.full_url]))
    sleeps = []
    monkeypatch.setattr(acquire.time, "sleep", sleeps.append)
    output = tmp_path / "downloaded"
    assert fetch_papers(catalog, output) == {"downloaded": 2, "reused": 0}
    assert reads == [p.pdf_bytes + 1 for p in catalog.papers]
    assert sleeps == [3]
    assert sorted(p.name for p in output.iterdir()) == sorted(p.filename for p in catalog.papers)
    for paper in catalog.papers:
        checked_pdf(output / paper.filename, paper)


@pytest.mark.parametrize("bad", [b"<html>server error</html>", b"%PDF-oversized" * 1000])
def test_bad_download_leaves_no_pdf_or_temporary_file(paper_inputs, tmp_path, monkeypatch, bad):
    monkeypatch.setattr(acquire, "urlopen", lambda *a, **k: BytesIO(bad))
    output = tmp_path / "downloaded"
    with pytest.raises(PaperError, match="checksum/size"):
        fetch_papers(paper_inputs[3], output)
    assert list(output.iterdir()) == []


def test_http_failure_is_a_domain_error(paper_inputs, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise URLError("offline")

    monkeypatch.setattr(acquire, "urlopen", fail)
    with pytest.raises(PaperError, match="offline"):
        fetch_papers(paper_inputs[3], tmp_path / "downloaded")


def test_bad_cache_fails_before_downloading_missing_entry(paper_inputs, monkeypatch):
    catalog, raw = paper_inputs[3], paper_inputs[2]
    (raw / catalog.papers[0].filename).unlink()
    bad = raw / catalog.papers[1].filename
    bad.write_bytes(b"keep this corrupted evidence for inspection")
    monkeypatch.setattr(acquire, "urlopen", lambda *a, **k: pytest.fail("no download expected"))
    with pytest.raises(PaperError, match="size mismatch"):
        fetch_papers(catalog, raw)
    assert bad.read_bytes() == b"keep this corrupted evidence for inspection"


def test_pdf_pages_blank_pages_and_exact_character_offsets(paper_inputs):
    paper = paper_inputs[3].papers[0]
    parsed = pdf.parse_pdf((paper_inputs[2] / paper.filename).read_bytes(), paper)
    assert len(parsed.source.sections) == 3
    assert parsed.document.text.count(pdf.PAGE_SEPARATOR) == 3
    chunks = pdf.chunk_pdf(parsed, ChunkingConfig())
    assert [chunk.page for chunk in chunks] == [1, 3]
    assert all(parsed.document.text[c.start : c.end] == c.text for c in chunks)
    assert parsed.document.source_uri == "https://arxiv.org/abs/2401.12345v1"


def test_unicode_pages_use_character_not_byte_offsets(paper_inputs, monkeypatch):
    monkeypatch.setattr(pdf, "extract_pages", lambda raw: ("α Scout.", "Gate east."))
    paper = paper_inputs[3].papers[0]
    parsed = pdf.parse_pdf((paper_inputs[2] / paper.filename).read_bytes(), paper)
    chunks = pdf.chunk_pdf(parsed, ChunkingConfig())
    assert chunks[1].start == len("α Scout." + pdf.PAGE_SEPARATOR)
    assert chunks[1].start != len(("α Scout." + pdf.PAGE_SEPARATOR).encode("utf-8"))


def test_empty_extraction_does_not_silently_drop_a_paper(paper_inputs, monkeypatch):
    monkeypatch.setattr(pdf, "extract_pages", lambda raw: ("", " "))
    paper = paper_inputs[3].papers[0]
    with pytest.raises(PaperError, match="no extractable text"):
        pdf.parse_pdf((paper_inputs[2] / paper.filename).read_bytes(), paper)


def test_parser_version_is_pinned(paper_inputs, monkeypatch):
    import pypdf

    monkeypatch.setattr(pypdf, "__version__", "future-version")
    with pytest.raises(PaperError, match="requires pypdf=="):
        pdf.extract_pages(b"unused")


def test_prepare_round_trip_and_labels_survive_rechunking(paper_inputs, tmp_path, monkeypatch):
    monkeypatch.setattr(acquire, "urlopen", lambda *a, **k: pytest.fail("network forbidden"))
    output = tmp_path / "bundle"
    manifest = prepare(paper_inputs, output)
    assert (manifest.document_count, manifest.page_count, manifest.question_count) == (2, 4, 1)
    assert verify_papers(output, raw_directory=paper_inputs[2]) == manifest
    dataset = load_benchmark(
        output / "ingestion/documents.jsonl", output / "questions.jsonl", split="dev"
    )
    question = dataset.questions[0]
    assert question.required_document_count == 2
    assert question.reasoning_hops == 2
    assert all(
        s.chunk_id is None
        for e in question.sufficient_evidence_sets
        for f in e.facts
        for s in f.spans
    )
    other = tmp_path / "different-chunks"
    prepare_papers(
        *paper_inputs[:3], other, ChunkingConfig(max_chars=15, max_units=1, overlap_units=0)
    )
    assert (other / "questions.jsonl").read_bytes() == (output / "questions.jsonl").read_bytes()
    assert (other / "ingestion/chunks.jsonl").read_bytes() != (
        output / "ingestion/chunks.jsonl"
    ).read_bytes()


def test_relocation_reproduces_every_artifact(paper_inputs, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    prepare(paper_inputs, first)
    moved = tmp_path / "moved-raw"
    shutil.copytree(paper_inputs[2], moved)
    prepare_papers(*paper_inputs[:2], moved, second)
    for name in BUNDLE_FILES | {"manifest.json"}:
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_answer_labels_cannot_change_source_corpus(paper_inputs, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    prepare(paper_inputs, first)
    mutate(
        paper_inputs[1],
        lambda data: data["questions"][0].update(expected_answer="Deliberately wrong."),
    )
    prepare(paper_inputs, second)
    for name in ("documents.jsonl", "chunks.jsonl", "manifest.json"):
        assert (first / "ingestion" / name).read_bytes() == (
            second / "ingestion" / name
        ).read_bytes()
    assert (first / "questions.jsonl").read_bytes() != (second / "questions.jsonl").read_bytes()


@pytest.mark.parametrize(
    "locator",
    [
        {"paper_id": "missing", "page": 1, "quote": "Eren"},
        {"paper_id": "scout-a", "page": 9, "quote": "Eren"},
        {"paper_id": "scout-a", "page": 1, "quote": "Invented quote"},
        {"paper_id": "scout-a", "page": 3, "quote": "Repeated."},
    ],
)
def test_bad_or_ambiguous_annotations_fail_before_writing(paper_inputs, tmp_path, locator):
    mutate(
        paper_inputs[1],
        lambda data: data["questions"][0]["sufficient_evidence_sets"][0][0].update(spans=[locator]),
    )
    output = tmp_path / "invalid"
    with pytest.raises(PaperError):
        prepare(paper_inputs, output)
    assert not output.exists()


def test_pilot_cannot_masquerade_as_held_out_labels(paper_inputs, tmp_path):
    mutate(paper_inputs[1], lambda data: data.update(split="test"))
    with pytest.raises(PaperError, match="dev"):
        prepare(paper_inputs, tmp_path / "test")


@pytest.mark.parametrize("field", ["parser_version", "chunker_version"])
def test_unrecognized_pdf_implementation_versions_are_rejected(paper_inputs, tmp_path, field):
    output = tmp_path / "bundle"
    prepare(paper_inputs, output)
    mutate(output / "ingestion/manifest.json", lambda data: data.update({field: "unknown"}))
    rehash_bundle(output, "ingestion/manifest.json")
    with pytest.raises(PaperError, match="unsupported PDF"):
        verify_papers(output)


def test_manifest_cannot_request_arbitrary_files(paper_inputs, tmp_path):
    output = tmp_path / "bundle"
    prepare(paper_inputs, output)
    mutate(
        output / "manifest.json",
        lambda data: data["artifact_hashes"].update({"../outside.json": "0" * 64}),
    )
    with pytest.raises(PaperError, match="unexpected artifact names"):
        verify_papers(output)


def test_source_catalog_provenance_is_rechecked(paper_inputs, tmp_path):
    output = tmp_path / "bundle"
    prepare(paper_inputs, output)
    mutate(output / "catalog.json", lambda data: data["papers"][0].update(pdf_sha256="0" * 64))
    rehash_bundle(output, "catalog.json")
    with pytest.raises(PaperError, match="source provenance mismatch"):
        verify_papers(output)


@pytest.mark.parametrize("name", sorted(BUNDLE_FILES))
def test_all_bundle_artifacts_are_integrity_checked(paper_inputs, tmp_path, name):
    output = tmp_path / "bundle"
    prepare(paper_inputs, output)
    with (output / name).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(PaperError, match="checksum mismatch"):
        verify_papers(output)


def test_rehashed_compiled_labels_still_require_annotation_match(paper_inputs, tmp_path):
    output = tmp_path / "bundle"
    prepare(paper_inputs, output)
    mutate(output / "questions.jsonl", lambda data: data.update(expected_answer="West gate"))
    rehash_bundle(output, "questions.jsonl")
    with pytest.raises(PaperError, match="compiled paper annotations mismatch"):
        verify_papers(output)


def test_bad_page_number_is_rejected_by_existing_ingestion_reader(paper_inputs, tmp_path):
    output = tmp_path / "bundle"
    prepare(paper_inputs, output)
    chunk_path = output / "ingestion/chunks.jsonl"
    chunks = [json.loads(line) for line in chunk_path.read_text().splitlines()]
    chunks[0]["page"] = 2
    chunk_path.write_text("".join(json.dumps(c) + "\n" for c in chunks))
    mutate(
        output / "ingestion/manifest.json",
        lambda data: data["artifact_hashes"].update(
            {"chunks.jsonl": sha256(chunk_path.read_bytes()).hexdigest()}
        ),
    )
    with pytest.raises(IngestionError, match="page boundary"):
        load_ingestion(output / "ingestion")


def test_verification_without_raw_needs_no_pdf_or_neural_library(
    paper_inputs, tmp_path, monkeypatch
):
    output = tmp_path / "bundle"
    prepare(paper_inputs, output)
    original = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.split(".")[0] in {"pypdf", "torch", "transformers", "sentence_transformers"}:
            raise ImportError(f"blocked dependency {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    verify_papers(output)
    with pytest.raises(PaperError, match="requires graphrag-bench"):
        verify_papers(output, raw_directory=paper_inputs[2])


def test_outputs_are_fresh_and_outside_raw(paper_inputs, tmp_path):
    with pytest.raises(PaperError, match="already exists"):
        prepare(paper_inputs, tmp_path)
    with pytest.raises(PaperError, match="outside the raw"):
        prepare(paper_inputs, paper_inputs[2] / "output")


def test_cli_prepares_verifies_and_reports_pending_review(paper_inputs, tmp_path, capsys):
    output = tmp_path / "bundle"
    assert (
        main(
            [
                "prepare-papers",
                "--catalog",
                str(paper_inputs[0]),
                "--annotations",
                str(paper_inputs[1]),
                "--raw",
                str(paper_inputs[2]),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["review_status"] == "pending-independent-review"
    assert main(["verify-papers", str(output), "--raw", str(paper_inputs[2])]) == 0
    assert json.loads(capsys.readouterr().out)["raw_verified"] is True
    assert (
        main(
            [
                "prepare-papers",
                "--catalog",
                str(paper_inputs[0]),
                "--annotations",
                str(paper_inputs[1]),
                "--raw",
                str(paper_inputs[2]),
                "--output",
                str(output),
            ]
        )
        == 1
    )
    assert "already exists" in capsys.readouterr().err
