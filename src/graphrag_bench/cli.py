"""Offline fixture validation, document ingestion, and graph construction commands."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from graphrag_bench import __version__
from graphrag_bench.config import ConfigurationError, load_chunking_config
from graphrag_bench.corpus import CorpusError
from graphrag_bench.extraction.config import ExtractionError
from graphrag_bench.fixtures import FixtureError, load_fixture
from graphrag_bench.graph.builder import GraphError
from graphrag_bench.graph.pipeline import build_graph_to_directory
from graphrag_bench.ingestion.pipeline import ingest_to_directory
from graphrag_bench.ingestion.types import IngestionError


def _ingest(args: argparse.Namespace) -> dict[str, str | int]:
    overrides = {
        name: getattr(args, name)
        for name in ("max_chars", "max_units", "overlap_units")
        if getattr(args, name) is not None
    }
    manifest = ingest_to_directory(
        args.source,
        args.output,
        load_chunking_config(args.config, overrides=overrides),
    )
    return {
        "status": "ingested",
        "documents": manifest.document_count,
        "chunks": manifest.chunk_count,
        "ignored_files": len(manifest.ignored_files),
        "output": str(args.output),
    }


def _validate(path: Path) -> dict[str, str | int]:
    fixture = load_fixture(path)
    return {
        "documents": len(fixture.documents),
        "chunks": len(fixture.chunks),
        "entities": len(fixture.entities),
        "relations": len(fixture.relations),
        "questions": len(fixture.questions),
        "status": "valid",
    }


def _build_graph(args: argparse.Namespace) -> dict[str, str | int]:
    manifest = build_graph_to_directory(args.source, args.output, args.rules)
    return {
        "status": "built",
        "entities": manifest.entity_count,
        "assertions": manifest.assertion_count,
        "issues": manifest.issue_count,
        "output": str(args.output),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GraphRAG Bench development tools")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-fixtures", help="audit a fixture directory")
    validate.add_argument("path", type=Path)
    ingest = commands.add_parser(
        "ingest", help="parse and chunk a directory of UTF-8 text/Markdown"
    )
    ingest.add_argument("source", type=Path)
    ingest.add_argument("--output", type=Path, required=True, help="new output directory")
    ingest.add_argument("--config", type=Path, help="TOML file with a [chunking] table")
    ingest.add_argument("--max-chars", type=int, help="hard Unicode character limit per chunk")
    ingest.add_argument("--max-units", type=int, help="maximum sentence/fragment units per chunk")
    ingest.add_argument(
        "--overlap-units", type=int, help="requested overlap between adjacent windows"
    )
    graph = commands.add_parser("build-graph", help="extract a graph from M2 ingestion artifacts")
    graph.add_argument("source", type=Path, help="ingestion artifact directory")
    graph.add_argument("--output", type=Path, required=True, help="new output directory")
    graph.add_argument("--rules", type=Path, required=True, help="TOML extraction grammar")
    args = parser.parse_args(argv)
    try:
        if args.command == "ingest":
            summary = _ingest(args)
        elif args.command == "build-graph":
            summary = _build_graph(args)
        else:
            summary = _validate(args.path)
    except (
        FixtureError,
        ConfigurationError,
        IngestionError,
        ValidationError,
        CorpusError,
        ExtractionError,
        GraphError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 0
