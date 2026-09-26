"""A small fixture validation command; experiment commands arrive later."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from graphrag_bench import __version__
from graphrag_bench.fixtures import FixtureError, load_fixture


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GraphRAG Bench development tools")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-fixtures", help="audit a fixture directory")
    validate.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    try:
        fixture = load_fixture(args.path)
    except FixtureError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    summary = {
        "documents": len(fixture.documents),
        "chunks": len(fixture.chunks),
        "entities": len(fixture.entities),
        "relations": len(fixture.relations),
        "questions": len(fixture.questions),
        "status": "valid",
    }
    print(json.dumps(summary, sort_keys=True))
    return 0
