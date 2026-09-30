"""Export or verify a pending, source-bound M11 extraction review packet."""

import argparse
import json
from pathlib import Path

from graphrag_bench.extraction.llm.review import export_review_packet, verify_review_packet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--checks", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--review", type=Path)
    args = parser.parse_args()
    if bool(args.output) == bool(args.review):
        parser.error("choose exactly one of --output or --review")
    if args.output:
        packet = export_review_packet(args.source, args.checks, args.config, args.output)
        print(json.dumps({"cases": len(packet.cases), "status": "pending-review"}))
    else:
        print(json.dumps(verify_review_packet(args.source, args.checks, args.config, args.review)))


if __name__ == "__main__":
    main()
