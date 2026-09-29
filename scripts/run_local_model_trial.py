"""Run the approved, bounded M11 trial independently of an editor session.

Use a systemd user service to survive closing VS Code. This script never closes
applications, launches a full-corpus job, changes reference labels, or pushes Git.
"""

import argparse
import json
import os
import resource
import subprocess
import sys
from datetime import UTC, datetime
from hashlib import file_digest, sha256
from pathlib import Path
from time import monotonic, sleep

from check_extraction_sample import available_memory_bytes, preflight

from graphrag_bench.extraction.llm.config import load_llm_config

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "datasets/processed/m10-expanded-03/ingestion"
CHECKS = ROOT / "datasets/examples/extraction-study/m11-source-checks.json"
CONFIG = ROOT / "configs/llm-extraction-indexed-3b.toml"
DIAGNOSTIC = ROOT / "scripts/check_extraction_sample.py"
FILES = [
    "LICENSE",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "model.safetensors.index.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
]
WEIGHT_HASHES = {
    "model-00001-of-00002.safetensors": (
        "67347b23fb4165b652eb6611f5e1f2a06dfcddba8e909df1b2b0b1857bee06c2"
    ),
    "model-00002-of-00002.safetensors": (
        "a40d941d0e7e0b966ad8b62bb6d6b7c88cce1299197b599d9d0a4ce59aabfc1d"
    ),
}
MIN_AVAILABLE_BYTES = 18 * 1024**3


def fetch_model(config, allow_download: bool) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=config.model.model_id,
            revision=config.model.revision,
            allow_patterns=FILES,
            max_workers=2,
            local_files_only=not allow_download,
        )
    )


def verify_weights(snapshot: Path) -> None:
    for name, expected in WEIGHT_HASHES.items():
        with (snapshot / name).open("rb") as stream:
            if file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"downloaded weight checksum mismatch: {name}")


def run_trial(output: Path, *, allow_download: bool, wait_seconds: int) -> dict:
    output = output.resolve()
    if output.is_relative_to(SOURCE.resolve()):
        raise ValueError("trial output must be outside source ingestion")
    output.mkdir(parents=True, exist_ok=False)
    status = {"kind": "bounded-local-model-trial-v1", "started_at": datetime.now(UTC).isoformat()}

    def record(state, **details):
        status.update(state=state, updated_at=datetime.now(UTC).isoformat(), **details)
        temp = output / "status.tmp"
        temp.write_text(json.dumps(status, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        temp.replace(output / "status.json")
        print(json.dumps({"state": state, **details}), flush=True)

    try:
        record("validating_source")
        # Preserve the exact recipe and case bytes if the working tree changes while waiting.
        saved_config, saved_checks = output / "config.toml", output / "checks.json"
        saved_config.write_bytes(CONFIG.read_bytes())
        saved_checks.write_bytes(CHECKS.read_bytes())
        config = load_llm_config(saved_config)
        if (
            config.model.model_id != "Qwen/Qwen2.5-3B-Instruct"
            or config.model.revision != "aa8e72537993ba99e69dfaafa59ed015b17504d1"
        ):
            raise ValueError("runner is restricted to the pinned 3B model")
        ready = preflight(SOURCE, saved_checks, config)
        if ready["cases"] != 8:
            raise ValueError("runner is restricted to the eight-case development diagnostic")
        record("loading_model_cache", download_allowed=allow_download, preflight=ready)
        snapshot = fetch_model(config, allow_download)
        record("verifying_weights", model_snapshot=str(snapshot))
        verify_weights(snapshot)
        deadline = monotonic() + wait_seconds
        while True:
            available = available_memory_bytes()
            if available is not None and available >= MIN_AVAILABLE_BYTES:
                break
            record("waiting_for_memory", available_memory_bytes=available)
            if monotonic() >= deadline:
                raise TimeoutError("18 GiB available RAM was not reached within the wait limit")
            sleep(min(10, max(0, deadline - monotonic())))

        report = output / "diagnostic.json"
        command = [sys.executable, str(DIAGNOSTIC), str(SOURCE), "--checks", str(saved_checks)]
        environment = os.environ | {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        }
        record("running_eight_cases", available_memory_bytes=available)
        started = monotonic()
        with (output / "inference.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                command
                + [
                    "--config",
                    str(saved_config),
                    "--min-available-gib",
                    "18",
                    "--response-cache",
                    str(ROOT / "experiments/runs/m11-research-extraction-cache"),
                    "--output",
                    str(report),
                ],
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=3600,
            )
        record(
            "replaying",
            inference_wall_seconds=monotonic() - started,
            inference_peak_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        )
        with (output / "replay.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                command + ["--replay", str(report)],
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=120,
            )
        raw = report.read_bytes()
        record(
            "completed",
            report_sha256=sha256(raw).hexdigest(),
            summary=json.loads(raw)["summary"],
            semantic_review="pending",
        )
        return status
    except BaseException as error:
        record("failed", error=f"{type(error).__name__}: {error}")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="new directory for trial artifacts"
    )
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--wait-minutes", type=int, default=30)
    args = parser.parse_args()
    if args.wait_minutes <= 0:
        parser.error("wait-minutes must be positive")
    run_trial(args.output, allow_download=args.allow_download, wait_seconds=60 * args.wait_minutes)


if __name__ == "__main__":
    main()
