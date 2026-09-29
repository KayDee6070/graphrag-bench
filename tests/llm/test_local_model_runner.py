import json
import runpy
import subprocess
from hashlib import sha256
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    scope = runpy.run_path(str(ROOT / "scripts/run_local_model_trial.py"))["run_trial"].__globals__
    monkeypatch.setitem(scope, "preflight", lambda *args: {"cases": 8})
    monkeypatch.setitem(scope, "fetch_model", lambda *args: ROOT)
    monkeypatch.setitem(scope, "verify_weights", lambda *args: None)
    return scope


def test_wait_then_run_and_replay_with_frozen_inputs(runner, monkeypatch, tmp_path):
    output = tmp_path / "trial"
    available = iter([1024**3, 18 * 1024**3])
    monkeypatch.setitem(runner, "available_memory_bytes", lambda: next(available))
    waits, calls = [], []
    monkeypatch.setitem(runner, "sleep", waits.append)

    def run(command, **kwargs):
        calls.append(command)
        assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        assert kwargs["env"]["TRANSFORMERS_OFFLINE"] == "1"
        assert kwargs["check"] is True
        assert command[command.index("--checks") + 1] == str(output / "checks.json")
        if "--output" in command:
            assert command[command.index("--config") + 1] == str(output / "config.toml")
            assert command[command.index("--min-available-gib") + 1] == "18"
            Path(command[command.index("--output") + 1]).write_text('{"summary":{"cases":8}}')
        else:
            assert command[-2:] == ["--replay", str(output / "diagnostic.json")]

    monkeypatch.setattr(runner["subprocess"], "run", run)
    status = runner["run_trial"](output, allow_download=False, wait_seconds=30)
    assert waits and len(calls) == 2
    assert status["state"] == "completed" and status["semantic_review"] == "pending"
    assert status["report_sha256"] == sha256((output / "diagnostic.json").read_bytes()).hexdigest()
    assert (output / "checks.json").read_bytes() == runner["CHECKS"].read_bytes()
    assert (output / "config.toml").read_bytes() == runner["CONFIG"].read_bytes()
    assert json.loads((output / "status.json").read_bytes()) == status
    with pytest.raises(FileExistsError):
        runner["run_trial"](output, allow_download=False, wait_seconds=30)


def test_wait_limit_never_starts_inference(runner, monkeypatch, tmp_path):
    monkeypatch.setitem(runner, "available_memory_bytes", lambda: None)
    ticks = iter([0, 2])
    monkeypatch.setitem(runner, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(
        runner["subprocess"], "run", lambda *args, **kwargs: pytest.fail("inference")
    )
    output = tmp_path / "trial"
    with pytest.raises(TimeoutError, match="wait limit"):
        runner["run_trial"](output, allow_download=False, wait_seconds=1)
    status = json.loads((output / "status.json").read_bytes())
    assert status["state"] == "failed" and "TimeoutError" in status["error"]


def test_failed_inference_is_not_reported_as_completed(runner, monkeypatch, tmp_path):
    monkeypatch.setitem(runner, "available_memory_bytes", lambda: 18 * 1024**3)

    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(2, command)

    monkeypatch.setattr(runner["subprocess"], "run", fail)
    output = tmp_path / "trial"
    with pytest.raises(subprocess.CalledProcessError):
        runner["run_trial"](output, allow_download=False, wait_seconds=30)
    status = json.loads((output / "status.json").read_bytes())
    assert status["state"] == "failed" and "summary" not in status
    assert not (output / "replay.log").exists()


def test_weights_must_match_pinned_hashes(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    verify = runpy.run_path(str(ROOT / "scripts/run_local_model_trial.py"))["verify_weights"]
    payload = b"test weights"
    monkeypatch.setitem(
        verify.__globals__, "WEIGHT_HASHES", {"weights": sha256(payload).hexdigest()}
    )
    path = tmp_path / "weights"
    path.write_bytes(payload)
    verify(tmp_path)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify(tmp_path)
