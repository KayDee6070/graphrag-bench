import json
import subprocess
import sys

from graphrag_bench.cli import main


def test_cli_validates_fixture(fixture_root, capsys):
    assert main(["validate-fixtures", str(fixture_root)]) == 0
    output = capsys.readouterr()
    assert json.loads(output.out) == {
        "status": "valid",
        "documents": 8,
        "chunks": 25,
        "entities": 15,
        "relations": 23,
        "questions": 20,
    }
    assert output.err == ""


def test_cli_failure_has_nonzero_status_without_traceback(tmp_path, capsys):
    assert main(["validate-fixtures", str(tmp_path)]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "manifest.json" in output.err
    assert "Traceback" not in output.err


def test_installed_module_runs_outside_repository(fixture_root, tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "graphrag_bench", "validate-fixtures", str(fixture_root)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "valid"


def test_ingest_cli_overrides_toml_without_loading_gold(tmp_path, capsys, monkeypatch):
    def fail_gold(*args, **kwargs):
        raise AssertionError("ingestion must not read gold annotations")

    monkeypatch.setattr("graphrag_bench.cli.load_fixture", fail_gold)
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.txt").write_text("Alpha. Beta. Gamma.")
    config = tmp_path / "config.toml"
    config.write_text("[chunking]\nmax_units = 2\noverlap_units = 1\n")
    output = tmp_path / "output"
    status = main(
        [
            "ingest",
            str(source),
            "--output",
            str(output),
            "--config",
            str(config),
            "--max-units",
            "1",
            "--overlap-units",
            "0",
        ]
    )
    assert status == 0
    summary = json.loads(capsys.readouterr().out)
    assert (summary["documents"], summary["chunks"], summary["status"]) == (1, 3, "ingested")
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["chunking"]["max_units"] == 1


def test_ingest_cli_invalid_config_fails_without_traceback(tmp_path, capsys):
    assert (
        main(["ingest", str(tmp_path), "--output", str(tmp_path / "out"), "--max-chars", "0"]) == 1
    )
    output = capsys.readouterr()
    assert output.out == ""
    assert "max_chars" in output.err
    assert "Traceback" not in output.err
    assert not (tmp_path / "out").exists()
