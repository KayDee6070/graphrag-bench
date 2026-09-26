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
