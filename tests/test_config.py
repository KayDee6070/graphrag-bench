import pytest
from pydantic import ValidationError

from graphrag_bench.config import ChunkingConfig, ConfigurationError, load_chunking_config


@pytest.mark.parametrize(
    "parameters",
    [
        {"max_chars": 0},
        {"max_units": 0},
        {"overlap_units": -1},
        {"max_units": 1, "overlap_units": 1},
        {"max_chars": True},
        {"max_chars": 2.5},
        {"max_characters": 200},
    ],
)
def test_invalid_chunking_parameters_are_rejected(parameters):
    with pytest.raises(ValidationError):
        ChunkingConfig(**parameters)


def test_toml_defaults_and_explicit_settings(tmp_path):
    assert load_chunking_config() == ChunkingConfig()
    path = tmp_path / "config.toml"
    path.write_text("[chunking]\nmax_chars = 120\nmax_units = 2\noverlap_units = 0\n")
    assert load_chunking_config(path) == ChunkingConfig(max_chars=120, max_units=2, overlap_units=0)


@pytest.mark.parametrize("text", ["[chunker]\n", "[chunking\n", '[chunking]\nmax_chars = "120"\n'])
def test_bad_toml_reports_configuration_error(tmp_path, text):
    path = tmp_path / "bad.toml"
    path.write_text(text)
    with pytest.raises(ConfigurationError):
        load_chunking_config(path)


def test_missing_config_reports_path(tmp_path):
    with pytest.raises(ConfigurationError, match="missing.toml"):
        load_chunking_config(tmp_path / "missing.toml")


def test_overrides_are_applied_before_validating_combined_parameters(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("[chunking]\nmax_units = 1\n")
    config = load_chunking_config(path, overrides={"overlap_units": 0})
    assert config == ChunkingConfig(max_units=1, overlap_units=0)


def test_invalid_utf8_config_reports_a_configuration_error(tmp_path):
    path = tmp_path / "encoding.toml"
    path.write_bytes(b"\xff")
    with pytest.raises(ConfigurationError, match="encoding.toml"):
        load_chunking_config(path)
