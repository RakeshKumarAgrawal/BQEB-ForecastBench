"""Tests for typed application configuration."""

from pathlib import Path

import pytest

from benchmark.config import ConfigurationError, Environment, load_config
from benchmark.constants import ARTIFACTS_DIR, DATA_DIR


def test_load_default_config() -> None:
    """The repository configuration should load into immutable typed settings."""
    config = load_config(environ={})

    assert config.environment is Environment.DEVELOPMENT
    assert config.paths.data_dir == DATA_DIR
    assert config.paths.artifacts_dir == ARTIFACTS_DIR
    assert config.logging.level == "INFO"
    assert config.random_seed == 42


def test_environment_variables_override_yaml(tmp_path: Path) -> None:
    """Environment variables should take precedence over file values."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "environment: test\nrandom_seed: 1\nlogging:\n  level: WARNING\n",
        encoding="utf-8",
    )

    config = load_config(
        config_file,
        {
            "BQEB_ENVIRONMENT": "production",
            "BQEB_DATA_DIR": str(tmp_path / "input"),
            "BQEB_LOG_LEVEL": "debug",
            "BQEB_RANDOM_SEED": "7",
        },
    )

    assert config.environment is Environment.PRODUCTION
    assert config.paths.data_dir == (tmp_path / "input").resolve()
    assert config.logging.level == "DEBUG"
    assert config.random_seed == 7


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("- invalid\n- root\n", "root must be a mapping"),
        ("logging: enabled\n", "logging must be a mapping"),
        ("logging:\n  console: sometimes\n", "logging.console must be a boolean"),
        ("random_seed: random\n", "random_seed must be an integer"),
    ],
)
def test_load_config_rejects_invalid_values(
    tmp_path: Path, content: str, message: str
) -> None:
    """Invalid configuration values should fail with a specific message."""
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text(content, encoding="utf-8")

    with pytest.raises(ConfigurationError, match=message):
        load_config(config_file, {})


def test_load_config_rejects_missing_file(tmp_path: Path) -> None:
    """A missing explicit configuration file should fail clearly."""
    with pytest.raises(ConfigurationError, match="does not exist"):
        load_config(tmp_path / "missing.yaml", {})
