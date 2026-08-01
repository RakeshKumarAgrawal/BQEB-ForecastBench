"""Typed application configuration loaded from YAML and environment variables."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from benchmark.config.environment import Environment, get_environment
from benchmark.constants import (
    ARTIFACTS_DIR,
    DATA_DIR,
    DEFAULT_CONFIG_FILENAME,
    ENV_PREFIX,
    PROJECT_ROOT,
)
from benchmark.utils.filesystem import resolve_path


class ConfigurationError(ValueError):
    """Raised when ForecastBench configuration is invalid."""


@dataclass(frozen=True, slots=True)
class PathSettings:
    """Filesystem locations used by ForecastBench."""

    data_dir: Path = DATA_DIR
    artifacts_dir: Path = ARTIFACTS_DIR


@dataclass(frozen=True, slots=True)
class LoggingSettings:
    """Logging behavior for ForecastBench processes."""

    level: str = "INFO"
    file: Path | None = None
    console: bool = True


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Complete immutable ForecastBench application configuration."""

    environment: Environment = Environment.DEVELOPMENT
    paths: PathSettings = PathSettings()
    logging: LoggingSettings = LoggingSettings()
    random_seed: int = 42


def load_config(
    path: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> AppConfig:
    """Load configuration from YAML, overridden by ``BQEB_`` variables."""
    environment_values = os.environ if environ is None else environ
    config_path = (
        path or PROJECT_ROOT / "benchmark" / "config" / DEFAULT_CONFIG_FILENAME
    )
    values = _load_yaml(config_path)

    path_values = _mapping(values.get("paths"), "paths")
    logging_values = _mapping(values.get("logging"), "logging")

    data_dir = environment_values.get(
        f"{ENV_PREFIX}DATA_DIR", str(path_values.get("data_dir", DATA_DIR))
    )
    artifacts_dir = environment_values.get(
        f"{ENV_PREFIX}ARTIFACTS_DIR",
        str(path_values.get("artifacts_dir", ARTIFACTS_DIR)),
    )
    log_file_value = environment_values.get(
        f"{ENV_PREFIX}LOG_FILE", logging_values.get("file")
    )
    level = environment_values.get(
        f"{ENV_PREFIX}LOG_LEVEL", str(logging_values.get("level", "INFO"))
    ).upper()
    random_seed_value = environment_values.get(
        f"{ENV_PREFIX}RANDOM_SEED", str(values.get("random_seed", 42))
    )

    try:
        random_seed = int(random_seed_value)
    except (TypeError, ValueError) as error:
        raise ConfigurationError("random_seed must be an integer") from error

    return AppConfig(
        environment=_get_environment(environment_values, values),
        paths=PathSettings(
            data_dir=resolve_path(data_dir, PROJECT_ROOT),
            artifacts_dir=resolve_path(artifacts_dir, PROJECT_ROOT),
        ),
        logging=LoggingSettings(
            level=level,
            file=(
                resolve_path(str(log_file_value), PROJECT_ROOT)
                if log_file_value
                else None
            ),
            console=_as_bool(logging_values.get("console", True), "logging.console"),
        ),
        random_seed=random_seed,
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {path}")
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ConfigurationError(
            f"Invalid YAML in configuration file: {path}"
        ) from error
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigurationError("Configuration root must be a mapping")
    return loaded


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigurationError(f"{name} must be a mapping")
    return value


def _get_environment(
    environ: Mapping[str, str], values: Mapping[str, Any]
) -> Environment:
    merged_environment = dict(environ)
    if f"{ENV_PREFIX}ENVIRONMENT" not in merged_environment:
        merged_environment[f"{ENV_PREFIX}ENVIRONMENT"] = str(
            values.get("environment", Environment.DEVELOPMENT.value)
        )
    try:
        return get_environment(merged_environment)
    except ValueError as error:
        raise ConfigurationError(str(error)) from error


def _as_bool(value: object, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ConfigurationError(f"{name} must be a boolean")
