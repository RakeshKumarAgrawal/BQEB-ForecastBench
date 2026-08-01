"""Runtime environment definitions and parsing."""

from collections.abc import Mapping
from enum import StrEnum

from benchmark.constants import ENV_PREFIX


class Environment(StrEnum):
    """Supported ForecastBench runtime environments."""

    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


def get_environment(environ: Mapping[str, str]) -> Environment:
    """Return the configured runtime environment from an environment mapping."""
    value = environ.get(f"{ENV_PREFIX}ENVIRONMENT", Environment.DEVELOPMENT.value)
    try:
        return Environment(value.strip().lower())
    except ValueError as error:
        supported = ", ".join(environment.value for environment in Environment)
        raise ValueError(
            f"Unsupported environment {value!r}; expected one of: {supported}"
        ) from error
