"""Configuration-driven construction of registered forecasting models."""

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from benchmark.constants import CONFIG_DIR
from benchmark.models.base_model import BaseForecastModel
from benchmark.models.registry import (
    MODEL_REGISTRY,
    ModelNotRegisteredError,
    ModelRegistry,
)

LOGGER = logging.getLogger(__name__)


class ModelFactoryError(RuntimeError):
    """Raised when model configuration cannot produce a registered model."""


def create_model(
    model_name: str | None = None,
    *,
    config_path: Path | None = None,
    registry: ModelRegistry = MODEL_REGISTRY,
) -> BaseForecastModel:
    """Create a registered model using parameters from a YAML configuration."""
    source = (config_path or CONFIG_DIR / "models.yaml").expanduser().resolve()
    LOGGER.info("Creating model requested=%s config=%s", model_name, source)
    configuration = _load_configuration(source)
    selected = model_name or _required_string(configuration, "default_model")
    definitions = _required_mapping(configuration, "models")
    definition = _required_mapping(definitions, selected)

    enabled = definition.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ModelFactoryError(f"enabled must be boolean for model {selected!r}")
    if not enabled:
        raise ModelFactoryError(f"Model {selected!r} is disabled")

    parameter_definitions = definition.get("parameters", {})
    if not isinstance(parameter_definitions, dict) or not all(
        isinstance(key, str) for key in parameter_definitions
    ):
        raise ModelFactoryError(
            f"parameters must be a string-keyed mapping for model {selected!r}"
        )
    parameters = _parameter_defaults(parameter_definitions, selected)
    try:
        model_class = registry.get(selected)
        model = model_class(parameters=parameters)
    except ModelNotRegisteredError as error:
        LOGGER.exception("Configured model is not registered name=%s", selected)
        raise ModelFactoryError(f"Model {selected!r} is not registered") from error
    except Exception as error:
        LOGGER.exception("Model construction failed name=%s", selected)
        raise ModelFactoryError(f"Unable to create model {selected!r}") from error

    LOGGER.info(
        "Created model name=%s version=%s", model.get_name(), model.get_version()
    )
    return model


def _load_configuration(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise ModelFactoryError(f"Model configuration does not exist: {path}")
    try:
        values = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        LOGGER.exception("Unable to read model configuration path=%s", path)
        raise ModelFactoryError(
            f"Unable to read model configuration: {path}"
        ) from error
    if not isinstance(values, dict):
        raise ModelFactoryError("Model configuration root must be a mapping")
    return values


def _required_mapping(values: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = values.get(name)
    if not isinstance(value, dict):
        raise ModelFactoryError(f"Configuration entry {name!r} must be a mapping")
    return value


def _required_string(values: Mapping[str, Any], name: str) -> str:
    value = values.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ModelFactoryError(
            f"Configuration entry {name!r} must be a non-empty string"
        )
    return value.strip()


def _parameter_defaults(
    definitions: Mapping[str, Any], model_name: str
) -> dict[str, Any]:
    parameters: dict[str, Any] = {}
    for name, definition in definitions.items():
        if isinstance(definition, dict):
            if "default" not in definition:
                raise ModelFactoryError(
                    f"Parameter {name!r} for model {model_name!r} has no default"
                )
            parameters[name] = definition["default"]
        else:
            parameters[name] = definition
    return parameters
