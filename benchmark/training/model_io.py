"""Versioned joblib persistence for trained forecasting models."""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from benchmark.constants import VERSION
from benchmark.models.base_model import BaseForecastModel
from benchmark.utils.filesystem import ensure_directory

LOGGER = logging.getLogger(__name__)
SERIALIZATION_VERSION = 1


class ModelIOError(RuntimeError):
    """Raised when a trained-model artifact cannot be saved or loaded."""


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    """Serializable model envelope with configuration and provenance metadata."""

    serialization_version: int
    model: BaseForecastModel
    model_name: str
    model_version: str
    repository_version: str
    saved_at: str
    configuration: dict[str, Any]
    metadata: dict[str, Any]


def save_model(
    model: BaseForecastModel,
    path: Path,
    *,
    configuration: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    compression: int = 3,
    protocol: int = 5,
) -> Path:
    """Persist a trained model with a versioned metadata envelope."""
    destination = ensure_directory(path.expanduser().parent) / path.name
    artifact = ModelArtifact(
        serialization_version=SERIALIZATION_VERSION,
        model=model,
        model_name=model.get_name(),
        model_version=model.get_version(),
        repository_version=VERSION,
        saved_at=datetime.now(UTC).isoformat(),
        configuration=deepcopy(configuration or {}),
        metadata=deepcopy(metadata or {}),
    )
    try:
        joblib.dump(artifact, destination, compress=compression, protocol=protocol)
    except Exception as error:
        LOGGER.exception("Unable to save trained model path=%s", destination)
        raise ModelIOError(f"Unable to save trained model: {destination}") from error
    LOGGER.info("Saved trained model name=%s path=%s", model.get_name(), destination)
    return destination


def load_model(path: Path) -> BaseForecastModel:
    """Load and return the model from a trusted versioned artifact."""
    return load_model_artifact(path).model


def load_model_artifact(path: Path) -> ModelArtifact:
    """Load a trusted model artifact including metadata and configuration."""
    source = path.expanduser().resolve()
    if not source.is_file():
        raise ModelIOError(f"Trained model artifact does not exist: {source}")
    try:
        artifact = joblib.load(source)
    except Exception as error:
        LOGGER.exception("Unable to load trained model path=%s", source)
        raise ModelIOError(f"Unable to load trained model: {source}") from error
    if not isinstance(artifact, ModelArtifact):
        raise ModelIOError("Serialized value is not a ForecastBench model artifact")
    if artifact.serialization_version != SERIALIZATION_VERSION:
        raise ModelIOError(
            f"Unsupported model serialization version: {artifact.serialization_version}"
        )
    if not isinstance(artifact.model, BaseForecastModel):
        raise ModelIOError("Model artifact does not contain a forecasting model")
    LOGGER.info("Loaded trained model name=%s path=%s", artifact.model_name, source)
    return artifact
