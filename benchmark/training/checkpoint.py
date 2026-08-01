"""Checkpoint persistence and discovery for trained forecasting models."""

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
from benchmark.training.model_io import SERIALIZATION_VERSION
from benchmark.utils.filesystem import ensure_directory

LOGGER = logging.getLogger(__name__)


class CheckpointError(RuntimeError):
    """Raised when checkpoint persistence or loading fails."""


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """Serializable recovery record for one trained model state."""

    serialization_version: int
    model: BaseForecastModel
    configuration: dict[str, Any]
    timestamp: str
    model_version: str
    repository_version: str


def save_checkpoint(
    model: BaseForecastModel,
    configuration: dict[str, Any],
    directory: Path,
    *,
    compression: int = 3,
    protocol: int = 5,
) -> Path:
    """Persist a timestamped checkpoint and return its path."""
    destination = ensure_directory(directory)
    now = datetime.now(UTC)
    path = destination / (
        f"{model.get_name()}-{now.strftime('%Y%m%dT%H%M%S%fZ')}.checkpoint.joblib"
    )
    checkpoint = Checkpoint(
        serialization_version=SERIALIZATION_VERSION,
        model=model,
        configuration=deepcopy(configuration),
        timestamp=now.isoformat(),
        model_version=model.get_version(),
        repository_version=VERSION,
    )
    try:
        joblib.dump(checkpoint, path, compress=compression, protocol=protocol)
    except Exception as error:
        LOGGER.exception("Unable to save checkpoint path=%s", path)
        raise CheckpointError(f"Unable to save checkpoint: {path}") from error
    LOGGER.info("Saved checkpoint model=%s path=%s", model.get_name(), path)
    return path


def load_checkpoint(path: Path) -> Checkpoint:
    """Load a trusted checkpoint and validate its serialization version."""
    source = path.expanduser().resolve()
    if not source.is_file():
        raise CheckpointError(f"Checkpoint does not exist: {source}")
    try:
        checkpoint = joblib.load(source)
    except Exception as error:
        LOGGER.exception("Unable to load checkpoint path=%s", source)
        raise CheckpointError(f"Unable to load checkpoint: {source}") from error
    if not isinstance(checkpoint, Checkpoint):
        raise CheckpointError("Serialized value is not a ForecastBench checkpoint")
    if checkpoint.serialization_version != SERIALIZATION_VERSION:
        raise CheckpointError(
            f"Unsupported checkpoint version: {checkpoint.serialization_version}"
        )
    LOGGER.info(
        "Loaded checkpoint model=%s path=%s", checkpoint.model.get_name(), source
    )
    return checkpoint


def latest_checkpoint(directory: Path, model_name: str | None = None) -> Path | None:
    """Return the newest matching checkpoint, or ``None`` when none exist."""
    source = directory.expanduser().resolve()
    if not source.is_dir():
        return None
    pattern = (
        f"{model_name}-*.checkpoint.joblib"
        if model_name is not None
        else "*.checkpoint.joblib"
    )
    candidates = list(source.glob(pattern))
    return max(candidates, key=lambda path: path.stat().st_mtime_ns, default=None)
