"""Configuration-driven orchestration for forecasting model training."""

from __future__ import annotations

import logging
import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from benchmark.constants import CONFIG_DIR, PROJECT_ROOT
from benchmark.models import (
    MODEL_REGISTRY,
    BaseForecastModel,
    ModelRegistry,
    create_model,
)
from benchmark.models.base_model import FeatureMatrix, TargetVector
from benchmark.training.callbacks import CallbackList, TrainingCallback, TrainingContext
from benchmark.training.checkpoint import save_checkpoint
from benchmark.training.history import (
    HISTORY_FILENAME,
    TrainingHistory,
    TrainingRecord,
    configuration_hash,
)
from benchmark.training.model_io import save_model
from benchmark.utils.filesystem import resolve_path

LOGGER = logging.getLogger(__name__)


class TrainingConfigurationError(ValueError):
    """Raised when training configuration is missing or invalid."""


@dataclass(frozen=True, slots=True)
class TrainingArtifactPaths:
    """Resolved output directories for the training lifecycle."""

    models: Path
    checkpoints: Path
    training: Path


@dataclass(frozen=True, slots=True)
class SerializationSettings:
    """Joblib options used for model and checkpoint artifacts."""

    compression: int
    protocol: int


@dataclass(frozen=True, slots=True)
class TrainingSettings:
    """Validated settings consumed by ``ModelTrainer``."""

    default_model: str
    checkpoint_interval: int
    random_seed: int
    artifacts: TrainingArtifactPaths
    logging_level: str
    logging_console: bool
    serialization: SerializationSettings

    def snapshot(self) -> dict[str, Any]:
        """Return a portable configuration snapshot for persisted artifacts."""
        return {
            "default_model": self.default_model,
            "checkpoint_interval": self.checkpoint_interval,
            "random_seed": self.random_seed,
            "artifact_paths": {
                "models": str(self.artifacts.models),
                "checkpoints": str(self.artifacts.checkpoints),
                "training": str(self.artifacts.training),
            },
            "logging": {
                "level": self.logging_level,
                "console": self.logging_console,
            },
            "serialization": {
                "compression": self.serialization.compression,
                "protocol": self.serialization.protocol,
            },
        }


def load_training_settings(path: Path | None = None) -> TrainingSettings:
    """Load and validate training settings from YAML."""
    source = (path or CONFIG_DIR / "training.yaml").expanduser().resolve()
    if not source.is_file():
        raise TrainingConfigurationError(
            f"Training configuration does not exist: {source}"
        )
    try:
        values = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise TrainingConfigurationError(
            f"Unable to read training configuration: {source}"
        ) from error
    root = _mapping(values, "configuration root")
    training = _mapping(root.get("training"), "training")
    artifacts = _mapping(training.get("artifact_paths"), "training.artifact_paths")
    logging_values = _mapping(training.get("logging"), "training.logging")
    serialization = _mapping(training.get("serialization"), "training.serialization")

    default_model = _string(training.get("default_model"), "default_model")
    checkpoint_interval = _integer(
        training.get("checkpoint_interval"), "checkpoint_interval", minimum=1
    )
    random_seed = _integer(training.get("random_seed"), "random_seed", minimum=0)
    compression = _integer(
        serialization.get("compression"), "serialization.compression", minimum=0
    )
    if compression > 9:
        raise TrainingConfigurationError("serialization.compression must be <= 9")

    return TrainingSettings(
        default_model=default_model,
        checkpoint_interval=checkpoint_interval,
        random_seed=random_seed,
        artifacts=TrainingArtifactPaths(
            models=resolve_path(
                _string(artifacts.get("models"), "artifact_paths.models"),
                PROJECT_ROOT,
            ),
            checkpoints=resolve_path(
                _string(artifacts.get("checkpoints"), "artifact_paths.checkpoints"),
                PROJECT_ROOT,
            ),
            training=resolve_path(
                _string(artifacts.get("training"), "artifact_paths.training"),
                PROJECT_ROOT,
            ),
        ),
        logging_level=_string(logging_values.get("level"), "logging.level").upper(),
        logging_console=_boolean(logging_values.get("console"), "logging.console"),
        serialization=SerializationSettings(
            compression=compression,
            protocol=_integer(
                serialization.get("protocol"), "serialization.protocol", minimum=1
            ),
        ),
    )


class ModelTrainer:
    """Coordinate model creation, fitting, persistence, callbacks, and history."""

    def __init__(
        self,
        config_path: Path | None = None,
        *,
        callbacks: Iterable[TrainingCallback] = (),
        registry: ModelRegistry = MODEL_REGISTRY,
    ) -> None:
        """Initialize a trainer from configuration and optional dependencies."""
        self.settings = load_training_settings(config_path)
        self._callbacks = CallbackList(callbacks)
        self._registry = registry
        history_path = self.settings.artifacts.training / HISTORY_FILENAME
        self.history = (
            TrainingHistory.load(history_path)
            if history_path.is_file()
            else TrainingHistory()
        )
        self._completed_runs = sum(
            record.status == "completed" for record in self.history.records
        )

    def train(
        self,
        model_name: str | None,
        features: FeatureMatrix,
        target: TargetVector,
    ) -> BaseForecastModel:
        """Train one configured model and persist its operational artifacts."""
        selected = model_name or self.settings.default_model
        started_at = datetime.now(UTC)
        checkpoint_path: Path | None = None
        model: BaseForecastModel | None = None
        snapshot = self.settings.snapshot()
        context = TrainingContext(selected, {}, snapshot, started_at)
        random.seed(self.settings.random_seed)
        np.random.seed(self.settings.random_seed)
        LOGGER.info("Training run started model=%s", selected)

        try:
            model = create_model(selected, registry=self._registry)
            snapshot["model"] = {
                "name": model.get_name(),
                "parameters": model.get_parameters(),
            }
            context = TrainingContext(
                selected, model.get_parameters(), snapshot, started_at
            )
            self._callbacks.on_train_begin(context)
            model.fit(features, target)
            timestamp = started_at.strftime("%Y%m%dT%H%M%S%fZ")
            save_model(
                model,
                self.settings.artifacts.models / f"{selected}-{timestamp}.joblib",
                configuration=snapshot,
                metadata={"training_rows": len(features)},
                compression=self.settings.serialization.compression,
                protocol=self.settings.serialization.protocol,
            )
            self._completed_runs += 1
            if self._completed_runs % self.settings.checkpoint_interval == 0:
                checkpoint_path = save_checkpoint(
                    model,
                    snapshot,
                    self.settings.artifacts.checkpoints,
                    compression=self.settings.serialization.compression,
                    protocol=self.settings.serialization.protocol,
                )
                self._callbacks.on_checkpoint_saved(context, checkpoint_path)
            record = self._record(
                context,
                model,
                status="completed",
                checkpoint_path=checkpoint_path,
            )
            self._callbacks.on_train_end(context, model, record)
            self.history.add(record)
            self.history.export(self.settings.artifacts.training)
            LOGGER.info("Training run completed model=%s", selected)
            return model
        except Exception as error:
            LOGGER.exception("Training run failed model=%s", selected)
            failed = self._record(
                context,
                model,
                status="failed",
                checkpoint_path=checkpoint_path,
            )
            self.history.add(failed)
            self.history.export(self.settings.artifacts.training)
            self._callbacks.on_error(context, error)
            raise

    def _record(
        self,
        context: TrainingContext,
        model: BaseForecastModel | None,
        *,
        status: str,
        checkpoint_path: Path | None,
    ) -> TrainingRecord:
        ended_at = datetime.now(UTC)
        return TrainingRecord(
            model_name=context.model_name,
            parameters=model.get_parameters() if model is not None else {},
            start_time=context.started_at.isoformat(),
            end_time=ended_at.isoformat(),
            duration_seconds=(ended_at - context.started_at).total_seconds(),
            status=status,
            configuration_hash=configuration_hash(context.configuration),
            checkpoint_location=(str(checkpoint_path) if checkpoint_path else None),
        )


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise TrainingConfigurationError(f"{name} must be a mapping")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrainingConfigurationError(f"{name} must be a non-empty string")
    return value.strip()


def _integer(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TrainingConfigurationError(f"{name} must be an integer >= {minimum}")
    return value


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TrainingConfigurationError(f"{name} must be a boolean")
    return value
