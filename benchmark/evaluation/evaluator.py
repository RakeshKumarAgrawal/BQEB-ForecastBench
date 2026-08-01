"""Configuration-driven benchmark evaluation orchestration."""

from __future__ import annotations

import logging
import random
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from benchmark.constants import CONFIG_DIR, PROJECT_ROOT, VERSION
from benchmark.evaluation.registry import METRIC_REGISTRY, MetricRegistry
from benchmark.evaluation.results import EvaluationResult, write_evaluation_artifacts
from benchmark.models.base_model import BaseForecastModel, FeatureMatrix, TargetVector
from benchmark.training.history import configuration_hash
from benchmark.utils.filesystem import resolve_path

LOGGER = logging.getLogger(__name__)


class EvaluationConfigurationError(ValueError):
    """Raised when evaluation configuration is missing or invalid."""


@dataclass(frozen=True, slots=True)
class EvaluationArtifactSettings:
    """Resolved JSON artifact locations and filenames."""

    directory: Path
    experiments_directory: Path
    metrics_filename: str
    log_filename: str
    metrics_csv_filename: str
    benchmark_results_filename: str
    predictions_filename: str
    model_comparison_filename: str
    manifest_filename: str


@dataclass(frozen=True, slots=True)
class PredictionOutputSettings:
    """Control whether prediction values are retained in the JSON log."""

    enabled: bool
    include_in_evaluation_log: bool


@dataclass(frozen=True, slots=True)
class BenchmarkDatasetSettings:
    """Dataset split paths and columns consumed by benchmark experiments."""

    split_paths: dict[str, Path]
    target_column: str
    timestamp_column: str
    sample_id_column: str | None


@dataclass(frozen=True, slots=True)
class BenchmarkModelSettings:
    """Registry selection and model acquisition behavior for benchmark runs."""

    names: tuple[str, ...] | None
    source: str
    artifact_directory: Path


@dataclass(frozen=True, slots=True)
class EvaluationSettings:
    """Validated settings consumed by ``BenchmarkEvaluator``."""

    dataset: str
    experiment_name: str
    enabled_metrics: tuple[str, ...]
    primary_metric: str
    persist_artifacts: bool
    prediction_output: PredictionOutputSettings
    artifacts: EvaluationArtifactSettings
    benchmark_dataset: BenchmarkDatasetSettings
    benchmark_models: BenchmarkModelSettings
    random_seed: int
    logging_level: str
    logging_console: bool

    def snapshot(self) -> dict[str, Any]:
        """Return a portable configuration snapshot for provenance."""
        return {
            "dataset": self.dataset,
            "experiment_name": self.experiment_name,
            "enabled_metrics": list(self.enabled_metrics),
            "options": {
                "primary_metric": self.primary_metric,
                "persist_artifacts": self.persist_artifacts,
            },
            "prediction_output": {
                "enabled": self.prediction_output.enabled,
                "include_in_evaluation_log": (
                    self.prediction_output.include_in_evaluation_log
                ),
            },
            "artifact_locations": {
                "evaluation": str(self.artifacts.directory),
                "experiments": str(self.artifacts.experiments_directory),
                "metrics_filename": self.artifacts.metrics_filename,
                "log_filename": self.artifacts.log_filename,
                "metrics_csv_filename": self.artifacts.metrics_csv_filename,
                "benchmark_results_filename": (
                    self.artifacts.benchmark_results_filename
                ),
                "predictions_filename": self.artifacts.predictions_filename,
                "model_comparison_filename": (self.artifacts.model_comparison_filename),
                "manifest_filename": self.artifacts.manifest_filename,
            },
            "dataset_splits": {
                "paths": {
                    name: str(path)
                    for name, path in sorted(self.benchmark_dataset.split_paths.items())
                },
                "target_column": self.benchmark_dataset.target_column,
                "timestamp_column": self.benchmark_dataset.timestamp_column,
                "sample_id_column": self.benchmark_dataset.sample_id_column,
            },
            "models": {
                "selection": (
                    "all_registered"
                    if self.benchmark_models.names is None
                    else list(self.benchmark_models.names)
                ),
                "source": self.benchmark_models.source,
                "artifact_directory": str(self.benchmark_models.artifact_directory),
            },
            "random_seed": self.random_seed,
            "logging": {
                "level": self.logging_level,
                "console": self.logging_console,
            },
        }


def load_evaluation_settings(path: Path | None = None) -> EvaluationSettings:
    """Load and validate evaluation settings from YAML."""
    source = (path or CONFIG_DIR / "evaluation.yaml").expanduser().resolve()
    if not source.is_file():
        raise EvaluationConfigurationError(
            f"Evaluation configuration does not exist: {source}"
        )
    try:
        values = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise EvaluationConfigurationError(
            f"Unable to read evaluation configuration: {source}"
        ) from error
    root = _mapping(values, "configuration root")
    evaluation = _mapping(root.get("evaluation"), "evaluation")
    options = _mapping(evaluation.get("options"), "evaluation.options")
    prediction = _mapping(
        evaluation.get("prediction_output"), "evaluation.prediction_output"
    )
    artifacts = _mapping(
        evaluation.get("artifact_locations"), "evaluation.artifact_locations"
    )
    split_values = _optional_mapping(
        evaluation.get("dataset_splits"), "evaluation.dataset_splits"
    )
    split_paths = _optional_mapping(
        split_values.get("paths"), "evaluation.dataset_splits.paths"
    )
    model_values = _optional_mapping(evaluation.get("models"), "evaluation.models")
    logging_values = _mapping(evaluation.get("logging"), "evaluation.logging")
    enabled_metrics = _string_list(
        evaluation.get("enabled_metrics"), "evaluation.enabled_metrics"
    )
    primary_metric = _string(options.get("primary_metric"), "options.primary_metric")
    if primary_metric not in enabled_metrics:
        raise EvaluationConfigurationError(
            "options.primary_metric must be included in enabled_metrics"
        )
    return EvaluationSettings(
        dataset=_string(evaluation.get("dataset"), "evaluation.dataset"),
        experiment_name=_string(
            evaluation.get("experiment_name", "forecastbench-baselines"),
            "evaluation.experiment_name",
        ),
        enabled_metrics=enabled_metrics,
        primary_metric=primary_metric,
        persist_artifacts=_boolean(
            options.get("persist_artifacts"), "options.persist_artifacts"
        ),
        prediction_output=PredictionOutputSettings(
            enabled=_boolean(prediction.get("enabled"), "prediction_output.enabled"),
            include_in_evaluation_log=_boolean(
                prediction.get("include_in_evaluation_log"),
                "prediction_output.include_in_evaluation_log",
            ),
        ),
        artifacts=EvaluationArtifactSettings(
            directory=resolve_path(
                _string(artifacts.get("evaluation"), "artifact_locations.evaluation"),
                PROJECT_ROOT,
            ),
            experiments_directory=resolve_path(
                _string(
                    artifacts.get("experiments", "artifacts/experiments"),
                    "artifact_locations.experiments",
                ),
                PROJECT_ROOT,
            ),
            metrics_filename=_filename(
                artifacts.get("metrics_filename"), "artifact_locations.metrics_filename"
            ),
            log_filename=_filename(
                artifacts.get("log_filename"), "artifact_locations.log_filename"
            ),
            metrics_csv_filename=_filename_with_suffix(
                artifacts.get("metrics_csv_filename", "metrics.csv"),
                "artifact_locations.metrics_csv_filename",
                ".csv",
            ),
            benchmark_results_filename=_filename_with_suffix(
                artifacts.get("benchmark_results_filename", "benchmark_results.csv"),
                "artifact_locations.benchmark_results_filename",
                ".csv",
            ),
            predictions_filename=_filename_with_suffix(
                artifacts.get("predictions_filename", "predictions.csv"),
                "artifact_locations.predictions_filename",
                ".csv",
            ),
            model_comparison_filename=_filename_with_suffix(
                artifacts.get("model_comparison_filename", "model_comparison.csv"),
                "artifact_locations.model_comparison_filename",
                ".csv",
            ),
            manifest_filename=_filename(
                artifacts.get("manifest_filename", "experiment_manifest.json"),
                "artifact_locations.manifest_filename",
            ),
        ),
        benchmark_dataset=BenchmarkDatasetSettings(
            split_paths={
                name: resolve_path(
                    _string(
                        split_paths.get(name, f"artifacts/splits/{name}.csv"),
                        f"dataset_splits.paths.{name}",
                    ),
                    PROJECT_ROOT,
                )
                for name in ("train", "validation", "test")
            },
            target_column=_string(
                split_values.get("target_column", "load_kw"),
                "dataset_splits.target_column",
            ),
            timestamp_column=_string(
                split_values.get("timestamp_column", "timestamp"),
                "dataset_splits.timestamp_column",
            ),
            sample_id_column=_optional_string(
                split_values.get("sample_id_column"),
                "dataset_splits.sample_id_column",
            ),
        ),
        benchmark_models=BenchmarkModelSettings(
            names=_model_names(model_values.get("selection", "all_registered")),
            source=_model_source(model_values.get("source", "fit_from_training_split")),
            artifact_directory=resolve_path(
                _string(
                    model_values.get("artifact_directory", "artifacts/models"),
                    "models.artifact_directory",
                ),
                PROJECT_ROOT,
            ),
        ),
        random_seed=_integer(evaluation.get("random_seed"), "random_seed", minimum=0),
        logging_level=_string(logging_values.get("level"), "logging.level").upper(),
        logging_console=_boolean(logging_values.get("console"), "logging.console"),
    )


class BenchmarkEvaluator:
    """Generate model predictions, compute configured metrics, and persist results."""

    def __init__(
        self,
        config_path: Path | None = None,
        *,
        registry: MetricRegistry = METRIC_REGISTRY,
    ) -> None:
        """Initialize an evaluator from configuration and a metric registry."""
        self.settings = load_evaluation_settings(config_path)
        self._registry = registry
        for metric_name in self.settings.enabled_metrics:
            self._registry.get(metric_name)

    def evaluate(
        self,
        model: BaseForecastModel,
        features: FeatureMatrix,
        target: TargetVector,
    ) -> EvaluationResult:
        """Evaluate a trained model against one configured dataset."""
        random.seed(self.settings.random_seed)
        np.random.seed(self.settings.random_seed)
        LOGGER.info(
            "Evaluation started model=%s dataset=%s",
            model.get_name(),
            self.settings.dataset,
        )
        predictions = model.predict(features)
        return self.evaluate_predictions(
            model.get_name(), target, predictions, dataset=self.settings.dataset
        )

    def evaluate_predictions(
        self,
        model_name: str,
        target: TargetVector,
        predictions: TargetVector,
        *,
        dataset: str,
        persist: bool | None = None,
    ) -> EvaluationResult:
        """Compute configured metrics for aligned precomputed predictions."""
        values = {
            name: self._registry.get(name)(target, predictions)
            for name in self.settings.enabled_metrics
        }
        standard_names = {"mae", "rmse", "mape", "r2"}
        result = EvaluationResult(
            model_name=model_name,
            dataset=dataset,
            mae=values.get("mae"),
            rmse=values.get("rmse"),
            mape=values.get("mape"),
            r2=values.get("r2"),
            prediction_count=len(predictions),
            evaluation_timestamp=datetime.now(UTC).isoformat(),
            configuration_hash=configuration_hash(self.settings.snapshot()),
            repository_version=VERSION,
            additional_metrics={
                name: value
                for name, value in values.items()
                if name not in standard_names
            },
        )
        should_persist = self.settings.persist_artifacts if persist is None else persist
        if should_persist:
            include_predictions = (
                self.settings.prediction_output.enabled
                and self.settings.prediction_output.include_in_evaluation_log
            )
            write_evaluation_artifacts(
                result,
                self.settings.artifacts.directory,
                metrics_filename=self.settings.artifacts.metrics_filename,
                log_filename=self.settings.artifacts.log_filename,
                predictions=np.asarray(predictions) if include_predictions else None,
            )
        LOGGER.info(
            "Evaluation completed model=%s predictions=%d",
            model_name,
            len(predictions),
        )
        return result


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise EvaluationConfigurationError(f"{name} must be a mapping")
    return value


def _optional_mapping(value: object, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    return _mapping(value, name)


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationConfigurationError(f"{name} must be a non-empty string")
    return value.strip()


def _optional_string(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _string(value, name)


def _string_list(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise EvaluationConfigurationError(f"{name} must be a non-empty list")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise EvaluationConfigurationError(f"{name} entries must be non-empty strings")
    normalized = tuple(item.strip().lower() for item in value)
    if len(set(normalized)) != len(normalized):
        raise EvaluationConfigurationError(f"{name} must not contain duplicates")
    return normalized


def _integer(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise EvaluationConfigurationError(f"{name} must be an integer >= {minimum}")
    return value


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise EvaluationConfigurationError(f"{name} must be a boolean")
    return value


def _filename(value: object, name: str) -> str:
    return _filename_with_suffix(value, name, ".json")


def _filename_with_suffix(value: object, name: str, suffix: str) -> str:
    filename = _string(value, name)
    if Path(filename).name != filename or not filename.endswith(suffix):
        raise EvaluationConfigurationError(f"{name} must be a {suffix} filename")
    return filename


def _model_names(value: object) -> tuple[str, ...] | None:
    if value == "all_registered":
        return None
    return _string_list(value, "models.selection")


def _model_source(value: object) -> str:
    source = _string(value, "models.source")
    if source not in {"fit_from_training_split", "artifacts"}:
        raise EvaluationConfigurationError(
            "models.source must be 'fit_from_training_split' or 'artifacts'"
        )
    return source
