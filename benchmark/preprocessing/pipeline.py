"""End-to-end, configuration-driven dataset preprocessing pipeline."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml

from benchmark.config import ConfigurationError, load_config
from benchmark.constants import DEFAULT_CONFIG_FILENAME, PROJECT_ROOT
from benchmark.preprocessing.loader import DatasetLoader
from benchmark.preprocessing.profiling import DatasetProfiler
from benchmark.preprocessing.schema import DatasetSchema
from benchmark.preprocessing.splitter import DatasetSplits, DatasetSplitter, SplitConfig
from benchmark.preprocessing.transformer import (
    DatasetTransformer,
    MissingValueStrategy,
    TransformationConfig,
)
from benchmark.preprocessing.validator import (
    DatasetValidator,
    ValidationOptions,
    ValidationReport,
)
from benchmark.utils.filesystem import ensure_directory, resolve_path

LOGGER = logging.getLogger(__name__)


class DatasetValidationError(RuntimeError):
    """Raised when error-level validation findings block preprocessing."""

    def __init__(self, report: ValidationReport) -> None:
        """Initialize the error with the report that blocked processing."""
        super().__init__("Dataset validation failed; inspect validation reports")
        self.report = report


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    """Complete settings for one reproducible preprocessing pipeline."""

    input_path: Path
    artifacts_dir: Path
    schema: DatasetSchema
    encoding: str = "utf-8"
    delimiter: str = ","
    validation: ValidationOptions = field(default_factory=ValidationOptions)
    transformation: TransformationConfig = field(default_factory=TransformationConfig)
    split: SplitConfig = field(default_factory=SplitConfig)

    @property
    def profiles_dir(self) -> Path:
        """Return the configured dataset profile artifact directory."""
        return self.artifacts_dir / "profiles"

    @property
    def reports_dir(self) -> Path:
        """Return the configured validation report artifact directory."""
        return self.artifacts_dir / "reports"

    @property
    def splits_dir(self) -> Path:
        """Return the configured dataset split artifact directory."""
        return self.artifacts_dir / "splits"


def load_preprocessing_config(path: Path | None = None) -> PreprocessingConfig:
    """Load typed preprocessing settings from the project YAML configuration."""
    config_path = (
        path or PROJECT_ROOT / "benchmark" / "config" / DEFAULT_CONFIG_FILENAME
    )
    app_config = load_config(config_path)
    values = _load_yaml(config_path)
    preprocessing = _mapping(
        values.get("preprocessing"), "preprocessing", required=True
    )
    schema_values = _mapping(
        preprocessing.get("schema"), "preprocessing.schema", required=True
    )
    loader_values = _mapping(preprocessing.get("loader"), "preprocessing.loader")
    validation_values = _mapping(
        preprocessing.get("validation"), "preprocessing.validation"
    )
    transformation_values = _mapping(
        preprocessing.get("transformation"), "preprocessing.transformation"
    )
    split_values = _mapping(preprocessing.get("split"), "preprocessing.split")

    target_column = _required_string(schema_values, "target_column")
    timestamp_value = schema_values.get("timestamp_column")
    timestamp_column = str(timestamp_value) if timestamp_value is not None else None
    input_path = resolve_path(
        _required_string(preprocessing, "input_path"), PROJECT_ROOT
    )

    try:
        missing_strategy = MissingValueStrategy(
            str(transformation_values.get("missing_values", "none"))
        )
    except ValueError as error:
        raise ConfigurationError("Unsupported missing-value strategy") from error

    return PreprocessingConfig(
        input_path=input_path,
        artifacts_dir=app_config.paths.artifacts_dir,
        schema=DatasetSchema(
            target_column=target_column,
            timestamp_column=timestamp_column,
            required_columns=_string_tuple(schema_values.get("required_columns", [])),
            optional_columns=_string_tuple(schema_values.get("optional_columns", [])),
            data_types=_string_mapping(schema_values.get("data_types", {})),
        ),
        encoding=str(loader_values.get("encoding", "utf-8")),
        delimiter=str(loader_values.get("delimiter", ",")),
        validation=ValidationOptions(
            check_missing_values=_boolean(
                validation_values.get("check_missing_values", True),
                "check_missing_values",
            ),
            allow_missing_values=_boolean(
                validation_values.get("allow_missing_values", False),
                "allow_missing_values",
            ),
            check_duplicate_records=_boolean(
                validation_values.get("check_duplicate_records", True),
                "check_duplicate_records",
            ),
            allow_duplicate_records=_boolean(
                validation_values.get("allow_duplicate_records", False),
                "allow_duplicate_records",
            ),
            check_duplicate_timestamps=_boolean(
                validation_values.get("check_duplicate_timestamps", True),
                "check_duplicate_timestamps",
            ),
            allow_duplicate_timestamps=_boolean(
                validation_values.get("allow_duplicate_timestamps", False),
                "allow_duplicate_timestamps",
            ),
            check_data_types=_boolean(
                validation_values.get("check_data_types", True),
                "check_data_types",
            ),
        ),
        transformation=TransformationConfig(
            missing_values=missing_strategy,
            convert_timestamp=_boolean(
                transformation_values.get("convert_timestamp", True),
                "convert_timestamp",
            ),
            selected_columns=_optional_string_tuple(
                transformation_values.get("selected_columns")
            ),
            scale_columns=_string_tuple(transformation_values.get("scale_columns", [])),
        ),
        split=SplitConfig(
            train_ratio=_float(split_values.get("train_ratio", 0.7), "train_ratio"),
            validation_ratio=_float(
                split_values.get("validation_ratio", 0.15), "validation_ratio"
            ),
            test_ratio=_float(split_values.get("test_ratio", 0.15), "test_ratio"),
            random_seed=_integer(
                split_values.get("random_seed", app_config.random_seed), "random_seed"
            ),
            shuffle=_boolean(split_values.get("shuffle", False), "shuffle"),
            time_series=_boolean(split_values.get("time_series", True), "time_series"),
        ),
    )


@dataclass(slots=True)
class BQEBPreprocessingPipeline:
    """Orchestrate loading, validation, profiling, transformation, and splitting."""

    config: PreprocessingConfig
    loader: DatasetLoader = field(init=False)
    validator: DatasetValidator = field(init=False)
    profiler: DatasetProfiler = field(init=False)
    transformer: DatasetTransformer = field(init=False)
    splitter: DatasetSplitter = field(init=False)

    def __post_init__(self) -> None:
        """Build focused preprocessing services from immutable configuration."""
        self.loader = DatasetLoader(self.config.encoding, self.config.delimiter)
        self.validator = DatasetValidator(self.config.schema, self.config.validation)
        self.profiler = DatasetProfiler()
        self.transformer = DatasetTransformer(
            self.config.schema, self.config.transformation
        )
        self.splitter = DatasetSplitter(self.config.split)

    def fit(
        self, source: Path | pd.DataFrame | None = None
    ) -> BQEBPreprocessingPipeline:
        """Load, inspect, and fit transformation state without creating splits."""
        dataset = self._load(source)
        self._inspect(dataset)
        self.transformer.fit(dataset)
        return self

    def transform(self, source: Path | pd.DataFrame | None = None) -> DatasetSplits:
        """Inspect, transform, split, and export data using fitted state."""
        dataset = self._load(source)
        self._inspect(dataset)
        transformed = self.transformer.transform(dataset)
        return self._split_and_export(transformed)

    def fit_transform(self, source: Path | pd.DataFrame | None = None) -> DatasetSplits:
        """Execute the complete preprocessing flow and export all artifacts."""
        dataset = self._load(source)
        self._inspect(dataset)
        transformed = self.transformer.fit_transform(dataset)
        return self._split_and_export(transformed)

    def save(self, path: Path) -> Path:
        """Serialize the fitted pipeline to ``path`` using joblib."""
        destination = ensure_directory(path.parent) / path.name
        joblib.dump(self, destination)
        LOGGER.info("Saved preprocessing pipeline path=%s", destination)
        return destination

    @classmethod
    def load(cls, path: Path) -> BQEBPreprocessingPipeline:
        """Load a pipeline from a trusted joblib file."""
        source = path.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Pipeline file does not exist: {source}")
        pipeline = joblib.load(source)
        if not isinstance(pipeline, cls):
            raise TypeError("Serialized object is not a BQEB preprocessing pipeline")
        LOGGER.info("Loaded preprocessing pipeline path=%s", source)
        return pipeline

    def _load(self, source: Path | pd.DataFrame | None) -> pd.DataFrame:
        if isinstance(source, pd.DataFrame):
            LOGGER.info(
                "Using in-memory dataset rows=%d columns=%d",
                len(source),
                len(source.columns),
            )
            return source.copy(deep=True)
        return self.loader.load(source or self.config.input_path)

    def _inspect(self, dataset: pd.DataFrame) -> None:
        report = self.validator.validate(dataset)
        report.save_json(self.config.reports_dir / "validation_report.json")
        report.save_markdown(self.config.reports_dir / "validation_report.md")
        self.profiler.profile(dataset).export(self.config.profiles_dir)
        if not report.valid:
            raise DatasetValidationError(report)

    def _split_and_export(self, dataset: pd.DataFrame) -> DatasetSplits:
        splits = self.splitter.split(dataset, self.config.schema.timestamp_column)
        destination = ensure_directory(self.config.splits_dir)
        splits.train.to_csv(destination / "train.csv", index=False)
        splits.validation.to_csv(destination / "validation.csv", index=False)
        splits.test.to_csv(destination / "test.csv", index=False)
        LOGGER.info("Exported dataset splits directory=%s", destination)
        return splits


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        values = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigurationError(
            f"Unable to read preprocessing config: {path}"
        ) from error
    if not isinstance(values, dict):
        raise ConfigurationError("Configuration root must be a mapping")
    return values


def _mapping(value: object, name: str, *, required: bool = False) -> Mapping[str, Any]:
    if value is None and not required:
        return {}
    if not isinstance(value, dict):
        raise ConfigurationError(f"{name} must be a mapping")
    return value


def _required_string(values: Mapping[str, Any], name: str) -> str:
    value = values.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{name} must be a non-empty string")
    return value


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigurationError("Column lists must contain only strings")
    return tuple(value)


def _optional_string_tuple(value: object) -> tuple[str, ...] | None:
    return None if value is None else _string_tuple(value)


def _string_mapping(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise ConfigurationError("data_types must map column names to strings")
    return dict(value)


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{name} must be a boolean")
    return value


def _float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigurationError(f"{name} must be numeric")
    return float(value)


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{name} must be an integer")
    return value
