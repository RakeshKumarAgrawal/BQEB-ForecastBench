"""Tests for end-to-end preprocessing orchestration."""

from pathlib import Path

import pandas as pd
import pytest

from benchmark.config import ConfigurationError
from benchmark.preprocessing.pipeline import (
    BQEBPreprocessingPipeline,
    DatasetValidationError,
    PreprocessingConfig,
    load_preprocessing_config,
)
from benchmark.preprocessing.schema import DatasetSchema
from benchmark.preprocessing.splitter import SplitConfig
from benchmark.preprocessing.transformer import (
    MissingValueStrategy,
    TransformationConfig,
)
from benchmark.preprocessing.validator import ValidationOptions


def _dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=10, freq="h").astype(str),
            "load": [float(value) for value in range(10)],
            "temperature": [20.0 + value for value in range(10)],
        }
    )


def _config(tmp_path: Path) -> PreprocessingConfig:
    return PreprocessingConfig(
        input_path=tmp_path / "unused.csv",
        artifacts_dir=tmp_path / "artifacts",
        schema=DatasetSchema(
            "load",
            "timestamp",
            required_columns=("temperature",),
            data_types={"load": "numeric", "temperature": "numeric"},
        ),
        validation=ValidationOptions(),
        transformation=TransformationConfig(
            missing_values=MissingValueStrategy.MEAN,
            scale_columns=("temperature",),
        ),
        split=SplitConfig(0.6, 0.2, 0.2),
    )


def test_pipeline_fit_transform_generates_all_artifacts(tmp_path: Path) -> None:
    """The complete flow should create reports, profiles, and three splits."""
    pipeline = BQEBPreprocessingPipeline(_config(tmp_path))

    splits = pipeline.fit_transform(_dataset())

    assert (len(splits.train), len(splits.validation), len(splits.test)) == (6, 2, 2)
    expected = {
        "reports/validation_report.json",
        "reports/validation_report.md",
        "profiles/dataset_statistics.csv",
        "profiles/dataset_statistics.md",
        "splits/train.csv",
        "splits/validation.csv",
        "splits/test.csv",
    }
    generated = {
        path.relative_to(tmp_path / "artifacts").as_posix()
        for path in (tmp_path / "artifacts").rglob("*")
        if path.is_file()
    }
    assert generated == expected


def test_pipeline_save_load_and_transform(tmp_path: Path) -> None:
    """A fitted pipeline should retain transformation state after serialization."""
    pipeline = BQEBPreprocessingPipeline(_config(tmp_path)).fit(_dataset())
    path = pipeline.save(tmp_path / "state" / "pipeline.joblib")

    restored = BQEBPreprocessingPipeline.load(path)
    splits = restored.transform(_dataset())

    assert len(splits.train) == 6


def test_pipeline_writes_report_before_rejecting_invalid_data(tmp_path: Path) -> None:
    """Failed validation should still leave actionable reports for diagnosis."""
    invalid = _dataset().drop(columns="load")
    pipeline = BQEBPreprocessingPipeline(_config(tmp_path))

    with pytest.raises(DatasetValidationError) as error:
        pipeline.fit_transform(invalid)

    assert error.value.report.valid is False
    assert (tmp_path / "artifacts/reports/validation_report.json").is_file()


def test_load_preprocessing_config_reads_nested_yaml(tmp_path: Path) -> None:
    """YAML settings should populate every pipeline configuration group."""
    config_path = tmp_path / "forecastbench.yaml"
    config_path.write_text(
        """
environment: test
random_seed: 9
paths:
  data_dir: data
  artifacts_dir: artifacts
logging:
  console: false
preprocessing:
  input_path: data/input.csv
  loader:
    encoding: utf-16
    delimiter: ";"
  schema:
    target_column: load
    timestamp_column: timestamp
    required_columns: [temperature]
    optional_columns: []
    data_types:
      load: numeric
  validation:
    allow_missing_values: true
  transformation:
    missing_values: median
    scale_columns: [temperature]
  split:
    train_ratio: 0.6
    validation_ratio: 0.2
    test_ratio: 0.2
    random_seed: 7
    shuffle: true
    time_series: false
""".lstrip(),
        encoding="utf-8",
    )

    config = load_preprocessing_config(config_path)

    assert config.encoding == "utf-16"
    assert config.delimiter == ";"
    assert config.schema.target_column == "load"
    assert config.validation.allow_missing_values
    assert config.transformation.missing_values is MissingValueStrategy.MEDIAN
    assert config.split.random_seed == 7


def test_load_preprocessing_config_rejects_invalid_types(tmp_path: Path) -> None:
    """Configuration type errors should fail before any dataset is loaded."""
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        """
paths:
  artifacts_dir: artifacts
preprocessing:
  input_path: data.csv
  schema:
    target_column: load
    required_columns: invalid
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="Column lists"):
        load_preprocessing_config(config_path)
