"""Tests for configuration-driven benchmark evaluation."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from benchmark.constants import CONFIG_DIR, VERSION
from benchmark.evaluation import BenchmarkEvaluator, load_evaluation_settings
from benchmark.evaluation.evaluator import EvaluationConfigurationError
from benchmark.models import LinearRegressionModel


def _write_config(path: Path, artifact_directory: Path) -> Path:
    values: dict[str, Any] = {
        "evaluation": {
            "dataset": "unit_test",
            "enabled_metrics": ["mae", "rmse", "mape", "r2"],
            "options": {"primary_metric": "rmse", "persist_artifacts": True},
            "prediction_output": {
                "enabled": False,
                "include_in_evaluation_log": False,
            },
            "artifact_locations": {
                "evaluation": str(artifact_directory),
                "metrics_filename": "metrics.json",
                "log_filename": "evaluation_log.json",
            },
            "random_seed": 7,
            "logging": {"level": "INFO", "console": True},
        }
    }
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    return path


def test_evaluator_computes_metrics_and_writes_json_artifacts(tmp_path: Path) -> None:
    """A trained model should produce a complete result and two JSON artifacts."""
    features = np.array([[0.0], [1.0], [2.0], [3.0]])
    target = np.array([1.0, 3.0, 5.0, 7.0])
    model = LinearRegressionModel().fit(features, target)
    artifact_directory = tmp_path / "evaluation"
    evaluator = BenchmarkEvaluator(
        _write_config(tmp_path / "evaluation.yaml", artifact_directory)
    )

    result = evaluator.evaluate(model, features, target)

    assert result.model_name == "linear_regression"
    assert result.dataset == "unit_test"
    assert result.mae == pytest.approx(0.0, abs=1e-12)
    assert result.rmse == pytest.approx(0.0, abs=1e-12)
    assert result.mape == pytest.approx(0.0, abs=1e-12)
    assert result.r2 == pytest.approx(1.0)
    assert result.prediction_count == 4
    assert result.repository_version == VERSION
    assert len(result.configuration_hash) == 64
    assert (artifact_directory / "metrics.json").is_file()
    log = json.loads(
        (artifact_directory / "evaluation_log.json").read_text(encoding="utf-8")
    )
    assert len(log["records"]) == 1
    assert "predictions" not in log["records"][0]


def test_repository_evaluation_configuration_loads() -> None:
    """The checked-in evaluation configuration should expose all required settings."""
    settings = load_evaluation_settings(CONFIG_DIR / "evaluation.yaml")

    assert settings.enabled_metrics == ("mae", "rmse", "mape", "r2")
    assert settings.primary_metric == "rmse"
    assert settings.experiment_name == "forecastbench-baselines"
    assert settings.artifacts.metrics_filename == "metrics.json"
    assert settings.artifacts.log_filename == "evaluation_log.json"
    assert settings.artifacts.metrics_csv_filename == "metrics.csv"
    assert settings.artifacts.manifest_filename == "experiment_manifest.json"
    assert settings.benchmark_models.names is None
    assert settings.benchmark_models.source == "fit_from_training_split"
    assert set(settings.benchmark_dataset.split_paths) == {
        "train",
        "validation",
        "test",
    }


def test_configuration_rejects_primary_metric_not_enabled(tmp_path: Path) -> None:
    """The comparison metric must be computed by the evaluator."""
    path = _write_config(tmp_path / "evaluation.yaml", tmp_path / "artifacts")
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    values["evaluation"]["options"]["primary_metric"] = "bias"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")

    with pytest.raises(EvaluationConfigurationError, match="included"):
        load_evaluation_settings(path)
