"""Tests for registry-driven benchmark orchestration and exports."""

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from benchmark.evaluation.benchmark import BenchmarkRunner
from benchmark.models import MODEL_REGISTRY, LinearRegressionModel
from benchmark.training import save_model


def _write_split(path: Path, start: int, count: int) -> Path:
    values = list(range(start, start + count))
    pd.DataFrame(
        {
            "timestamp": [f"2026-01-01T{value:02d}:00:00Z" for value in values],
            "target": [float(2 * value + 1) for value in values],
            "feature": [float(value) for value in values],
        }
    ).to_csv(path, index=False)
    return path


def _write_benchmark_config(
    path: Path,
    root: Path,
    *,
    selection: str | list[str] = "all_registered",
    source: str = "fit_from_training_split",
) -> Path:
    values: dict[str, Any] = {
        "evaluation": {
            "dataset": "unit_benchmark",
            "experiment_name": "unit-baselines",
            "enabled_metrics": ["mae", "rmse", "mape", "r2"],
            "options": {"primary_metric": "rmse", "persist_artifacts": False},
            "prediction_output": {
                "enabled": False,
                "include_in_evaluation_log": False,
            },
            "artifact_locations": {
                "evaluation": str(root / "evaluation"),
                "experiments": str(root / "experiments"),
                "metrics_filename": "metrics.json",
                "log_filename": "evaluation_log.json",
                "metrics_csv_filename": "metrics.csv",
                "benchmark_results_filename": "benchmark_results.csv",
                "predictions_filename": "predictions.csv",
                "model_comparison_filename": "model_comparison.csv",
                "manifest_filename": "experiment_manifest.json",
            },
            "dataset_splits": {
                "paths": {
                    "train": str(_write_split(root / "train.csv", 0, 6)),
                    "validation": str(_write_split(root / "validation.csv", 6, 2)),
                    "test": str(_write_split(root / "test.csv", 8, 2)),
                },
                "target_column": "target",
                "timestamp_column": "timestamp",
                "sample_id_column": None,
            },
            "models": {
                "selection": selection,
                "source": source,
                "artifact_directory": str(root / "models"),
            },
            "random_seed": 42,
            "logging": {"level": "INFO", "console": True},
        }
    }
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    return path


def test_runner_executes_all_registered_models_and_exports(tmp_path: Path) -> None:
    """All registered baselines should run without hardcoded model selection."""
    runner = BenchmarkRunner(
        _write_benchmark_config(tmp_path / "config.yaml", tmp_path)
    )

    runs = runner.run_all_models()

    assert tuple(run.model_name for run in runs) == MODEL_REGISTRY.list_models()
    assert all(len(run.results) == 3 for run in runs)
    output = tmp_path / "evaluation"
    assert {path.name for path in output.glob("*.csv")} == {
        "metrics.csv",
        "benchmark_results.csv",
        "predictions.csv",
        "model_comparison.csv",
    }
    with (output / "benchmark_results.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(MODEL_REGISTRY.list_models())
    assert [int(row["Rank"]) for row in rows] == sorted(
        int(row["Rank"]) for row in rows
    )
    manifest = json.loads(
        (tmp_path / "experiments" / "experiment_manifest.json").read_text(
            encoding="utf-8"
        )
    )["manifest"]
    assert set(manifest["model_versions"]) == set(MODEL_REGISTRY.list_models())
    assert len(manifest["dataset_fingerprint"]) == 64


def test_runner_loads_trained_artifact_for_single_model(tmp_path: Path) -> None:
    """Artifact model source should execute without fitting the model again."""
    model_name = "linear_regression"
    model = LinearRegressionModel().fit(
        np.array([[0.0], [1.0], [2.0]]), np.array([1.0, 3.0, 5.0])
    )
    save_model(model, tmp_path / "models" / f"{model_name}-test.joblib")
    config = _write_benchmark_config(
        tmp_path / "config.yaml",
        tmp_path,
        selection=[model_name],
        source="artifacts",
    )

    run = BenchmarkRunner(config).run(model_name)

    assert run.model_name == model_name
    assert run.training_time == 0.0
