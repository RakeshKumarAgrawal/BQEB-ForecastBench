"""Tests for exact-schema benchmark CSV exports."""

import csv
from pathlib import Path

from benchmark.evaluation.exporter import BenchmarkExporter, ranking_json
from benchmark.evaluation.predictions import PredictionRecord
from benchmark.evaluation.results import EvaluationResult


def _fieldnames(path: Path) -> list[str] | None:
    with path.open(encoding="utf-8", newline="") as stream:
        return csv.DictReader(stream).fieldnames


def _result() -> EvaluationResult:
    return EvaluationResult(
        model_name="model",
        dataset="test",
        mae=1.0,
        rmse=1.1,
        mape=0.1,
        r2=0.9,
        prediction_count=2,
        evaluation_timestamp="2026-08-01T00:00:00+00:00",
        configuration_hash="hash",
        repository_version="0.4.0",
        additional_metrics={},
    )


def test_exporter_writes_prediction_schema(tmp_path: Path) -> None:
    """Prediction CSV columns should match the Batch 2 contract exactly."""
    exporter = BenchmarkExporter(tmp_path)
    path = exporter.export_predictions(
        [PredictionRecord("test-000000", "2026-01-01", "model", 2.0, 2.5)]
    )

    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)

    assert reader.fieldnames == [
        "SampleID",
        "Timestamp",
        "Model",
        "Actual",
        "Predicted",
        "AbsoluteError",
    ]
    assert rows[0]["AbsoluteError"] == "0.5"


def test_ranking_json_is_compact_and_ordered() -> None:
    """Comparison rankings should remain machine-readable inside CSV."""
    assert ranking_json(("first", "second")) == '["first","second"]'


def test_exporter_writes_all_requested_csv_schemas(tmp_path: Path) -> None:
    """Every benchmark CSV should preserve its prescribed column order."""
    exporter = BenchmarkExporter(tmp_path)
    metrics = exporter.export_metrics([_result()])
    benchmark = exporter.export_benchmark_results(
        [
            {
                "Rank": 1,
                "Model": "model",
                "MAE": 1.0,
                "RMSE": 1.1,
                "MAPE": 0.1,
                "R²": 0.9,
                "TrainingTime": 0.01,
                "PredictionTime": 0.001,
                "RepositoryVersion": "0.4.0",
            }
        ]
    )
    comparison = exporter.export_model_comparison(
        [
            {
                "Metric": "rmse",
                "BestModel": "model",
                "BestValue": 1.1,
                "WorstModel": "model",
                "WorstValue": 1.1,
                "Ranking": '["model"]',
            }
        ]
    )

    assert _fieldnames(metrics) == [
        "Model",
        "Dataset",
        "MAE",
        "RMSE",
        "MAPE",
        "R²",
        "Samples",
        "Timestamp",
        "RepositoryVersion",
    ]
    assert _fieldnames(benchmark) == [
        "Rank",
        "Model",
        "MAE",
        "RMSE",
        "MAPE",
        "R²",
        "TrainingTime",
        "PredictionTime",
        "RepositoryVersion",
    ]
    assert _fieldnames(comparison) == [
        "Metric",
        "BestModel",
        "BestValue",
        "WorstModel",
        "WorstValue",
        "Ranking",
    ]
