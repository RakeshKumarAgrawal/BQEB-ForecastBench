"""Exact-schema CSV exports for machine-readable benchmark outputs."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmark.evaluation.predictions import PredictionRecord
from benchmark.evaluation.results import EvaluationResult
from benchmark.utils.filesystem import ensure_directory


class BenchmarkExporter:
    """Write deterministic benchmark CSV files with stable column order."""

    def __init__(self, directory: Path) -> None:
        """Initialize exports under one resolved output directory."""
        self.directory = ensure_directory(directory)

    def export_metrics(
        self, results: Sequence[EvaluationResult], filename: str = "metrics.csv"
    ) -> Path:
        """Export one metrics row per model evaluation."""
        rows = (
            {
                "Model": result.model_name,
                "Dataset": result.dataset,
                "MAE": result.mae,
                "RMSE": result.rmse,
                "MAPE": result.mape,
                "R²": result.r2,
                "Samples": result.prediction_count,
                "Timestamp": result.evaluation_timestamp,
                "RepositoryVersion": result.repository_version,
            }
            for result in results
        )
        return self._write_csv(
            filename,
            (
                "Model",
                "Dataset",
                "MAE",
                "RMSE",
                "MAPE",
                "R²",
                "Samples",
                "Timestamp",
                "RepositoryVersion",
            ),
            rows,
        )

    def export_benchmark_results(
        self,
        rows: Iterable[Mapping[str, Any]],
        filename: str = "benchmark_results.csv",
    ) -> Path:
        """Export ranked benchmark metrics and runtime measurements."""
        return self._write_csv(
            filename,
            (
                "Rank",
                "Model",
                "MAE",
                "RMSE",
                "MAPE",
                "R²",
                "TrainingTime",
                "PredictionTime",
                "RepositoryVersion",
            ),
            rows,
        )

    def export_predictions(
        self,
        records: Sequence[PredictionRecord],
        filename: str = "predictions.csv",
    ) -> Path:
        """Export aligned actual and predicted values for all partitions."""
        rows = (
            {
                "SampleID": record.sample_id,
                "Timestamp": record.timestamp,
                "Model": record.model_name,
                "Actual": record.actual,
                "Predicted": record.predicted,
                "AbsoluteError": record.absolute_error,
            }
            for record in records
        )
        return self._write_csv(
            filename,
            ("SampleID", "Timestamp", "Model", "Actual", "Predicted", "AbsoluteError"),
            rows,
        )

    def export_model_comparison(
        self,
        rows: Iterable[Mapping[str, Any]],
        filename: str = "model_comparison.csv",
    ) -> Path:
        """Export best, worst, and complete model ranking for each metric."""
        return self._write_csv(
            filename,
            ("Metric", "BestModel", "BestValue", "WorstModel", "WorstValue", "Ranking"),
            rows,
        )

    def _write_csv(
        self,
        filename: str,
        fieldnames: tuple[str, ...],
        rows: Iterable[Mapping[str, Any]],
    ) -> Path:
        destination = self.directory / filename
        with destination.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
        return destination


def ranking_json(model_names: Sequence[str]) -> str:
    """Encode a model ranking as a compact machine-readable CSV cell."""
    return json.dumps(list(model_names), separators=(",", ":"))
