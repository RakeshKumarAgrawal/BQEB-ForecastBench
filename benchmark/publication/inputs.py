"""Strict loading and validation of Batch 2 publication source artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd

from benchmark.constants import PROJECT_ROOT

METRICS_COLUMNS = (
    "Model",
    "Dataset",
    "MAE",
    "RMSE",
    "MAPE",
    "R²",
    "Samples",
    "Timestamp",
    "RepositoryVersion",
)
BENCHMARK_COLUMNS = (
    "Rank",
    "Model",
    "MAE",
    "RMSE",
    "MAPE",
    "R²",
    "TrainingTime",
    "PredictionTime",
    "RepositoryVersion",
)
PREDICTION_COLUMNS = (
    "SampleID",
    "Timestamp",
    "Model",
    "Actual",
    "Predicted",
    "AbsoluteError",
)
COMPARISON_COLUMNS = (
    "Metric",
    "BestModel",
    "BestValue",
    "WorstModel",
    "WorstValue",
    "Ranking",
)


class PublicationInputError(ValueError):
    """Raised when a required Batch 2 artifact is missing or malformed."""


@dataclass(frozen=True, slots=True)
class PublicationInputPaths:
    """The complete allowlist of source files accepted by Batch 3."""

    metrics: Path
    benchmark_results: Path
    predictions: Path
    model_comparison: Path
    experiment_manifest: Path

    @classmethod
    def repository_defaults(cls) -> PublicationInputPaths:
        """Return the five immutable repository artifact locations."""
        evaluation = PROJECT_ROOT / "artifacts" / "evaluation"
        return cls(
            metrics=evaluation / "metrics.csv",
            benchmark_results=evaluation / "benchmark_results.csv",
            predictions=evaluation / "predictions.csv",
            model_comparison=evaluation / "model_comparison.csv",
            experiment_manifest=(
                PROJECT_ROOT / "artifacts" / "experiments" / "experiment_manifest.json"
            ),
        )


@dataclass(frozen=True, slots=True)
class PublicationData:
    """Validated in-memory view of all publication source artifacts."""

    metrics: pd.DataFrame
    benchmark_results: pd.DataFrame
    predictions: pd.DataFrame
    model_comparison: pd.DataFrame
    manifest: dict[str, Any]
    paths: PublicationInputPaths

    @classmethod
    def load(cls, paths: PublicationInputPaths | None = None) -> PublicationData:
        """Load and validate only the five allowlisted Batch 2 artifacts."""
        sources = paths or PublicationInputPaths.repository_defaults()
        metrics = _read_csv(sources.metrics, METRICS_COLUMNS)
        benchmark = _read_csv(sources.benchmark_results, BENCHMARK_COLUMNS)
        predictions = _read_csv(sources.predictions, PREDICTION_COLUMNS)
        comparison = _read_csv(sources.model_comparison, COMPARISON_COLUMNS)
        manifest = _read_manifest(sources.experiment_manifest)
        return cls(metrics, benchmark, predictions, comparison, manifest, sources)


def _read_csv(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    if not path.is_file():
        raise PublicationInputError(f"Publication source does not exist: {path}")
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as error:
        raise PublicationInputError(
            f"Unable to read publication source: {path}"
        ) from error
    if tuple(frame.columns) != columns:
        raise PublicationInputError(f"Unexpected columns in publication source: {path}")
    if frame.empty:
        raise PublicationInputError(f"Publication source is empty: {path}")
    return frame


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PublicationInputError(f"Publication source does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PublicationInputError(
            f"Unable to read publication source: {path}"
        ) from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise PublicationInputError("Unsupported experiment manifest schema")
    manifest = payload.get("manifest")
    if not isinstance(manifest, dict):
        raise PublicationInputError("Experiment manifest must be a mapping")
    required = {
        "repository_version",
        "configuration_hash",
        "git_commit",
        "dataset_fingerprint",
    }
    if not required.issubset(manifest):
        raise PublicationInputError("Experiment manifest lacks traceability fields")
    return cast(dict[str, Any], manifest)
