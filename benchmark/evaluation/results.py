"""Structured evaluation results and JSON artifact persistence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self, cast

import numpy as np

from benchmark.utils.filesystem import ensure_directory

RESULT_SCHEMA_VERSION = 1


class ResultSerializationError(ValueError):
    """Raised when an evaluation result artifact is invalid."""


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Capture metrics and provenance for one model evaluation."""

    model_name: str
    dataset: str
    mae: float | None
    rmse: float | None
    mape: float | None
    r2: float | None
    prediction_count: int
    evaluation_timestamp: str
    configuration_hash: str
    repository_version: str
    additional_metrics: dict[str, float]

    def __post_init__(self) -> None:
        """Validate result metadata and metric values."""
        for name, value in (
            ("model_name", self.model_name),
            ("dataset", self.dataset),
            ("evaluation_timestamp", self.evaluation_timestamp),
            ("configuration_hash", self.configuration_hash),
            ("repository_version", self.repository_version),
        ):
            if not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.prediction_count < 0:
            raise ValueError("prediction_count must be non-negative")
        values = [self.mae, self.rmse, self.mape, self.r2]
        values.extend(self.additional_metrics.values())
        if any(value is not None and not np.isfinite(value) for value in values):
            raise ValueError("Metric values must be finite")
        object.__setattr__(self, "additional_metrics", dict(self.additional_metrics))

    @property
    def metrics(self) -> dict[str, float]:
        """Return all computed metrics as a defensive mapping."""
        values = {
            name: value
            for name, value in (
                ("mae", self.mae),
                ("rmse", self.rmse),
                ("mape", self.mape),
                ("r2", self.r2),
            )
            if value is not None
        }
        values.update(self.additional_metrics)
        return values

    def get_metric(self, name: str) -> float:
        """Return one computed metric by normalized name."""
        normalized = name.strip().lower()
        try:
            return self.metrics[normalized]
        except KeyError as error:
            raise KeyError(
                f"Metric {normalized!r} is absent from this result"
            ) from error

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(self)

    def to_json(self, path: Path) -> Path:
        """Serialize this result to a versioned JSON document."""
        destination = ensure_directory(path.expanduser().parent) / path.name
        payload = {"schema_version": RESULT_SCHEMA_VERSION, "result": self.to_dict()}
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return destination

    @classmethod
    def from_json(cls, path: Path) -> Self:
        """Load a result from a versioned JSON document."""
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ResultSerializationError(f"Unable to read result: {path}") from error
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ResultSerializationError("Unsupported evaluation result schema")
        result = payload.get("result")
        if not isinstance(result, dict):
            raise ResultSerializationError("Evaluation result must be a mapping")
        try:
            return cls(**cast(dict[str, Any], result))
        except (TypeError, ValueError) as error:
            raise ResultSerializationError(
                "Invalid evaluation result values"
            ) from error


def write_evaluation_artifacts(
    result: EvaluationResult,
    directory: Path,
    *,
    metrics_filename: str,
    log_filename: str,
    predictions: np.ndarray | None = None,
) -> tuple[Path, Path]:
    """Write latest metrics and append to the versioned evaluation log."""
    destination = ensure_directory(directory)
    metrics_path = result.to_json(destination / metrics_filename)
    log_path = destination / log_filename
    records: list[dict[str, Any]] = []
    if log_path.is_file():
        try:
            existing = json.loads(log_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ResultSerializationError(
                f"Unable to read evaluation log: {log_path}"
            ) from error
        if not isinstance(existing, dict) or existing.get("schema_version") != 1:
            raise ResultSerializationError("Unsupported evaluation log schema")
        loaded_records = existing.get("records")
        if not isinstance(loaded_records, list) or not all(
            isinstance(record, dict) for record in loaded_records
        ):
            raise ResultSerializationError("Evaluation log records must be mappings")
        records = cast(list[dict[str, Any]], loaded_records)
    record = result.to_dict()
    if predictions is not None:
        record["predictions"] = predictions.tolist()
    records.append(record)
    log_path.write_text(
        json.dumps(
            {"schema_version": RESULT_SCHEMA_VERSION, "records": records},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return metrics_path, log_path
