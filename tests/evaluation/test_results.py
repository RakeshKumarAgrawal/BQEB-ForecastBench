"""Tests for evaluation result validation and serialization."""

import json
from pathlib import Path

import pytest

from benchmark.evaluation.results import EvaluationResult, ResultSerializationError


def _result() -> EvaluationResult:
    return EvaluationResult(
        model_name="linear_regression",
        dataset="test",
        mae=0.5,
        rmse=0.6,
        mape=0.2,
        r2=0.9,
        prediction_count=4,
        evaluation_timestamp="2026-08-01T00:00:00+00:00",
        configuration_hash="abc123",
        repository_version="0.4.0",
        additional_metrics={"bias": -0.1},
    )


def test_result_round_trip_and_metric_access(tmp_path: Path) -> None:
    """Versioned JSON serialization should preserve every result field."""
    path = _result().to_json(tmp_path / "metrics.json")
    loaded = EvaluationResult.from_json(path)

    assert loaded == _result()
    assert loaded.get_metric("BIAS") == pytest.approx(-0.1)
    assert loaded.metrics["rmse"] == pytest.approx(0.6)


def test_result_rejects_invalid_values_and_schema(tmp_path: Path) -> None:
    """Invalid provenance and unsupported documents should fail clearly."""
    with pytest.raises(ValueError, match="model_name"):
        EvaluationResult(
            model_name="",
            dataset="test",
            mae=0.0,
            rmse=0.0,
            mape=0.0,
            r2=1.0,
            prediction_count=1,
            evaluation_timestamp="now",
            configuration_hash="hash",
            repository_version="version",
            additional_metrics={},
        )
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({"schema_version": 2}), encoding="utf-8")
    with pytest.raises(ResultSerializationError, match="Unsupported"):
        EvaluationResult.from_json(path)
