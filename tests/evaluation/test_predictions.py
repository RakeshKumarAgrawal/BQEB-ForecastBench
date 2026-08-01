"""Tests for prediction record construction and validation."""

import numpy as np
import pytest

from benchmark.evaluation.predictions import PredictionRecord


def test_prediction_records_include_partition_identifiers_and_errors() -> None:
    """Aligned arrays should produce stable identifiers and absolute errors."""
    records = PredictionRecord.from_values(
        partition="validation",
        model_name="linear_regression",
        actual=np.array([2.0, 4.0]),
        predicted=np.array([2.5, 3.0]),
        timestamps=("2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"),
    )

    assert [record.sample_id for record in records] == [
        "validation-000000",
        "validation-000001",
    ]
    assert [record.absolute_error for record in records] == [0.5, 1.0]


def test_prediction_records_reject_misaligned_values() -> None:
    """Prediction values and metadata must preserve row alignment."""
    with pytest.raises(ValueError, match="equal lengths"):
        PredictionRecord.from_values(
            partition="test",
            model_name="model",
            actual=np.array([1.0]),
            predicted=np.array([1.0, 2.0]),
            timestamps=("2026-01-01T00:00:00Z",),
        )
