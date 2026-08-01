"""Tests for validated forecasting metric wrappers."""

import numpy as np
import pytest

from benchmark.evaluation.metrics import (
    MetricInputError,
    coefficient_of_determination,
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)


@pytest.fixture
def targets() -> tuple[np.ndarray, np.ndarray]:
    """Return a known regression example for metric checks."""
    return np.array([3.0, -0.5, 2.0, 7.0]), np.array([2.5, 0.0, 2.0, 8.0])


def test_mean_absolute_error(targets: tuple[np.ndarray, np.ndarray]) -> None:
    """MAE should average absolute residual magnitudes."""
    assert mean_absolute_error(*targets) == pytest.approx(0.5)


def test_root_mean_squared_error(targets: tuple[np.ndarray, np.ndarray]) -> None:
    """RMSE should preserve the target's unit scale."""
    assert root_mean_squared_error(*targets) == pytest.approx(np.sqrt(0.375))


def test_mean_absolute_percentage_error(
    targets: tuple[np.ndarray, np.ndarray],
) -> None:
    """MAPE should use sklearn's relative rather than percentage-point scale."""
    assert mean_absolute_percentage_error(*targets) == pytest.approx(0.3273809524)


def test_coefficient_of_determination(
    targets: tuple[np.ndarray, np.ndarray],
) -> None:
    """R-squared should measure explained target variance."""
    assert coefficient_of_determination(*targets) == pytest.approx(0.9486081370)


@pytest.mark.parametrize(
    ("actual", "predicted", "message"),
    [
        (np.array([]), np.array([]), "must not be empty"),
        (np.array([1.0]), np.array([1.0, 2.0]), "same number"),
        (np.array([[1.0]]), np.array([1.0]), "one-dimensional"),
        (np.array([np.nan]), np.array([1.0]), "finite"),
    ],
)
def test_metrics_reject_invalid_inputs(
    actual: np.ndarray, predicted: np.ndarray, message: str
) -> None:
    """All wrappers should reject malformed arrays consistently."""
    with pytest.raises(MetricInputError, match=message):
        mean_absolute_error(actual, predicted)


def test_r_squared_requires_two_predictions() -> None:
    """R-squared should reject undefined single-observation evaluations."""
    with pytest.raises(MetricInputError, match="at least two"):
        coefficient_of_determination(np.array([1.0]), np.array([1.0]))
