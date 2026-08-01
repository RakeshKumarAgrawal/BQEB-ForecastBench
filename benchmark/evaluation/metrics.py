"""Validated wrappers for ForecastBench regression metrics."""

import numpy as np
from sklearn.metrics import mean_absolute_error as sklearn_mae
from sklearn.metrics import mean_absolute_percentage_error as sklearn_mape
from sklearn.metrics import r2_score as sklearn_r2
from sklearn.metrics import root_mean_squared_error as sklearn_rmse

from benchmark.evaluation.registry import METRIC_REGISTRY
from benchmark.models.base_model import TargetVector


class MetricInputError(ValueError):
    """Raised when metric inputs cannot produce a valid comparison."""


def mean_absolute_error(y_true: TargetVector, y_pred: TargetVector) -> float:
    """Return the arithmetic mean of absolute prediction errors."""
    actual, predicted = _validated_arrays(y_true, y_pred)
    return float(sklearn_mae(actual, predicted))


def root_mean_squared_error(y_true: TargetVector, y_pred: TargetVector) -> float:
    """Return the square root of the mean squared prediction error."""
    actual, predicted = _validated_arrays(y_true, y_pred)
    return float(sklearn_rmse(actual, predicted))


def mean_absolute_percentage_error(y_true: TargetVector, y_pred: TargetVector) -> float:
    """Return sklearn's relative MAPE value, where 1.0 represents 100 percent."""
    actual, predicted = _validated_arrays(y_true, y_pred)
    return float(sklearn_mape(actual, predicted))


def coefficient_of_determination(y_true: TargetVector, y_pred: TargetVector) -> float:
    """Return the coefficient of determination, commonly called R-squared."""
    actual, predicted = _validated_arrays(y_true, y_pred)
    if len(actual) < 2:
        raise MetricInputError("R-squared requires at least two predictions")
    return float(sklearn_r2(actual, predicted))


def _validated_arrays(
    y_true: TargetVector, y_pred: TargetVector
) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if actual.ndim != 1 or predicted.ndim != 1:
        raise MetricInputError("Metric inputs must be one-dimensional")
    if len(actual) == 0:
        raise MetricInputError("Metric inputs must not be empty")
    if len(actual) != len(predicted):
        raise MetricInputError("Metric inputs must contain the same number of values")
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(predicted)):
        raise MetricInputError("Metric inputs must contain only finite values")
    return actual, predicted


METRIC_REGISTRY.register("mae", mean_absolute_error)
METRIC_REGISTRY.register("rmse", root_mean_squared_error)
METRIC_REGISTRY.register("mape", mean_absolute_percentage_error)
METRIC_REGISTRY.register("r2", coefficient_of_determination)
