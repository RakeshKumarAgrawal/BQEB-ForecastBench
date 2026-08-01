"""Tests for the linear regression baseline model."""

from pathlib import Path

import numpy as np
import pytest

from benchmark.models import (
    MODEL_REGISTRY,
    LinearRegressionModel,
    ModelConfigurationError,
    create_model,
)


def test_linear_regression_lifecycle_and_parameters() -> None:
    """The model should fit, predict, and synchronize parameter updates."""
    features = np.array([[0.0], [1.0], [2.0], [3.0]])
    target = np.array([1.0, 3.0, 5.0, 7.0])
    model = LinearRegressionModel({"fit_intercept": True})

    model.set_parameters({"positive": False})
    predictions = model.fit(features, target).predict(features)

    assert predictions.shape == (4,)
    assert np.allclose(predictions, target)
    assert model.get_name() == "linear_regression"
    assert model.get_version() == "1.0"
    assert model.get_parameters()["fit_intercept"] is True


def test_linear_regression_persistence(tmp_path: Path) -> None:
    """A fitted model should preserve predictions and metadata in joblib."""
    features = np.array([[0.0], [1.0], [2.0]])
    target = np.array([1.0, 2.0, 3.0])
    model = LinearRegressionModel().fit(features, target)

    restored = LinearRegressionModel.load(model.save(tmp_path / "linear.joblib"))

    assert np.allclose(restored.predict(features), model.predict(features))
    assert restored.get_parameters() == model.get_parameters()
    assert restored.get_version() == model.get_version()


def test_linear_regression_registry_and_factory() -> None:
    """Automatic registration should let the real YAML factory create the model."""
    assert MODEL_REGISTRY.get("linear_regression") is LinearRegressionModel

    model = create_model("linear_regression")

    assert isinstance(model, LinearRegressionModel)
    assert model.get_parameters() == LinearRegressionModel.DEFAULT_PARAMETERS


def test_linear_regression_rejects_invalid_parameters() -> None:
    """Concrete models should preserve stable base configuration errors."""
    with pytest.raises(ModelConfigurationError, match="must be a mapping"):
        LinearRegressionModel([("positive", True)])  # type: ignore[arg-type]

    with pytest.raises(ModelConfigurationError, match="Unable to update"):
        LinearRegressionModel({"unsupported_parameter": True})
