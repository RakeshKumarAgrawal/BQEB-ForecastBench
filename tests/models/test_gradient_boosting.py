"""Tests for the gradient boosting baseline model."""

from pathlib import Path

import numpy as np

from benchmark.models import MODEL_REGISTRY, GradientBoostingModel, create_model


def test_gradient_boosting_lifecycle_and_parameters() -> None:
    """The model should train and predict with configuration overrides."""
    features = np.arange(20, dtype=float).reshape(10, 2)
    target = np.arange(10, dtype=float)
    model = GradientBoostingModel(
        {
            "learning_rate": 0.05,
            "n_estimators": 10,
            "subsample": 0.8,
            "max_depth": 2,
            "random_state": 7,
        }
    )

    predictions = model.fit(features, target).predict(features)

    assert predictions.shape == (10,)
    assert np.isfinite(predictions).all()
    assert model.get_parameters()["learning_rate"] == 0.05
    assert model.get_name() == "gradient_boosting"


def test_gradient_boosting_persistence(tmp_path: Path) -> None:
    """A fitted boosting model should survive a joblib round trip."""
    features = np.arange(16, dtype=float).reshape(8, 2)
    target = np.arange(8, dtype=float)
    model = GradientBoostingModel({"n_estimators": 5}).fit(features, target)

    restored = GradientBoostingModel.load(
        model.save(tmp_path / "gradient-boosting.joblib")
    )

    assert np.allclose(restored.predict(features), model.predict(features))
    assert restored.get_parameters() == model.get_parameters()
    assert restored.get_version() == "1.0"


def test_gradient_boosting_registry_and_factory() -> None:
    """All baselines should be registered and constructible from real YAML."""
    assert MODEL_REGISTRY.list_models() == (
        "gradient_boosting",
        "linear_regression",
        "random_forest",
    )

    model = create_model("gradient_boosting")

    assert isinstance(model, GradientBoostingModel)
    assert model.get_parameters() == GradientBoostingModel.DEFAULT_PARAMETERS
