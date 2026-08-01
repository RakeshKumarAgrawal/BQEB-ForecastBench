"""Tests for the random forest baseline model."""

from pathlib import Path

import numpy as np

from benchmark.models import MODEL_REGISTRY, RandomForestModel, create_model


def test_random_forest_lifecycle_and_parameters() -> None:
    """The model should train and predict with configuration overrides."""
    features = np.arange(16, dtype=float).reshape(8, 2)
    target = np.arange(8, dtype=float)
    model = RandomForestModel(
        {"n_estimators": 8, "max_depth": 3, "random_state": 7, "n_jobs": 1}
    )

    predictions = model.fit(features, target).predict(features)

    assert predictions.shape == (8,)
    assert np.isfinite(predictions).all()
    assert model.get_parameters()["n_estimators"] == 8
    assert model.get_name() == "random_forest"


def test_random_forest_persistence(tmp_path: Path) -> None:
    """A fitted forest should survive a joblib round trip."""
    features = np.arange(12, dtype=float).reshape(6, 2)
    target = np.arange(6, dtype=float)
    model = RandomForestModel({"n_estimators": 5, "n_jobs": 1}).fit(features, target)

    restored = RandomForestModel.load(model.save(tmp_path / "forest.joblib"))

    assert np.allclose(restored.predict(features), model.predict(features))
    assert restored.get_parameters() == model.get_parameters()
    assert restored.get_version() == "1.0"


def test_random_forest_registry_and_factory() -> None:
    """The registered class should be constructible from repository defaults."""
    assert MODEL_REGISTRY.get("random_forest") is RandomForestModel

    model = create_model("random_forest")

    assert isinstance(model, RandomForestModel)
    assert model.get_parameters() == RandomForestModel.DEFAULT_PARAMETERS
