"""Tests for versioned trained-model persistence."""

from pathlib import Path

import numpy as np
import pytest

from benchmark.constants import VERSION
from benchmark.models import LinearRegressionModel
from benchmark.training.model_io import (
    ModelIOError,
    load_model,
    load_model_artifact,
    save_model,
)


def test_model_io_preserves_behavior_and_metadata(tmp_path: Path) -> None:
    """A versioned model envelope should preserve predictions and provenance."""
    features = np.array([[0.0], [1.0], [2.0]])
    target = np.array([1.0, 3.0, 5.0])
    model = LinearRegressionModel().fit(features, target)
    path = save_model(
        model,
        tmp_path / "models" / "linear.joblib",
        configuration={"random_seed": 42},
        metadata={"rows": 3},
    )

    restored = load_model(path)
    artifact = load_model_artifact(path)

    assert np.allclose(restored.predict(features), model.predict(features))
    assert artifact.configuration == {"random_seed": 42}
    assert artifact.metadata == {"rows": 3}
    assert artifact.model_version == model.get_version()
    assert artifact.repository_version == VERSION


def test_model_io_rejects_missing_artifact(tmp_path: Path) -> None:
    """Missing model artifacts should raise a stable persistence error."""
    with pytest.raises(ModelIOError, match="does not exist"):
        load_model(tmp_path / "missing.joblib")
