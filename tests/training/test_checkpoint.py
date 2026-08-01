"""Tests for checkpoint persistence and discovery."""

from pathlib import Path

import numpy as np

from benchmark.constants import VERSION
from benchmark.models import LinearRegressionModel
from benchmark.training.checkpoint import (
    latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)


def test_checkpoint_round_trip_and_latest_discovery(tmp_path: Path) -> None:
    """Checkpoint records should preserve model state and required provenance."""
    features = np.array([[0.0], [1.0], [2.0]])
    target = np.array([1.0, 2.0, 3.0])
    model = LinearRegressionModel().fit(features, target)
    configuration = {"training": {"random_seed": 42}}

    path = save_checkpoint(model, configuration, tmp_path)
    checkpoint = load_checkpoint(path)

    assert latest_checkpoint(tmp_path) == path
    assert latest_checkpoint(tmp_path, "linear_regression") == path
    assert latest_checkpoint(tmp_path, "random_forest") is None
    assert np.allclose(checkpoint.model.predict(features), model.predict(features))
    assert checkpoint.configuration == configuration
    assert checkpoint.model_version == model.get_version()
    assert checkpoint.repository_version == VERSION
    assert checkpoint.timestamp
