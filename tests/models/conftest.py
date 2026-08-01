"""Shared fixtures for model framework tests."""

from collections.abc import Iterator

import numpy as np
import pytest

from benchmark.models.base_model import (
    BaseForecastModel,
    FeatureMatrix,
    PredictionVector,
    TargetVector,
)
from benchmark.models.registry import ModelRegistry


class RegisteredStubModel(BaseForecastModel):
    """Non-algorithmic test model used by registry and factory tests."""

    MODEL_NAME = "registered_stub"
    MODEL_VERSION = "1.0"

    def _fit(self, features: FeatureMatrix, target: TargetVector) -> None:
        return None

    def _predict(self, features: FeatureMatrix) -> PredictionVector:
        return np.zeros(len(features))


@pytest.fixture
def registry() -> Iterator[ModelRegistry]:
    """Provide an isolated model registry for each test."""
    instance = ModelRegistry()
    yield instance
    instance.clear()
