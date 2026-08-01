"""Shared adapter for scikit-learn forecasting regressors."""

from abc import abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar, Protocol

import numpy as np

from benchmark.models.base_model import (
    BaseForecastModel,
    FeatureMatrix,
    PredictionVector,
    TargetVector,
)


class Regressor(Protocol):
    """Structural interface required from wrapped regressors."""

    def fit(self, features: Any, target: Any) -> Any:
        """Fit estimator state."""

    def predict(self, features: Any) -> Any:
        """Return estimator predictions."""

    def set_params(self, **parameters: Any) -> Any:
        """Update estimator parameters."""


class SklearnRegressorModel(BaseForecastModel):
    """Centralize lifecycle delegation for scikit-learn regressors."""

    DEFAULT_PARAMETERS: ClassVar[dict[str, Any]] = {}

    def __init__(self, parameters: Mapping[str, Any] | None = None) -> None:
        """Initialize a wrapped estimator from defaults and overrides."""
        super().__init__(self.DEFAULT_PARAMETERS)
        if parameters is not None:
            self.set_parameters(parameters)
        self._estimator = self._create_estimator(self.get_parameters())

    def set_parameters(self, parameters: Mapping[str, Any]) -> None:
        """Validate parameters and synchronize the wrapped estimator."""
        super().set_parameters(parameters)
        if hasattr(self, "_estimator"):
            self._estimator.set_params(**self.get_parameters())

    def _validate_parameters(self, parameters: Mapping[str, Any]) -> None:
        self._create_estimator(parameters)

    def _fit(self, features: FeatureMatrix, target: TargetVector) -> None:
        self._estimator.fit(features, target)

    def _predict(self, features: FeatureMatrix) -> PredictionVector:
        return np.asarray(self._estimator.predict(features), dtype=float)

    @staticmethod
    @abstractmethod
    def _create_estimator(parameters: Mapping[str, Any]) -> Regressor:
        """Construct the model-specific scikit-learn estimator."""
