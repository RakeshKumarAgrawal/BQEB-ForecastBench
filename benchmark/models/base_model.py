"""Abstract contract shared by all ForecastBench forecasting models."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, ClassVar, Self

import joblib
import numpy as np
import pandas as pd

from benchmark.utils.filesystem import ensure_directory

LOGGER = logging.getLogger(__name__)

type FeatureMatrix = pd.DataFrame | np.ndarray
type TargetVector = pd.Series | np.ndarray
type PredictionVector = np.ndarray


class ModelError(RuntimeError):
    """Base exception for errors raised by the forecasting model framework."""


class ModelInputError(ModelError, ValueError):
    """Raised when model input does not satisfy the framework contract."""


class ModelConfigurationError(ModelError, ValueError):
    """Raised when model parameters or metadata are invalid."""


class ModelPersistenceError(ModelError):
    """Raised when a model cannot be saved or loaded safely."""


class ModelTrainingError(ModelError):
    """Raised when a model implementation fails during fitting."""


class ModelPredictionError(ModelError):
    """Raised when a model implementation fails during prediction."""


class BaseForecastModel(ABC):
    """Define validated lifecycle operations for forecasting implementations."""

    MODEL_NAME: ClassVar[str] = ""
    MODEL_VERSION: ClassVar[str] = ""

    def __init__(self, parameters: Mapping[str, Any] | None = None) -> None:
        """Initialize a model with a defensive copy of its parameters."""
        self._parameters: dict[str, Any] = {}
        if parameters is not None:
            self.set_parameters(parameters)

    def fit(self, features: FeatureMatrix, target: TargetVector) -> Self:
        """Validate inputs, fit implementation state, and return this model."""
        LOGGER.info("Fitting model name=%s", self.get_name())
        try:
            self.validate_input(features, target)
            self._fit(features, target)
        except ModelError:
            LOGGER.exception("Model fitting rejected name=%s", self.get_name())
            raise
        except Exception as error:
            LOGGER.exception("Model fitting failed name=%s", self.get_name())
            raise ModelTrainingError(
                f"Failed to fit model {self.get_name()!r}"
            ) from error
        LOGGER.info("Model fitting complete name=%s", self.get_name())
        return self

    def predict(self, features: FeatureMatrix) -> PredictionVector:
        """Validate features and return implementation predictions."""
        LOGGER.info("Generating predictions model=%s", self.get_name())
        try:
            self.validate_input(features)
            predictions = np.asarray(self._predict(features))
            if predictions.ndim != 1 or len(predictions) != len(features):
                raise ModelPredictionError(
                    "Predictions must be one-dimensional and match input rows"
                )
        except ModelError:
            LOGGER.exception("Prediction rejected model=%s", self.get_name())
            raise
        except Exception as error:
            LOGGER.exception("Prediction failed model=%s", self.get_name())
            raise ModelPredictionError(
                f"Failed to predict with model {self.get_name()!r}"
            ) from error
        LOGGER.info(
            "Prediction complete model=%s rows=%d", self.get_name(), len(predictions)
        )
        return predictions

    def save(self, path: Path) -> Path:
        """Serialize the model to ``path`` and return the resolved destination."""
        try:
            destination = ensure_directory(path.expanduser().parent) / path.name
            LOGGER.info("Saving model name=%s path=%s", self.get_name(), destination)
            joblib.dump(self, destination)
        except Exception as error:
            LOGGER.exception("Model save failed requested_path=%s", path)
            raise ModelPersistenceError(f"Unable to save model to: {path}") from error
        return destination

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load a model from a trusted joblib file and validate its type."""
        source = path.expanduser().resolve()
        LOGGER.info("Loading model class=%s path=%s", cls.__name__, source)
        if not source.is_file():
            raise ModelPersistenceError(f"Model file does not exist: {source}")
        try:
            model = joblib.load(source)
        except Exception as error:
            LOGGER.exception("Model load failed path=%s", source)
            raise ModelPersistenceError(
                f"Unable to load model from: {source}"
            ) from error
        if not isinstance(model, cls):
            raise ModelPersistenceError(
                f"Serialized object is not an instance of {cls.__name__}"
            )
        LOGGER.info("Loaded model name=%s path=%s", model.get_name(), source)
        return model

    def get_name(self) -> str:
        """Return the stable registry name declared by the model class."""
        name = self.MODEL_NAME.strip()
        if not name:
            LOGGER.error("Model class has no MODEL_NAME class=%s", type(self).__name__)
            raise ModelConfigurationError("MODEL_NAME must be a non-empty string")
        LOGGER.debug("Resolved model name=%s", name)
        return name

    def get_version(self) -> str:
        """Return the implementation version declared by the model class."""
        version = self.MODEL_VERSION.strip()
        if not version:
            LOGGER.error(
                "Model class has no MODEL_VERSION class=%s", type(self).__name__
            )
            raise ModelConfigurationError("MODEL_VERSION must be a non-empty string")
        LOGGER.debug("Resolved model version=%s name=%s", version, self.get_name())
        return version

    def get_parameters(self) -> dict[str, Any]:
        """Return a defensive copy of the current model parameters."""
        LOGGER.debug("Reading parameters model=%s", self.get_name())
        return deepcopy(self._parameters)

    def set_parameters(self, parameters: Mapping[str, Any]) -> None:
        """Validate and merge model parameters into the current configuration."""
        LOGGER.debug("Setting parameters model_class=%s", type(self).__name__)
        if not isinstance(parameters, Mapping):
            raise ModelConfigurationError("parameters must be a mapping")
        if not all(isinstance(name, str) and name.strip() for name in parameters):
            raise ModelConfigurationError("parameter names must be non-empty strings")
        try:
            updated = deepcopy({**self._parameters, **parameters})
            self._validate_parameters(updated)
            self._parameters = updated
        except ModelError:
            LOGGER.exception(
                "Parameter update rejected model_class=%s", type(self).__name__
            )
            raise
        except Exception as error:
            LOGGER.exception(
                "Parameter update failed model_class=%s", type(self).__name__
            )
            raise ModelConfigurationError(
                "Unable to update model parameters"
            ) from error

    def validate_input(
        self,
        features: FeatureMatrix,
        target: TargetVector | None = None,
    ) -> None:
        """Validate feature dimensions, target dimensions, and row alignment."""
        LOGGER.debug("Validating model input model_class=%s", type(self).__name__)
        if not isinstance(features, pd.DataFrame | np.ndarray):
            raise ModelInputError("features must be a pandas DataFrame or NumPy array")
        if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
            raise ModelInputError("features must be a non-empty two-dimensional matrix")
        if target is None:
            return
        if not isinstance(target, pd.Series | np.ndarray):
            raise ModelInputError("target must be a pandas Series or NumPy array")
        if target.ndim != 1:
            raise ModelInputError("target must be one-dimensional")
        if len(target) != len(features):
            raise ModelInputError("features and target must contain the same rows")

    def _validate_parameters(self, parameters: Mapping[str, Any]) -> None:
        """Allow subclasses to reject unsupported parameter combinations."""
        return None

    @abstractmethod
    def _fit(self, features: FeatureMatrix, target: TargetVector) -> None:
        """Fit implementation-specific state using validated inputs."""

    @abstractmethod
    def _predict(self, features: FeatureMatrix) -> PredictionVector:
        """Generate implementation-specific predictions for validated features."""
