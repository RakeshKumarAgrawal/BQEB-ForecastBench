"""Public framework API for ForecastBench forecasting models."""

from benchmark.models.base_model import (
    BaseForecastModel,
    FeatureMatrix,
    ModelConfigurationError,
    ModelError,
    ModelInputError,
    ModelPersistenceError,
    ModelPredictionError,
    ModelTrainingError,
    PredictionVector,
    TargetVector,
)
from benchmark.models.factory import ModelFactoryError, create_model
from benchmark.models.gradient_boosting import GradientBoostingModel
from benchmark.models.linear_regression import LinearRegressionModel
from benchmark.models.random_forest import RandomForestModel
from benchmark.models.registry import (
    MODEL_REGISTRY,
    DuplicateModelError,
    ModelNotRegisteredError,
    ModelRegistry,
    ModelRegistryError,
)

__all__ = [
    "MODEL_REGISTRY",
    "BaseForecastModel",
    "DuplicateModelError",
    "FeatureMatrix",
    "GradientBoostingModel",
    "LinearRegressionModel",
    "ModelConfigurationError",
    "ModelError",
    "ModelFactoryError",
    "ModelInputError",
    "ModelNotRegisteredError",
    "ModelPersistenceError",
    "ModelPredictionError",
    "ModelRegistry",
    "ModelRegistryError",
    "ModelTrainingError",
    "PredictionVector",
    "RandomForestModel",
    "TargetVector",
    "create_model",
]
