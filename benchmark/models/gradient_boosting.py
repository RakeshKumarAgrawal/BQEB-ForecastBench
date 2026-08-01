"""Gradient boosting baseline model."""

from collections.abc import Mapping
from typing import Any, ClassVar, cast

from sklearn.ensemble import GradientBoostingRegressor

from benchmark.models._sklearn_model import Regressor, SklearnRegressorModel
from benchmark.models.registry import MODEL_REGISTRY


class GradientBoostingModel(SklearnRegressorModel):
    """Forecast with a scikit-learn gradient boosting regressor."""

    MODEL_NAME = "gradient_boosting"
    MODEL_VERSION = "1.0"
    DEFAULT_PARAMETERS: ClassVar[dict[str, Any]] = {
        "loss": "squared_error",
        "learning_rate": 0.1,
        "n_estimators": 100,
        "subsample": 1.0,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_depth": 3,
        "random_state": 42,
    }

    @staticmethod
    def _create_estimator(parameters: Mapping[str, Any]) -> Regressor:
        return cast(Regressor, GradientBoostingRegressor(**dict(parameters)))


MODEL_REGISTRY.register(GradientBoostingModel.MODEL_NAME, GradientBoostingModel)
