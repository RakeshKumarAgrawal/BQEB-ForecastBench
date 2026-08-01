"""Random forest baseline model."""

from collections.abc import Mapping
from typing import Any, ClassVar, cast

from sklearn.ensemble import RandomForestRegressor

from benchmark.models._sklearn_model import Regressor, SklearnRegressorModel
from benchmark.models.registry import MODEL_REGISTRY


class RandomForestModel(SklearnRegressorModel):
    """Forecast with a scikit-learn random forest regressor."""

    MODEL_NAME = "random_forest"
    MODEL_VERSION = "1.0"
    DEFAULT_PARAMETERS: ClassVar[dict[str, Any]] = {
        "n_estimators": 200,
        "criterion": "squared_error",
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": 1.0,
        "bootstrap": True,
        "n_jobs": -1,
        "random_state": 42,
    }

    @staticmethod
    def _create_estimator(parameters: Mapping[str, Any]) -> Regressor:
        return cast(Regressor, RandomForestRegressor(**dict(parameters)))


MODEL_REGISTRY.register(RandomForestModel.MODEL_NAME, RandomForestModel)
