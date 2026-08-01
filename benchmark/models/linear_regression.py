"""Linear regression baseline model."""

from collections.abc import Mapping
from typing import Any, ClassVar, cast

from sklearn.linear_model import LinearRegression

from benchmark.models._sklearn_model import Regressor, SklearnRegressorModel
from benchmark.models.registry import MODEL_REGISTRY


class LinearRegressionModel(SklearnRegressorModel):
    """Forecast with scikit-learn ordinary least squares regression."""

    MODEL_NAME = "linear_regression"
    MODEL_VERSION = "1.0"
    DEFAULT_PARAMETERS: ClassVar[dict[str, Any]] = {
        "fit_intercept": True,
        "copy_X": True,
        "n_jobs": None,
        "positive": False,
    }

    @staticmethod
    def _create_estimator(parameters: Mapping[str, Any]) -> Regressor:
        return cast(Regressor, LinearRegression(**dict(parameters)))


MODEL_REGISTRY.register(LinearRegressionModel.MODEL_NAME, LinearRegressionModel)
