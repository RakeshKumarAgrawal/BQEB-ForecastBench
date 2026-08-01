"""Tests for the abstract forecasting model lifecycle."""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from benchmark.models.base_model import (
    BaseForecastModel,
    FeatureMatrix,
    ModelConfigurationError,
    ModelInputError,
    ModelPersistenceError,
    ModelPredictionError,
    ModelTrainingError,
    PredictionVector,
    TargetVector,
)


class StubModel(BaseForecastModel):
    """Small test implementation with no forecasting algorithm."""

    MODEL_NAME = "stub"
    MODEL_VERSION = "1.0"

    def _fit(self, features: FeatureMatrix, target: TargetVector) -> None:
        self.rows_seen = len(features)

    def _predict(self, features: FeatureMatrix) -> PredictionVector:
        return np.zeros(len(features), dtype=float)


def test_base_model_is_abstract() -> None:
    """The base contract should not be directly instantiable."""
    with pytest.raises(TypeError):
        BaseForecastModel()  # type: ignore[abstract]


def test_fit_predict_metadata_and_logging(caplog: pytest.LogCaptureFixture) -> None:
    """Public lifecycle methods should validate, delegate, and log."""
    model = StubModel({"alpha": 1})
    features = pd.DataFrame({"value": [1.0, 2.0]})
    target = pd.Series([3.0, 4.0])

    with caplog.at_level(logging.INFO):
        assert model.fit(features, target) is model
        predictions = model.predict(features)

    assert predictions.tolist() == [0.0, 0.0]
    assert model.rows_seen == 2
    assert model.get_name() == "stub"
    assert model.get_version() == "1.0"
    assert "Model fitting complete" in caplog.text
    assert "Prediction complete" in caplog.text


def test_parameters_are_merged_and_defensively_copied() -> None:
    """Parameter reads should not expose internal mutable state."""
    model = StubModel({"alpha": 1, "nested": {"value": 2}})

    model.set_parameters({"beta": 2})
    parameters = model.get_parameters()
    parameters["alpha"] = 99
    parameters["nested"]["value"] = 99

    assert model.get_parameters() == {
        "alpha": 1,
        "nested": {"value": 2},
        "beta": 2,
    }


@pytest.mark.parametrize(
    ("features", "target", "message"),
    [
        (np.array([]), None, "two-dimensional"),
        (np.empty((0, 2)), None, "non-empty"),
        (np.ones((2, 2)), np.ones((2, 1)), "one-dimensional"),
        (np.ones((2, 2)), np.ones(3), "same rows"),
    ],
)
def test_validate_input_rejects_invalid_shapes(
    features: FeatureMatrix,
    target: TargetVector | None,
    message: str,
) -> None:
    """Invalid matrix and target shapes should have actionable errors."""
    with pytest.raises(ModelInputError, match=message):
        StubModel().validate_input(features, target)


def test_model_save_and_load_round_trip(tmp_path: Path) -> None:
    """Persistence should retain model type and parameters."""
    model = StubModel({"alpha": 1})

    path = model.save(tmp_path / "nested" / "model.joblib")
    restored = StubModel.load(path)

    assert path.is_file()
    assert restored.get_parameters() == {"alpha": 1}


def test_model_load_rejects_missing_and_wrong_type(tmp_path: Path) -> None:
    """Persistence errors should distinguish missing and invalid model files."""
    with pytest.raises(ModelPersistenceError, match="does not exist"):
        StubModel.load(tmp_path / "missing.joblib")

    path = tmp_path / "wrong.joblib"
    joblib.dump({"not": "a model"}, path)
    with pytest.raises(ModelPersistenceError, match="not an instance"):
        StubModel.load(path)


def test_model_load_wraps_corrupt_serialization(tmp_path: Path) -> None:
    """Unreadable persistence data should produce a stable framework error."""
    path = tmp_path / "corrupt.joblib"
    path.write_text("not a joblib payload", encoding="utf-8")

    with pytest.raises(ModelPersistenceError, match="Unable to load"):
        StubModel.load(path)


def test_model_metadata_and_parameters_are_validated() -> None:
    """Invalid framework metadata and parameter names should be rejected."""

    class MissingMetadataModel(StubModel):
        MODEL_NAME = ""

    with pytest.raises(ModelConfigurationError, match="MODEL_NAME"):
        MissingMetadataModel().get_name()

    class MissingVersionModel(StubModel):
        MODEL_VERSION = ""

    with pytest.raises(ModelConfigurationError, match="MODEL_VERSION"):
        MissingVersionModel().get_version()
    with pytest.raises(ModelConfigurationError, match="parameter names"):
        StubModel().set_parameters({"": 1})


def test_parameter_mapping_type_is_checked() -> None:
    """Runtime callers should receive a stable error for non-mapping parameters."""
    with pytest.raises(ModelConfigurationError, match="must be a mapping"):
        StubModel().set_parameters([("alpha", 1)])  # type: ignore[arg-type]


def test_fit_wraps_implementation_failure() -> None:
    """Unexpected implementation errors should become ModelTrainingError."""

    class BrokenFitModel(StubModel):
        def _fit(self, features: FeatureMatrix, target: TargetVector) -> None:
            raise ArithmeticError("implementation failed")

    with pytest.raises(ModelTrainingError, match="Failed to fit") as error:
        BrokenFitModel().fit(np.ones((2, 1)), np.ones(2))

    assert isinstance(error.value.__cause__, ArithmeticError)


def test_predict_wraps_failure_and_rejects_invalid_output() -> None:
    """Prediction failures and malformed output should use framework errors."""

    class BrokenPredictModel(StubModel):
        def _predict(self, features: FeatureMatrix) -> PredictionVector:
            raise ArithmeticError("implementation failed")

    class InvalidOutputModel(StubModel):
        def _predict(self, features: FeatureMatrix) -> PredictionVector:
            return np.zeros((len(features), 1))

    features = np.ones((2, 1))
    with pytest.raises(ModelPredictionError, match="Failed to predict") as error:
        BrokenPredictModel().predict(features)
    assert isinstance(error.value.__cause__, ArithmeticError)

    with pytest.raises(ModelPredictionError, match="one-dimensional"):
        InvalidOutputModel().predict(features)


def test_set_parameters_wraps_subclass_validation_failure() -> None:
    """Unexpected parameter-validator failures should use a stable error."""

    class BrokenParametersModel(StubModel):
        def _validate_parameters(self, parameters: dict[str, object]) -> None:
            raise ArithmeticError("implementation failed")

    with pytest.raises(ModelConfigurationError, match="Unable to update") as error:
        BrokenParametersModel().set_parameters({"alpha": 1})

    assert isinstance(error.value.__cause__, ArithmeticError)
