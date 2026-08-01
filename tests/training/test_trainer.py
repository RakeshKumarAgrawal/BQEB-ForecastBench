"""Tests for configuration-driven model training orchestration."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from benchmark.models import LinearRegressionModel, ModelInputError
from benchmark.training.callbacks import TrainingCallback, TrainingContext
from benchmark.training.history import TrainingRecord
from benchmark.training.model_io import load_model
from benchmark.training.trainer import ModelTrainer, load_training_settings


class EventCallback(TrainingCallback):
    """Capture trainer lifecycle events."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def on_train_begin(self, context: TrainingContext) -> None:
        self.events.append("begin")

    def on_checkpoint_saved(self, context: TrainingContext, path: Path) -> None:
        self.events.append("checkpoint")

    def on_train_end(
        self,
        context: TrainingContext,
        model: LinearRegressionModel,
        record: TrainingRecord,
    ) -> None:
        self.events.append("end")

    def on_error(self, context: TrainingContext, error: Exception) -> None:
        self.events.append("error")


def _write_config(tmp_path: Path) -> Path:
    values = {
        "training": {
            "default_model": "linear_regression",
            "checkpoint_interval": 1,
            "random_seed": 42,
            "artifact_paths": {
                "models": str(tmp_path / "models"),
                "checkpoints": str(tmp_path / "checkpoints"),
                "training": str(tmp_path / "training"),
            },
            "logging": {"level": "INFO", "console": True},
            "serialization": {"compression": 3, "protocol": 5},
        }
    }
    path = tmp_path / "training.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    return path


def test_trainer_generates_complete_training_artifacts(tmp_path: Path) -> None:
    """A successful run should train, persist, checkpoint, record, and callback."""
    callback = EventCallback()
    trainer = ModelTrainer(_write_config(tmp_path), callbacks=[callback])
    features = np.array([[0.0], [1.0], [2.0], [3.0]])
    target = np.array([1.0, 3.0, 5.0, 7.0])

    model = trainer.train(None, features, target)

    model_paths = list((tmp_path / "models").glob("*.joblib"))
    checkpoint_paths = list((tmp_path / "checkpoints").glob("*.joblib"))
    history_path = tmp_path / "training" / "training_history.json"
    assert isinstance(model, LinearRegressionModel)
    assert np.allclose(model.predict(features), target)
    assert np.allclose(load_model(model_paths[0]).predict(features), target)
    assert len(checkpoint_paths) == 1
    assert history_path.is_file()
    assert (tmp_path / "training" / "training_summary.md").is_file()
    assert (
        json.loads(history_path.read_text(encoding="utf-8"))["records"][0]["status"]
        == "completed"
    )
    assert callback.events == ["begin", "checkpoint", "end"]


def test_trainer_records_failure_and_invokes_error_callback(tmp_path: Path) -> None:
    """Failed input validation should be recorded before the error is reraised."""
    callback = EventCallback()
    trainer = ModelTrainer(_write_config(tmp_path), callbacks=[callback])

    with pytest.raises(ModelInputError):
        trainer.train("linear_regression", np.array([1.0]), np.array([1.0]))

    assert trainer.history.records[-1].status == "failed"
    assert callback.events == ["begin", "error"]


def test_training_settings_load_required_options(tmp_path: Path) -> None:
    """Training settings should resolve paths and serialization options."""
    settings = load_training_settings(_write_config(tmp_path))

    assert settings.default_model == "linear_regression"
    assert settings.checkpoint_interval == 1
    assert settings.random_seed == 42
    assert settings.artifacts.models == (tmp_path / "models").resolve()
    assert settings.serialization.compression == 3
