"""Tests for training callback dispatch."""

from datetime import UTC, datetime
from pathlib import Path

from benchmark.models import LinearRegressionModel
from benchmark.training.callbacks import CallbackList, TrainingCallback, TrainingContext
from benchmark.training.history import TrainingRecord


class RecordingCallback(TrainingCallback):
    """Record lifecycle calls for callback assertions."""

    def __init__(self, label: str, events: list[str]) -> None:
        self.label = label
        self.events = events

    def on_train_begin(self, context: TrainingContext) -> None:
        self.events.append(f"{self.label}:begin:{context.model_name}")

    def on_train_end(
        self,
        context: TrainingContext,
        model: LinearRegressionModel,
        record: TrainingRecord,
    ) -> None:
        self.events.append(f"{self.label}:end:{record.status}")

    def on_checkpoint_saved(self, context: TrainingContext, path: Path) -> None:
        self.events.append(f"{self.label}:checkpoint:{path.name}")

    def on_error(self, context: TrainingContext, error: Exception) -> None:
        self.events.append(f"{self.label}:error:{type(error).__name__}")


def test_callback_list_dispatches_multiple_callbacks_in_order() -> None:
    """Every lifecycle event should preserve callback registration order."""
    events: list[str] = []
    callbacks = CallbackList(
        [RecordingCallback("first", events), RecordingCallback("second", events)]
    )
    context = TrainingContext(
        model_name="linear_regression",
        parameters={},
        configuration={},
        started_at=datetime.now(UTC),
    )
    record = TrainingRecord(
        model_name="linear_regression",
        parameters={},
        start_time="start",
        end_time="end",
        duration_seconds=1.0,
        status="completed",
        configuration_hash="hash",
        checkpoint_location="checkpoint.joblib",
    )

    callbacks.on_train_begin(context)
    callbacks.on_checkpoint_saved(context, Path("checkpoint.joblib"))
    callbacks.on_train_end(context, LinearRegressionModel(), record)
    callbacks.on_error(context, RuntimeError("failed"))

    assert events == [
        "first:begin:linear_regression",
        "second:begin:linear_regression",
        "first:checkpoint:checkpoint.joblib",
        "second:checkpoint:checkpoint.joblib",
        "first:end:completed",
        "second:end:completed",
        "first:error:RuntimeError",
        "second:error:RuntimeError",
    ]
