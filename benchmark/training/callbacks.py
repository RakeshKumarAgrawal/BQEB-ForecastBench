"""Callback interfaces for observing model training lifecycles."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmark.models.base_model import BaseForecastModel
from benchmark.training.history import TrainingRecord


@dataclass(frozen=True, slots=True)
class TrainingContext:
    """Immutable context shared with callbacks during one training run."""

    model_name: str
    parameters: Mapping[str, Any]
    configuration: Mapping[str, Any]
    started_at: datetime


class TrainingCallback:
    """Provide optional hooks for observing a training run."""

    def on_train_begin(self, context: TrainingContext) -> None:
        """Run immediately before model fitting begins."""

    def on_train_end(
        self,
        context: TrainingContext,
        model: BaseForecastModel,
        record: TrainingRecord,
    ) -> None:
        """Run after model fitting and artifact persistence complete."""

    def on_checkpoint_saved(self, context: TrainingContext, path: Path) -> None:
        """Run after a checkpoint is persisted."""

    def on_error(self, context: TrainingContext, error: Exception) -> None:
        """Run when the training lifecycle fails."""


class CallbackList:
    """Dispatch lifecycle events to callbacks in registration order."""

    def __init__(self, callbacks: Iterable[TrainingCallback] = ()) -> None:
        """Store an immutable callback sequence."""
        self._callbacks = tuple(callbacks)

    def on_train_begin(self, context: TrainingContext) -> None:
        """Dispatch the train-begin event."""
        for callback in self._callbacks:
            callback.on_train_begin(context)

    def on_train_end(
        self,
        context: TrainingContext,
        model: BaseForecastModel,
        record: TrainingRecord,
    ) -> None:
        """Dispatch the train-end event."""
        for callback in self._callbacks:
            callback.on_train_end(context, model, record)

    def on_checkpoint_saved(self, context: TrainingContext, path: Path) -> None:
        """Dispatch the checkpoint-saved event."""
        for callback in self._callbacks:
            callback.on_checkpoint_saved(context, path)

    def on_error(self, context: TrainingContext, error: Exception) -> None:
        """Dispatch the error event to every callback."""
        for callback in self._callbacks:
            callback.on_error(context, error)
