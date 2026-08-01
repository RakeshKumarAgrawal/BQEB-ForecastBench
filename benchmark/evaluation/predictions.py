"""Structured prediction records for benchmark dataset partitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    """Capture one model prediction and its source observation metadata."""

    sample_id: str
    timestamp: str
    model_name: str
    actual: float
    predicted: float

    def __post_init__(self) -> None:
        """Validate prediction identity and finite numeric values."""
        if not self.sample_id.strip():
            raise ValueError("sample_id must be a non-empty string")
        if not self.timestamp.strip():
            raise ValueError("timestamp must be a non-empty string")
        if not self.model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if not np.isfinite(self.actual) or not np.isfinite(self.predicted):
            raise ValueError("Prediction values must be finite")

    @property
    def absolute_error(self) -> float:
        """Return the non-negative absolute prediction error."""
        return abs(self.actual - self.predicted)

    @classmethod
    def from_values(
        cls,
        *,
        partition: str,
        model_name: str,
        actual: np.ndarray,
        predicted: np.ndarray,
        timestamps: tuple[str, ...],
        sample_ids: tuple[str, ...] | None = None,
    ) -> tuple[Self, ...]:
        """Build aligned records for one model and dataset partition."""
        if actual.ndim != 1 or predicted.ndim != 1:
            raise ValueError("Prediction arrays must be one-dimensional")
        count = len(actual)
        if len(predicted) != count or len(timestamps) != count:
            raise ValueError("Prediction values and metadata must have equal lengths")
        identifiers = sample_ids or tuple(
            f"{partition}-{index:06d}" for index in range(count)
        )
        if len(identifiers) != count:
            raise ValueError("Sample identifiers must match prediction count")
        return tuple(
            cls(
                sample_id=identifiers[index],
                timestamp=timestamps[index],
                model_name=model_name,
                actual=float(actual[index]),
                predicted=float(predicted[index]),
            )
            for index in range(count)
        )
