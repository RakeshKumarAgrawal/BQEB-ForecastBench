"""Reproducible execution of one model across benchmark dataset splits."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from benchmark.evaluation.evaluator import BenchmarkEvaluator
from benchmark.evaluation.predictions import PredictionRecord
from benchmark.evaluation.results import EvaluationResult
from benchmark.models.base_model import BaseForecastModel

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DatasetPartition:
    """Feature, target, and observation metadata for one dataset split."""

    name: str
    features: np.ndarray
    target: np.ndarray
    timestamps: tuple[str, ...]
    sample_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BenchmarkDataset:
    """Loaded train, validation, and test partitions with a shared fingerprint."""

    train: DatasetPartition
    validation: DatasetPartition
    test: DatasetPartition
    fingerprint: str

    @property
    def partitions(self) -> tuple[DatasetPartition, ...]:
        """Return partitions in stable lifecycle order."""
        return self.train, self.validation, self.test


@dataclass(frozen=True, slots=True)
class ExperimentRun:
    """Capture one model's metrics, predictions, versions, and runtime values."""

    model_name: str
    model_version: str
    training_time: float
    prediction_time: float
    results: tuple[EvaluationResult, ...]
    predictions: tuple[PredictionRecord, ...]

    @property
    def test_result(self) -> EvaluationResult:
        """Return the test-partition evaluation result."""
        return next(result for result in self.results if result.dataset == "test")


def load_benchmark_dataset(
    split_paths: dict[str, Path],
    *,
    target_column: str,
    timestamp_column: str,
    sample_id_column: str | None = None,
) -> BenchmarkDataset:
    """Load configured split CSV files and compute their combined SHA-256 hash."""
    partitions: dict[str, DatasetPartition] = {}
    digest = hashlib.sha256()
    for name in ("train", "validation", "test"):
        try:
            path = split_paths[name]
        except KeyError as error:
            raise ValueError(f"Missing dataset split path: {name}") from error
        if not path.is_file():
            raise ValueError(f"Dataset split does not exist: {path}")
        content = path.read_bytes()
        digest.update(name.encode("utf-8"))
        digest.update(content)
        frame = pd.read_csv(path)
        required = {target_column, timestamp_column}
        if sample_id_column is not None:
            required.add(sample_id_column)
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Dataset split {name!r} lacks columns: {sorted(missing)}")
        excluded = required
        features = frame.drop(columns=list(excluded))
        if features.empty or not all(
            pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes
        ):
            raise ValueError(f"Dataset split {name!r} features must be numeric")
        identifiers = (
            tuple(frame[sample_id_column].astype(str))
            if sample_id_column is not None
            else tuple(f"{name}-{index:06d}" for index in range(len(frame)))
        )
        partitions[name] = DatasetPartition(
            name=name,
            features=features.to_numpy(dtype=float),
            target=frame[target_column].to_numpy(dtype=float),
            timestamps=tuple(frame[timestamp_column].astype(str)),
            sample_ids=identifiers,
        )
    return BenchmarkDataset(
        train=partitions["train"],
        validation=partitions["validation"],
        test=partitions["test"],
        fingerprint=digest.hexdigest(),
    )


class ExperimentExecutor:
    """Fit and evaluate one model across all configured dataset partitions."""

    def __init__(self, evaluator: BenchmarkEvaluator) -> None:
        """Initialize execution with the shared metric evaluator."""
        self._evaluator = evaluator

    def execute(
        self,
        model: BaseForecastModel,
        dataset: BenchmarkDataset,
        *,
        fit_model: bool,
    ) -> ExperimentRun:
        """Optionally fit a model, then predict and evaluate all three splits."""
        training_time = 0.0
        if fit_model:
            started = perf_counter()
            model.fit(dataset.train.features, dataset.train.target)
            training_time = perf_counter() - started
        results: list[EvaluationResult] = []
        records: list[PredictionRecord] = []
        prediction_time = 0.0
        for partition in dataset.partitions:
            started = perf_counter()
            predicted = model.predict(partition.features)
            prediction_time += perf_counter() - started
            results.append(
                self._evaluator.evaluate_predictions(
                    model.get_name(),
                    partition.target,
                    predicted,
                    dataset=partition.name,
                    persist=False,
                )
            )
            records.extend(
                PredictionRecord.from_values(
                    partition=partition.name,
                    model_name=model.get_name(),
                    actual=partition.target,
                    predicted=predicted,
                    timestamps=partition.timestamps,
                    sample_ids=partition.sample_ids,
                )
            )
        LOGGER.info("Experiment model complete name=%s", model.get_name())
        return ExperimentRun(
            model_name=model.get_name(),
            model_version=model.get_version(),
            training_time=training_time,
            prediction_time=prediction_time,
            results=tuple(results),
            predictions=tuple(records),
        )
