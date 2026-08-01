"""Registry-driven execution and export of reproducible benchmark experiments."""

from __future__ import annotations

import logging
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from benchmark.constants import VERSION
from benchmark.evaluation.comparison import ModelComparison
from benchmark.evaluation.evaluator import BenchmarkEvaluator
from benchmark.evaluation.experiment import (
    BenchmarkDataset,
    ExperimentExecutor,
    ExperimentRun,
    load_benchmark_dataset,
)
from benchmark.evaluation.exporter import BenchmarkExporter, ranking_json
from benchmark.evaluation.manifest import ExperimentManifest, create_manifest
from benchmark.models import MODEL_REGISTRY, ModelRegistry, create_model
from benchmark.models.base_model import BaseForecastModel
from benchmark.training import load_model
from benchmark.training.history import configuration_hash

LOGGER = logging.getLogger(__name__)


class BenchmarkRunnerError(RuntimeError):
    """Raised when a configured benchmark experiment cannot execute."""


class BenchmarkRunner:
    """Load benchmark inputs, execute registered models, and export outputs."""

    def __init__(
        self,
        config_path: Path | None = None,
        *,
        registry: ModelRegistry = MODEL_REGISTRY,
    ) -> None:
        """Initialize a runner from evaluation configuration and model registry."""
        self._config_path = config_path
        self._registry = registry
        self._evaluator = BenchmarkEvaluator(config_path)
        self.settings = self._evaluator.settings
        self._dataset: BenchmarkDataset | None = None

    def run(self, model_name: str) -> ExperimentRun:
        """Execute and export a benchmark for one configured registered model."""
        self._validate_selection(model_name)
        run = self._execute(model_name)
        self._export((run,))
        return run

    def run_all_models(self) -> tuple[ExperimentRun, ...]:
        """Execute and export every model selected from the registry."""
        names = self._selected_model_names()
        if not names:
            raise BenchmarkRunnerError("No registered models selected for evaluation")
        runs = tuple(self._execute(name) for name in names)
        self._export(runs)
        return runs

    def _execute(self, model_name: str) -> ExperimentRun:
        random.seed(self.settings.random_seed)
        np.random.seed(self.settings.random_seed)
        model, fit_model = self._acquire_model(model_name)
        LOGGER.info("Benchmark model started name=%s", model_name)
        return ExperimentExecutor(self._evaluator).execute(
            model, self._load_dataset(), fit_model=fit_model
        )

    def _acquire_model(self, model_name: str) -> tuple[BaseForecastModel, bool]:
        self._registry.get(model_name)
        if self.settings.benchmark_models.source == "fit_from_training_split":
            return create_model(model_name, registry=self._registry), True
        candidates = sorted(
            self.settings.benchmark_models.artifact_directory.glob(
                f"{model_name}-*.joblib"
            )
        )
        if not candidates:
            raise BenchmarkRunnerError(
                f"No trained model artifact found for {model_name!r}"
            )
        model = load_model(candidates[-1])
        if model.get_name() != model_name:
            raise BenchmarkRunnerError(
                f"Model artifact identity does not match {model_name!r}"
            )
        return model, False

    def _load_dataset(self) -> BenchmarkDataset:
        if self._dataset is None:
            dataset = self.settings.benchmark_dataset
            self._dataset = load_benchmark_dataset(
                dataset.split_paths,
                target_column=dataset.target_column,
                timestamp_column=dataset.timestamp_column,
                sample_id_column=dataset.sample_id_column,
            )
        return self._dataset

    def _selected_model_names(self) -> tuple[str, ...]:
        configured = self.settings.benchmark_models.names
        if configured is None:
            return self._registry.list_models()
        for name in configured:
            self._registry.get(name)
        return configured

    def _validate_selection(self, model_name: str) -> None:
        if model_name not in self._selected_model_names():
            raise BenchmarkRunnerError(
                f"Model {model_name!r} is not selected by evaluation configuration"
            )

    def _export(self, runs: Sequence[ExperimentRun]) -> ExperimentManifest:
        exporter = BenchmarkExporter(self.settings.artifacts.directory)
        all_results = tuple(result for run in runs for result in run.results)
        test_results = tuple(run.test_result for run in runs)
        comparison = ModelComparison(list(test_results))
        primary_ranking = comparison.rank(self.settings.primary_metric)
        ranks = {
            result.model_name: rank
            for rank, result in enumerate(primary_ranking, start=1)
        }
        exporter.export_metrics(
            all_results, self.settings.artifacts.metrics_csv_filename
        )
        exporter.export_benchmark_results(
            (
                self._benchmark_result_row(run, ranks[run.model_name])
                for run in sorted(runs, key=lambda item: ranks[item.model_name])
            ),
            self.settings.artifacts.benchmark_results_filename,
        )
        exporter.export_predictions(
            tuple(record for run in runs for record in run.predictions),
            self.settings.artifacts.predictions_filename,
        )
        exporter.export_model_comparison(
            self._comparison_rows(comparison),
            self.settings.artifacts.model_comparison_filename,
        )
        snapshot = self.settings.snapshot()
        manifest = create_manifest(
            experiment_name=self.settings.experiment_name,
            dataset_fingerprint=self._load_dataset().fingerprint,
            configuration_hash=configuration_hash(snapshot),
            configuration=snapshot,
            model_versions={run.model_name: run.model_version for run in runs},
            random_seed=self.settings.random_seed,
        )
        manifest.to_json(
            self.settings.artifacts.experiments_directory
            / self.settings.artifacts.manifest_filename
        )
        return manifest

    @staticmethod
    def _benchmark_result_row(run: ExperimentRun, rank: int) -> Mapping[str, Any]:
        result = run.test_result
        return {
            "Rank": rank,
            "Model": run.model_name,
            "MAE": result.mae,
            "RMSE": result.rmse,
            "MAPE": result.mape,
            "R²": result.r2,
            "TrainingTime": run.training_time,
            "PredictionTime": run.prediction_time,
            "RepositoryVersion": VERSION,
        }

    def _comparison_rows(
        self, comparison: ModelComparison
    ) -> tuple[Mapping[str, Any], ...]:
        rows: list[Mapping[str, Any]] = []
        for metric in self.settings.enabled_metrics:
            ranking = comparison.rank(metric)
            rows.append(
                {
                    "Metric": metric,
                    "BestModel": ranking[0].model_name,
                    "BestValue": ranking[0].get_metric(metric),
                    "WorstModel": ranking[-1].model_name,
                    "WorstValue": ranking[-1].get_metric(metric),
                    "Ranking": ranking_json(
                        tuple(result.model_name for result in ranking)
                    ),
                }
            )
        return tuple(rows)
