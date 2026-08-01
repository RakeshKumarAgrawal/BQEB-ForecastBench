"""Public API for benchmark evaluation and model comparison."""

from benchmark.evaluation.benchmark import BenchmarkRunner, BenchmarkRunnerError
from benchmark.evaluation.comparison import ModelComparison
from benchmark.evaluation.evaluator import (
    BenchmarkDatasetSettings,
    BenchmarkEvaluator,
    BenchmarkModelSettings,
    EvaluationArtifactSettings,
    EvaluationConfigurationError,
    EvaluationSettings,
    PredictionOutputSettings,
    load_evaluation_settings,
)
from benchmark.evaluation.experiment import (
    BenchmarkDataset,
    DatasetPartition,
    ExperimentExecutor,
    ExperimentRun,
    load_benchmark_dataset,
)
from benchmark.evaluation.exporter import BenchmarkExporter, ranking_json
from benchmark.evaluation.manifest import (
    ExperimentManifest,
    create_manifest,
)
from benchmark.evaluation.metrics import (
    MetricInputError,
    coefficient_of_determination,
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)
from benchmark.evaluation.predictions import PredictionRecord
from benchmark.evaluation.registry import (
    METRIC_REGISTRY,
    DuplicateMetricError,
    MetricFunction,
    MetricNotRegisteredError,
    MetricRegistry,
    MetricRegistryError,
)
from benchmark.evaluation.results import (
    EvaluationResult,
    ResultSerializationError,
    write_evaluation_artifacts,
)

__all__ = [
    "METRIC_REGISTRY",
    "BenchmarkDataset",
    "BenchmarkDatasetSettings",
    "BenchmarkEvaluator",
    "BenchmarkExporter",
    "BenchmarkModelSettings",
    "BenchmarkRunner",
    "BenchmarkRunnerError",
    "DatasetPartition",
    "DuplicateMetricError",
    "EvaluationArtifactSettings",
    "EvaluationConfigurationError",
    "EvaluationResult",
    "EvaluationSettings",
    "ExperimentExecutor",
    "ExperimentManifest",
    "ExperimentRun",
    "MetricFunction",
    "MetricInputError",
    "MetricNotRegisteredError",
    "MetricRegistry",
    "MetricRegistryError",
    "ModelComparison",
    "PredictionOutputSettings",
    "PredictionRecord",
    "ResultSerializationError",
    "coefficient_of_determination",
    "create_manifest",
    "load_benchmark_dataset",
    "load_evaluation_settings",
    "mean_absolute_error",
    "mean_absolute_percentage_error",
    "ranking_json",
    "root_mean_squared_error",
    "write_evaluation_artifacts",
]
