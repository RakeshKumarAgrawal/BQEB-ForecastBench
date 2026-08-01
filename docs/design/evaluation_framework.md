# Understand the evaluation framework

This page describes how ForecastBench computes metrics and executes reproducible baseline experiments. Batch 2 adds machine-readable CSV outputs and an experiment manifest. Publication figures, tables, PDF files, summaries, and benchmark reports remain outside this layer.

## Follow the evaluation lifecycle

`BenchmarkEvaluator.evaluate()` performs one deterministic evaluation:

1. Load and validate `benchmark/config/evaluation.yaml`
2. Resolve every enabled metric through `MetricRegistry`
3. Apply the configured Python and NumPy random seed
4. Ask the trained `BaseForecastModel` for predictions
5. Compute enabled metrics against the test target
6. Build an immutable `EvaluationResult` with provenance
7. Write `metrics.json` and append `evaluation_log.json` when persistence is enabled

The evaluator contains no metric-specific calculation branches. Standard and future metrics use the same callable registry contract.

## Keep metric behavior consistent

The metric wrappers validate one-dimensional, non-empty, equally sized, finite targets and predictions before calling scikit-learn. Batch 1 registers these names:

- `mae`: mean absolute error
- `rmse`: root mean squared error
- `mape`: mean absolute percentage error
- `r2`: coefficient of determination

MAPE follows scikit-learn's relative scale, where `1.0` represents 100 percent. R-squared requires at least two predictions.

## Extend metrics through the registry

`MetricRegistry` owns registration, lookup, removal, and deterministic listing. To add a metric, implement a callable with the same target and prediction inputs, register it under a stable name, and enable that name in YAML. The evaluator stores nonstandard values in `EvaluationResult.additional_metrics` without needing new evaluator branches.

Use an isolated registry in tests and extension packages. Duplicate names, empty names, non-callable values, and unknown lookups raise explicit registry exceptions.

## Execute benchmark experiments

`BenchmarkRunner` loads the configured train, validation, and test CSV files, resolves model names through `ModelRegistry`, executes each model with `ExperimentExecutor`, and exports the aggregate experiment. `run(name)` executes one selected model. `run_all_models()` executes every configured model; the repository setting `all_registered` derives this set from the registry without an algorithm list in runner code.

The default `fit_from_training_split` source creates configured model instances through `create_model()` and fits each one on the benchmark training split. The alternative `artifacts` source loads the latest trusted joblib envelope for each selected model. Models loaded from artifacts are not refit.

Each model predicts the training, validation, and test partitions. `PredictionRecord` preserves the sample identifier, source timestamp, actual value, predicted value, model name, and absolute error. When a split has no configured identifier column, stable identifiers use the partition and zero-based row position.

## Preserve reproducibility

Each result records the model name, dataset name, enabled metric values, prediction count, UTC timestamp, configuration hash, and repository version. The hash covers the effective resolved settings, including artifact locations and output options.

`metrics.json` contains the latest versioned result. `evaluation_log.json` retains an append-only list of evaluation records. Batch 2 also creates these exact machine-readable outputs:

- `metrics.csv`: partition-level metrics for every model
- `benchmark_results.csv`: test metrics, rank, fit time, and prediction time
- `predictions.csv`: row-level predictions for all three partitions
- `model_comparison.csv`: best, worst, and full ranking for each metric
- `experiment_manifest.json`: code, data, configuration, model, and environment provenance

The manifest records an experiment identifier, repository version, Git commit, combined split fingerprint, effective configuration and its hash, model versions, execution time, Python and package versions, platform, and random seed. The dataset fingerprint hashes the exact bytes of each named split in stable train-validation-test order.

## Compare evaluated models

`ModelComparison` accepts multiple `EvaluationResult` objects. Error metrics rank from low to high, while R-squared ranks from high to low. Callers can override the direction for extension metrics. The comparison engine also returns the best result and count, mean, population standard deviation, minimum, and maximum for each available metric.
