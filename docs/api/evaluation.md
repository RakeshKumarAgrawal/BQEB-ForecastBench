# Use the evaluation API

This reference covers metric wrappers, registry operations, model evaluation, benchmark execution, machine-readable exports, reproducibility manifests, result serialization, and comparison APIs exported from `benchmark.evaluation`.

## Run configured benchmarks

Use `BenchmarkRunner` to execute one selected model or the complete registry-driven model set:

```python
from benchmark.evaluation import BenchmarkRunner

runner = BenchmarkRunner()
all_runs = runner.run_all_models()
```

`run(model_name)` and `run_all_models()` return `ExperimentRun` values containing partition metrics, prediction records, model version, training time, and aggregate prediction time. The configured output directories receive four CSV files and one JSON manifest.

## Evaluate a trained model

Pass a trained `BaseForecastModel`, test features, and test targets to `BenchmarkEvaluator`:

```python
import numpy as np

from benchmark.evaluation import BenchmarkEvaluator
from benchmark.models import LinearRegressionModel

features = np.array([[0.0], [1.0], [2.0]])
target = np.array([1.0, 3.0, 5.0])
model = LinearRegressionModel().fit(features, target)
result = BenchmarkEvaluator().evaluate(model, features, target)
```

The call returns `EvaluationResult` and, with repository defaults, writes `artifacts/evaluation/metrics.json` and `artifacts/evaluation/evaluation_log.json`.

## Configure evaluation

`load_evaluation_settings()` validates these YAML settings:

- `dataset`
- `experiment_name`
- `enabled_metrics`
- `options.primary_metric` and `options.persist_artifacts`
- `prediction_output.enabled`
- `prediction_output.include_in_evaluation_log`
- `artifact_locations.evaluation`
- `artifact_locations.experiments`
- JSON and CSV artifact filenames
- `dataset_splits.paths.train`, `.validation`, and `.test`
- `dataset_splits.target_column`, `.timestamp_column`, and optional `.sample_id_column`
- `models.selection`, `.source`, and `.artifact_directory`
- `random_seed`
- `logging.level` and `logging.console`

Relative paths resolve from the repository root. Artifact filenames must match their configured JSON or CSV suffix, the primary metric must be enabled, and model source must be `fit_from_training_split` or `artifacts`. Set model selection to `all_registered` or a non-empty list of registry names.

## Load benchmark datasets

`load_benchmark_dataset()` reads all three split CSV files and returns `BenchmarkDataset`. Features are the numeric columns remaining after target, timestamp, and optional sample identifier columns are removed. The returned object includes a deterministic SHA-256 fingerprint of the split bytes.

`ExperimentExecutor.execute()` optionally fits one model on the training partition, generates predictions for every partition, and delegates metric computation to `BenchmarkEvaluator.evaluate_predictions()`.

## Export benchmark outputs

`BenchmarkExporter` writes exact-schema CSV artifacts:

- `export_metrics()`
- `export_benchmark_results()`
- `export_predictions()`
- `export_model_comparison()`

`create_manifest()` captures reproducibility metadata in `ExperimentManifest`; `to_json()` writes its versioned document. CSV ranking cells use compact JSON arrays so model order remains machine-readable.

## Compute metrics directly

The package exports consistent wrappers:

- `mean_absolute_error(actual, predicted)`
- `root_mean_squared_error(actual, predicted)`
- `mean_absolute_percentage_error(actual, predicted)`
- `coefficient_of_determination(actual, predicted)`

All return `float` and raise `MetricInputError` for malformed inputs.

## Work with the metric registry

`METRIC_REGISTRY` contains `mae`, `rmse`, `mape`, and `r2`. `MetricRegistry` supports `register()`, `unregister()`, `get()`, and `list_metrics()`.

```python
from benchmark.evaluation import METRIC_REGISTRY

metric = METRIC_REGISTRY.get("rmse")
names = METRIC_REGISTRY.list_metrics()
```

Inject a separate registry into `BenchmarkEvaluator` when testing extension metrics.

## Serialize results

`EvaluationResult.to_json(path)` writes a schema-versioned document. `EvaluationResult.from_json(path)` validates and restores it. Standard fields include model name, dataset, MAE, RMSE, MAPE, R-squared, prediction count, evaluation timestamp, configuration hash, and repository version. Extension values are available through `additional_metrics`, `metrics`, and `get_metric()`.

## Compare models

Create `ModelComparison(results)` from at least one result:

- `rank(metric="rmse")` returns results from best to worst
- `best(metric="rmse")` returns the leading result
- `summary_statistics()` returns descriptive values for every available metric

Use `higher_is_better` to define ranking direction for a custom metric.
