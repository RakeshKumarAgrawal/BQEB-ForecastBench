# Understand the training pipeline

This page explains how one `ModelTrainer.train()` call creates a configured model, persists recovery artifacts, dispatches callbacks, and records operational history. The pipeline does not calculate evaluation metrics or benchmark results.

## Execute one training run

`ModelTrainer` performs these steps:

1. Load and validate `benchmark/config/training.yaml`
2. Select the requested model or `default_model`
3. Apply the configured Python and NumPy random seed
4. Create the model through `create_model()` and the injected registry
5. Dispatch `on_train_begin()`
6. Fit the model with validated training data
7. Save a versioned trained-model envelope
8. Save a checkpoint when the completed-run interval matches
9. Dispatch `on_checkpoint_saved()` and `on_train_end()`
10. Export JSON history and a Markdown operational summary

If a step fails, the trainer records a failed run, exports history, dispatches `on_error()`, and reraises the original exception.

## Configure artifact ownership

`training.yaml` separates output locations by responsibility:

- `artifact_paths.models`: trained-model envelopes
- `artifact_paths.checkpoints`: timestamped recovery records
- `artifact_paths.training`: history JSON and Markdown summary

`checkpoint_interval` counts completed calls to `train()`. The baseline scikit-learn estimators expose one fit operation rather than an epoch lifecycle.

## Persist models and checkpoints

`model_io.save_model()` writes a `ModelArtifact` envelope with these values:

- Serialization schema version
- Trained model
- Model name and version
- Repository version
- Save timestamp
- Effective configuration snapshot
- Caller metadata

`save_checkpoint()` writes a `Checkpoint` with the trained model, configuration snapshot, timestamp, model version, and repository version. `latest_checkpoint()` filters by model name when supplied.

Both loaders reject unknown envelope types and unsupported schema versions. Load only trusted joblib artifacts.

## Observe runs with callbacks

Subclass `TrainingCallback` and implement the hooks you need:

```python
from benchmark.training import TrainingCallback, TrainingContext


class AuditCallback(TrainingCallback):
    def on_train_begin(self, context: TrainingContext) -> None:
        print(f"training {context.model_name}")
```

`CallbackList` invokes callbacks in registration order. Callback exceptions fail the training lifecycle and trigger error handling.

## Read training history

`TrainingHistory` records model name, parameters, timestamps, duration, status, configuration hash, and checkpoint location. It exports `training_history.json` with schema version `1` and `training_summary.md` for operational inspection.

The Markdown file summarizes training runs only. It is not a benchmark report or publication artifact.
