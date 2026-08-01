# Use the training API

This reference describes trainer orchestration, callbacks, checkpoints, versioned model input/output (I/O), and history exports from `benchmark.training`.

## Train a configured model

Create `ModelTrainer` with the repository configuration or a custom YAML path:

```python
import numpy as np

from benchmark.training import ModelTrainer

features = np.array([[0.0], [1.0], [2.0]])
target = np.array([1.0, 3.0, 5.0])
trainer = ModelTrainer()
model = trainer.train("linear_regression", features, target)
```

Pass `None` as the model name to use the configured default. Inject `callbacks` or a custom `ModelRegistry` through the constructor.

## Configure training

`load_training_settings()` returns immutable settings for these YAML values:

- `default_model`
- `checkpoint_interval`
- `random_seed`
- `artifact_paths.models`
- `artifact_paths.checkpoints`
- `artifact_paths.training`
- `logging.level` and `logging.console`
- `serialization.compression` and `serialization.protocol`

Relative artifact paths resolve from the repository root.

## Register callbacks

Subclass `TrainingCallback` to observe lifecycle events:

- `on_train_begin(context)`
- `on_train_end(context, model, record)`
- `on_checkpoint_saved(context, path)`
- `on_error(context, error)`

`TrainingContext` contains model name, parameters, configuration, and start time. Multiple callbacks execute in registration order.

## Save and restore checkpoints

Use these functions for recovery records:

- `save_checkpoint(model, configuration, directory) -> Path`
- `load_checkpoint(path) -> Checkpoint`
- `latest_checkpoint(directory, model_name=None) -> Path | None`

The loaded `Checkpoint` exposes the model, configuration, timestamp, model version, repository version, and serialization version.

## Save and restore model artifacts

Use `save_model()` to persist a model with configuration and metadata. `load_model()` returns the trained model. `load_model_artifact()` returns the complete `ModelArtifact` envelope.

```python
from pathlib import Path

from benchmark.training import load_model_artifact, save_model

path = save_model(
    model,
    Path("artifacts/models/example.joblib"),
    configuration={"random_seed": 42},
    metadata={"training_rows": 3},
)
artifact = load_model_artifact(path)
```

Only load joblib files from trusted sources. Deserialization can execute code.

## Export and load history

`TrainingHistory` stores `TrainingRecord` values. Each record contains model name, parameters, start and end times, duration, status, configuration hash, and checkpoint location.

`export(directory)` writes `training_history.json` and `training_summary.md`. `load(path)` restores the JSON history and validates schema version `1`.

The history summary reports training operations. It does not contain evaluation metrics or benchmark results.
