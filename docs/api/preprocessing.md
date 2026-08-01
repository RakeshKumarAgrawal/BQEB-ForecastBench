# Use the preprocessing API

This reference describes the typed, configuration-driven data layer exported
from `benchmark.preprocessing`.

## Run the preprocessing pipeline

`BQEBPreprocessingPipeline` executes the reproducible flow:

1. Load a CSV dataset.
2. Validate schema and quality constraints.
3. Export dataset statistics.
4. Fit and apply configured transformations.
5. Create deterministic train, validation, and test partitions.

```python
from benchmark.preprocessing import (
    BQEBPreprocessingPipeline,
    load_preprocessing_config,
)

config = load_preprocessing_config()
pipeline = BQEBPreprocessingPipeline(config)
splits = pipeline.fit_transform()
pipeline.save(config.artifacts_dir / "preprocessing_pipeline.joblib")
```

Only trusted pipeline files should be passed to `BQEBPreprocessingPipeline.load()` because joblib serialization can execute code during deserialization.

## Configure preprocessing

The `preprocessing` section of `benchmark/config/forecastbench.yaml` controls CSV encoding and delimiter, schema columns and data types, validation policy, transformations, split ratios, random seed, shuffling, and chronological splitting.

## Inspect preprocessing artifacts

Successful runs write validation reports under `artifacts/reports`, dataset profiles under `artifacts/profiles`, and CSV partitions under `artifacts/splits`. Invalid datasets still produce validation and profile reports before processing stops.
