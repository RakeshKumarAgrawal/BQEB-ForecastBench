# Use the models API

This reference describes the public model contract, baseline implementations, registry, factory, and persistence behavior exported from `benchmark.models`.

## Use `BaseForecastModel`

All forecasting models implement this public lifecycle:

- `fit(features, target) -> Self`: validate inputs and fit implementation state
- `predict(features) -> numpy.ndarray`: return one prediction per input row
- `save(path) -> Path`: serialize the trusted model instance with joblib
- `load(path) -> Self`: restore and type-check a trusted model instance
- `get_parameters() -> dict`: return a defensive parameter copy
- `set_parameters(parameters) -> None`: merge and validate parameter updates
- `get_name() -> str`: return the stable registry name
- `get_version() -> str`: return the model implementation version
- `validate_input(features, target=None) -> None`: validate matrix shape and row alignment

Features accept a two-dimensional pandas DataFrame or NumPy array. Targets accept a one-dimensional pandas Series or NumPy array.

## Create baseline models

The package exports three implementations:

- `LinearRegressionModel`: wraps `sklearn.linear_model.LinearRegression`
- `RandomForestModel`: wraps `sklearn.ensemble.RandomForestRegressor`
- `GradientBoostingModel`: wraps `sklearn.ensemble.GradientBoostingRegressor`

Instantiate a class directly when code supplies parameters:

```python
from benchmark.models import RandomForestModel

model = RandomForestModel({"n_estimators": 100, "max_depth": 8, "random_state": 42})
```

Use `create_model()` when `models.yaml` owns the configuration:

```python
from benchmark.models import create_model

model = create_model("gradient_boosting")
```

Omit the name to use `default_model`.

## Work with `ModelRegistry`

`MODEL_REGISTRY` contains the three baseline classes after importing `benchmark.models`:

```python
from benchmark.models import MODEL_REGISTRY

names = MODEL_REGISTRY.list_models()
```

Use an independent `ModelRegistry` in tests or extension packages. `register()`, `get()`, `unregister()`, `list_models()`, and `clear()` are thread-safe.

## Handle model errors

The model layer exports stable exception categories:

- `ModelInputError`: invalid feature or target input
- `ModelConfigurationError`: invalid metadata or parameters
- `ModelTrainingError`: estimator fitting failure
- `ModelPredictionError`: estimator prediction failure or malformed output
- `ModelPersistenceError`: base model save or load failure
- `ModelRegistryError`: invalid registry operation
- `ModelFactoryError`: invalid YAML selection or construction failure

Only load joblib files from trusted sources.
