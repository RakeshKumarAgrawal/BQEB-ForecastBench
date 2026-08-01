# Understand the model architecture

This page explains how ForecastBench validates model behavior, resolves configured implementations, and avoids algorithm branching. Use it when you add or review a forecasting model.

## Follow the model lifecycle

`BaseForecastModel` owns the public lifecycle. Concrete implementations inherit these operations:

1. `fit()` validates features and targets, logs the run, and calls `_fit()`
2. `predict()` validates features, calls `_predict()`, and enforces one prediction per input row
3. `set_parameters()` validates and defensively copies configuration
4. `save()` and `load()` persist trusted model instances with joblib
5. `get_name()` and `get_version()` expose stable model metadata

Framework exceptions separate input, configuration, training, prediction, and persistence failures. Callers use public methods instead of protected estimator hooks.

## Share scikit-learn behavior

`SklearnRegressorModel` is a private adapter for scikit-learn regressors. It constructs estimators, synchronizes parameter updates, delegates fitting and prediction, and converts predictions to one-dimensional NumPy arrays.

Concrete classes declare only their identity, version, default parameters, and estimator constructor:

```text
BaseForecastModel
└── SklearnRegressorModel
    ├── LinearRegressionModel
    ├── RandomForestModel
    └── GradientBoostingModel
```

This adapter keeps estimator lifecycle logic out of each baseline implementation.

## Resolve models through the registry

`ModelRegistry` maps normalized names to instantiable `BaseForecastModel` subclasses. Registration rejects duplicate names, unrelated classes, and abstract model classes. `list_models()` returns sorted names under a lock.

Importing `benchmark.models` registers all three baseline classes in `MODEL_REGISTRY`. Tests can inject an isolated registry into `create_model()` or `ModelTrainer`.

## Construct models from configuration

`create_model()` loads `benchmark/config/models.yaml`, selects the requested name or `default_model`, checks that the entry is enabled, extracts parameter defaults, and asks the registry for the class. The factory contains no algorithm-specific conditionals.

Each parameter entry documents three fields:

- `default`: runtime value passed to the model
- `description`: purpose of the estimator option
- `type`: supported YAML value type

The factory also accepts flat parameter mappings for compatibility with injected test configurations.

## Extend the model layer

Add a baseline in this order:

1. Subclass `BaseForecastModel` or `SklearnRegressorModel`
2. Declare stable `MODEL_NAME` and `MODEL_VERSION` values
3. Define and validate supported parameters
4. Implement estimator-specific fitting and prediction hooks
5. Register the class under its `models.yaml` key
6. Export the class from `benchmark.models`
7. Test lifecycle, configuration, serialization, registry, and factory integration

Do not add model selection branches to the factory or trainer.

## Treat persistence as a trust boundary

The base model API persists raw model instances. The training package adds versioned envelopes with configuration and provenance. Only deserialize joblib files from trusted sources because loading can execute code.
