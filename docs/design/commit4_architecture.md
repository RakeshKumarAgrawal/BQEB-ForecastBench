# Review the Commit 4 architecture

## Understand the Commit 4 scope

Batch 1 defines the extension contract for forecasting models. Batch 2 adds
linear regression, random forest, and gradient boosting baselines. Batch 3 adds
configuration-driven training, callbacks, checkpoints, model artifacts, and
operational history. Evaluation, metrics, benchmark reports, and visualization
remain outside this architecture.

## Follow the class hierarchy

```text
abc.ABC
└── BaseForecastModel
    ├── MODEL_NAME
    ├── MODEL_VERSION
    ├── fit() -> _fit()
    ├── predict() -> _predict()
    ├── save() / load()
    ├── get_name() / get_version()
    ├── get_parameters() / set_parameters()
    └── validate_input()
```

`BaseForecastModel` uses the template-method pattern. Its public lifecycle
methods own logging, shared input validation, stable framework exceptions,
parameter storage, prediction shape checks, and joblib persistence. Concrete
models implement only `_fit()` and `_predict()` and may override
`_validate_parameters()` to enforce algorithm-specific settings.

Joblib files must be treated as trusted inputs because deserialization can
execute code.

`SklearnRegressorModel` is a private adapter between the framework lifecycle
and scikit-learn regressors. It centralizes estimator construction, parameter
synchronization, fitting, and prediction conversion so concrete models contain
only estimator-specific defaults and construction.

## Resolve models through the registry and factory

`ModelRegistry` maps normalized configuration keys to instantiable
`BaseForecastModel` subclasses. Registration is explicit, duplicate names are
rejected, and listing is deterministic. A registry does not import or discover
implementations automatically, which keeps ownership and startup behavior
predictable.

`create_model()` reads `benchmark/config/models.yaml`, selects either the
requested key or `default_model`, checks that the entry is enabled, and resolves
the implementation through a supplied registry. The factory contains no
algorithm-specific conditionals. Importing `benchmark.models` registers all
three baseline classes in the global registry.

## Execute the training pipeline

`ModelTrainer` is the execution coordinator. It loads `training.yaml`, creates
models exclusively through `create_model()`, fits one model per `train()` call,
and delegates artifact persistence to focused services.

- `model_io` stores a versioned joblib envelope containing the trained model,
   effective configuration, model version, repository version, and metadata.
- `checkpoint` stores timestamped recovery records and discovers the latest
   checkpoint without evaluating model quality.
- `callbacks` dispatches lifecycle events in registration order.
- `history` records run provenance and exports JSON plus a Markdown operational
   summary. It contains no evaluation metrics or benchmark results.

The configured checkpoint interval counts completed `train()` calls because
baseline estimators expose a single fit operation rather than epochs.

## Extend the model framework

1. Create a focused module under `benchmark/models`.
2. Subclass `BaseForecastModel` and declare stable `MODEL_NAME` and
   `MODEL_VERSION` values.
3. Implement `_fit(features, target)` and `_predict(features)`.
4. Override `_validate_parameters(parameters)` when configuration constraints
   are required.
5. Register the class under the matching key from `models.yaml`.
6. Add lifecycle, parameter, persistence, registry, and factory tests.
7. Export the implementation only after its behavior is complete and tested.

Implementations must return one prediction per input row. Public callers use
`fit()` and `predict()` rather than protected implementation hooks.

## Run the developer workflow

Run the framework test slice while developing:

```shell
python -m pytest tests/models -q --no-cov
python -m pytest tests/training -q --no-cov
```

Before submitting a batch, run all repository gates:

```shell
python -m ruff format --check .
python -m ruff check .
python -m mypy benchmark
python -m pytest
python -m pre_commit run --all-files
```

Framework tests use non-algorithmic stubs. Baseline and training tests exercise
model fitting, persistence, registration, factory construction, callbacks,
checkpoints, history export, and trainer failure handling.
