![BQEB-ForecastBench Banner](docs/bqeb-banner.png)

# BQEB-ForecastBench

## Benchmarking AI Models for Smart Grid Forecasting Using
## BIO-Quantum Energy Brain (BQEB) Dataset
[![GitHub Stars](https://img.shields.io/github/stars/RakeshKumarAgrawal/BQEB-ForecastBench)](https://github.com/RakeshKumarAgrawal/BQEB-ForecastBench)

[![License](https://img.shields.io/github/license/RakeshKumarAgrawal/BQEB-ForecastBench)]()

[![DOI](https://zenodo.org/badge/DOI/YOUR_DOI.svg)](YOUR_DOI)

[![Research](https://img.shields.io/badge/Research-AI%20Benchmark-blue)]()

BQEB-ForecastBench is an open research benchmark framework designed to evaluate Artificial Intelligence models for smart grid forecasting using the BIO-Quantum Energy Brain (BQEB) dataset.

The project provides reproducible datasets, forecasting workflows, evaluation metrics, and benchmarking methodologies for next-generation renewable energy intelligence systems.

# Research Motivation

The increasing adoption of renewable energy sources introduces uncertainty in modern power grids.

Accurate forecasting of solar generation, wind power, energy storage behavior, and grid demand requires advanced Artificial Intelligence approaches.

BQEB-ForecastBench addresses this challenge by providing a standardized evaluation framework for AI-driven forecasting models.
# Key Contributions

BQEB-ForecastBench provides:

- A benchmark framework for smart grid forecasting
- BIO-Quantum Energy Brain (BQEB) dataset integration
- AI model evaluation pipeline
- Reproducible forecasting experiments
- Standardized performance metrics
- Research-ready documentation

## Install the project

Create an isolated environment and install the package with development tools:

```shell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pre-commit install
```

On Linux or macOS, activate the environment with `source .venv/bin/activate`.

## Understand the architecture

The repository separates data preparation, model behavior, and training orchestration:

```text
benchmark/
├── config/          YAML configuration and application settings
├── evaluation/      Metrics, experiments, comparison, and benchmark exports
├── models/          Base contract, registry, factory, and baseline models
├── preprocessing/   Validation, profiling, transformation, and splitting
├── publication/     Publication tables, figures, captions, and reports
├── training/        Trainer, callbacks, checkpoints, history, and model I/O
└── utils/           Shared filesystem and logging utilities
data/
├── raw/             Immutable source data
├── processed/       Reproducible transformed data
└── sample/          Small distributable examples
artifacts/
├── evaluation/      Metrics, predictions, rankings, and evaluation logs
├── experiments/     Versioned reproducibility manifests
├── figures/         Publication figures and captions
├── models/          Versioned trained-model envelopes
├── checkpoints/     Timestamped recovery records
├── profiles/        Dataset profile exports
├── reports/         Validation and benchmark reports
├── splits/          Train, validation, and test partitions
├── tables/          Publication-ready benchmark tables
└── training/        JSON history and Markdown operational summaries
docs/
├── api/             Public API references
└── design/          Architecture and lifecycle descriptions
tests/               Evaluation, publication, model, data, and infrastructure tests
```

The model factory resolves classes through `ModelRegistry`. `ModelTrainer` uses that factory and delegates persistence, callbacks, and history to focused modules. See [model architecture](docs/design/model_architecture.md) and [training pipeline](docs/design/training_pipeline.md).

## Configure a run

ForecastBench uses four YAML files:

- `benchmark/config/forecastbench.yaml`: application paths, dataset schema, preprocessing, and split settings
- `benchmark/config/models.yaml`: enabled baseline models and documented estimator defaults
- `benchmark/config/training.yaml`: default model, random seed, checkpoint interval, artifact paths, logging, and serialization
- `benchmark/config/evaluation.yaml`: benchmark datasets, metrics, model selection, output paths, and reproducibility settings

Application settings support `BQEB_ENVIRONMENT`, `BQEB_DATA_DIR`, `BQEB_ARTIFACTS_DIR`, `BQEB_LOG_LEVEL`, `BQEB_LOG_FILE`, and `BQEB_RANDOM_SEED` overrides. Environment variables take precedence over `forecastbench.yaml`.

## Preprocess the dataset

The preprocessing pipeline loads a comma-separated values (CSV) dataset, validates its schema, writes data profiles, transforms configured columns, and creates deterministic partitions:

```python
from benchmark.preprocessing import (
    BQEBPreprocessingPipeline,
    load_preprocessing_config,
)

config = load_preprocessing_config()
pipeline = BQEBPreprocessingPipeline(config)
splits = pipeline.fit_transform()
```

The run writes validation reports, profiles, and split CSV files under `artifacts/`. Read the [preprocessing API](docs/api/preprocessing.md) for configuration and serialization details.

## Use baseline models

Commit 4 registers three scikit-learn regressors:

- Linear regression
- Random forest regression
- Gradient boosting regression

Create a configured model without algorithm-specific branching:

```python
import numpy as np

from benchmark.models import create_model

features = np.array([[0.0], [1.0], [2.0]])
target = np.array([1.0, 3.0, 5.0])
model = create_model("linear_regression")
predictions = model.fit(features, target).predict(features)
```

Read the [models API](docs/api/models.md) for lifecycle, registry, factory, and persistence contracts.

## Train and persist a model

`ModelTrainer` loads `training.yaml`, creates a model through the factory, fits it, and writes configured artifacts:

```python
import numpy as np

from benchmark.training import ModelTrainer

features = np.array([[0.0], [1.0], [2.0]])
target = np.array([1.0, 3.0, 5.0])
trainer = ModelTrainer()
model = trainer.train("linear_regression", features, target)
```

A successful run writes a trained model, an interval-controlled checkpoint, `training_history.json`, and `training_summary.md`. The summary is an operational training record, not a benchmark or publication report. Read the [training API](docs/api/training.md) for callbacks, checkpoint recovery, and model I/O.

Only load joblib files from trusted sources. Joblib deserialization can execute code.

## Evaluate and publish benchmark results

`BenchmarkRunner` executes the configured baseline models across the training,
validation, and test partitions. It writes metric, prediction, comparison, and
manifest artifacts under `artifacts/evaluation/` and `artifacts/experiments/`.
Read the [evaluation API](docs/api/evaluation.md) and
[evaluation framework](docs/design/evaluation_framework.md) for the execution
and reproducibility contracts.

The publication package consumes those persisted benchmark artifacts without
retraining models or recomputing metrics. Generated Table 5 files, Figures 4
and 5, captions, and Markdown reports are stored under `artifacts/tables/`,
`artifacts/figures/`, and `artifacts/reports/`.

## Run development checks

Run the same gates as continuous integration (CI):

```shell
python -m ruff format --check .
python -m ruff check .
python -m mypy benchmark
python -m pytest
python -m pre_commit run --all-files
```

GitHub Actions runs formatting, linting, strict type checks, and tests on Windows and Linux for every push and pull request.

## Follow the development roadmap

- ✓ **Commit 1 Complete**: repository bootstrap
- ✓ **Commit 2 Complete**: development infrastructure
- ✓ **Commit 3 Complete**: reproducible dataset preprocessing
- ✓ **Commit 4 Complete**: model framework, baseline models, and training pipeline
- ✓ **Commit 5 Complete**: evaluation engine, benchmark outputs, and publication assets
- ✓ **Commit 5.5 Complete**: publication evidence package
- ✓ **Commit 6 Complete**: official v1.0.0 release metadata

## Check current status

Commits 1 through 6 are complete and frozen. Version v1.0.0 is published as
an annotated Git tag, a GitHub Release, and a permanent Zenodo software archive.

## Citation

If you use **BQEB-ForecastBench** in academic research, please cite both the
software release and the associated scientific publication.

### Software Citation

**BQEB-ForecastBench v1.0.0**

Zenodo Software DOI: https://doi.org/10.5281/zenodo.21735978

See [CITATION.cff](CITATION.cff) for structured citation metadata and the
[GitHub v1.0.0 release](https://github.com/RakeshKumarAgrawal/BQEB-ForecastBench/releases/tag/v1.0.0)
for the archived release record.

### Scientific Publication

**Research Square Preprint**

Preprint DOI: https://doi.org/10.21203/rs.3.rs-10484554/v1

The Research Square DOI identifies the scientific preprint. The Zenodo DOI
identifies the versioned software implementation and permanent repository
archive; the two records are distinct and complementary.

## Code and data availability

The source code is public at
https://github.com/RakeshKumarAgrawal/BQEB-ForecastBench. Version v1.0.0 is
available through the GitHub Release and archived under the Zenodo software DOI
above. The benchmark uses BQEB-Data v1. Frozen benchmark artifacts, the
[experiment manifest](artifacts/experiments/experiment_manifest.json), and the
[reproducibility evidence package](release/publication_evidence/evidence_index.md)
are included in the repository archive.

See the [changelog](CHANGELOG.md) and [release notes](RELEASE_NOTES.md) for the
v1.0.0 release record.

## License

The project uses the MIT License. See [LICENSE](LICENSE).
