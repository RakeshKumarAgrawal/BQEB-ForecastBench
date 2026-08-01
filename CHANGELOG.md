# Changelog

All notable project milestones are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-08-01

### Commit 1 - Repository Bootstrap

- Established the Python package, repository layout, license, and project metadata.
- Added initial data, documentation, artifact, release, and test directories.

### Commit 2 - Development Infrastructure

- Added Python 3.12 packaging, dependency declarations, CI, pre-commit, Ruff, mypy, pytest, and coverage configuration.
- Added shared configuration, filesystem, environment, and logging infrastructure.

### Commit 3 - Dataset Management and Preprocessing

- Added schema validation, profiling, transformation, deterministic splitting, and preprocessing persistence.
- Produced dataset profiles, validation reports, and train/validation/test partitions.

### Commit 4 - Baseline Models and Training Pipeline

- Added the forecasting model contract, registry, factory, and linear regression, random forest, and gradient boosting baselines.
- Added configured training, callbacks, checkpoints, model envelopes, and training history.

### Commit 5 - Benchmark Evaluation Framework

- Added MAE, RMSE, MAPE, and R-squared metrics with registry-driven evaluation.
- Added reproducible benchmark execution, predictions, comparisons, CSV exports, and experiment manifests.
- Added publication-ready Table 5, Figures 4 and 5, captions, and benchmark reports.

### Commit 5.5 - Publication Evidence Package

- Added repository, artifact, environment, and tree inventories.
- Added a machine-readable publication manifest and evidence index.
- Verified artifact traceability, repository integrity, and publication readiness.

[1.0.0]: https://github.com/RakeshKumarAgrawal/BQEB-ForecastBench/releases/tag/v1.0.0
