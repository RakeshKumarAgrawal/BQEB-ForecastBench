# BQEB ForecastBench v1.0.0

BQEB ForecastBench v1.0.0 is the first stable release of the reproducible smart-grid forecasting benchmark framework for BQEB-Data v1.

## Major Features

- Configuration-driven dataset validation, profiling, transformation, and deterministic partitioning.
- A common forecasting model lifecycle with linear regression, random forest, and gradient boosting baselines.
- Versioned model persistence, checkpoints, callbacks, and training history.
- Registry-driven MAE, RMSE, MAPE, and R-squared evaluation.
- Machine-readable metrics, predictions, model comparisons, benchmark rankings, and experiment provenance.
- Publication-ready tables, figures, captions, reports, and a complete publication evidence package.

## Repository Statistics

- Python version: 3.12
- Tracked files at the frozen Commit 5.5 source snapshot: 175
- Automated tests: 132
- Test coverage: 91.25%
- Typed benchmark source files checked by mypy: 48
- Baseline models: 3
- Evaluation metrics: 4
- Inventoried scientific and publication artifacts: 30
- Publication evidence files: 6

## Benchmark Summary

The frozen benchmark evaluates three baseline regressors across training, validation, and test partitions. Gradient boosting ranks first on test RMSE.

| Rank | Model | Test MAE | Test RMSE | Test MAPE | Test R-squared |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Gradient Boosting | 25.240395589345088 | 25.344655920480122 | 0.14154082507783158 | -84.09220273105274 |
| 2 | Random Forest | 26.25816666666705 | 26.371866450506563 | 0.14725083253325696 | -91.12949750883355 |
| 3 | Linear Regression | 43.24327527788853 | 44.21482202185659 | 0.24224153388856481 | -257.97195139564326 |

These values are transcribed from the frozen `artifacts/evaluation/benchmark_results.csv`; this release process does not recompute them.

## Publication Assets

- Table 5 in CSV and Markdown.
- Figures 4 and 5 in 300-DPI PNG and scalable SVG formats.
- Figure captions in Markdown.
- Evaluation summary, experiment report, and benchmark report.
- Publication evidence package with repository, artifact, environment, and provenance inventories.

## Reproducibility

The frozen experiment manifest records the configuration hash, dataset fingerprint, source commit, model versions, Python runtime, dependencies, platform, and random seed. The publication manifest links the experiment snapshot to all release evidence.

Release metadata uses version `1.0.0`. Frozen scientific artifacts and the Commit 5.5 evidence snapshot retain repository version `0.4.0`, the version under which they were generated and verified. Preserving those historical values prevents provenance rewriting.

## Known Limitations

- The bundled benchmark dataset contains 20 observations and is intended to demonstrate the framework workflow rather than establish broad external validity.
- The release includes three classical regression baselines; neural, probabilistic, and domain-adapted forecasting models are not included.
- Test-partition R-squared values are negative for all bundled baselines, indicating limited generalization on the small frozen split.
- The historical experiment configuration records absolute Windows artifact paths from the execution environment.
- A Zenodo DOI is not assigned until the verified release is published and archived.

## Future Work

- Expand evaluation to larger public smart-grid datasets and additional forecasting horizons.
- Add probabilistic forecasts, uncertainty metrics, and stronger baseline families.
- Synchronize the software release with Paper Version 2.0 and archive the verified release in Zenodo.
