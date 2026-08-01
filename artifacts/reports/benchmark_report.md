# Benchmark Report

## Overview

This report summarizes the ranked baseline benchmark outcomes.

## Dataset summary

- Dataset: `bqeb_test`
- Unique samples: 20
- Partitions: train, validation, test
- Prediction records: 60

## Model summary

- Models: gradient_boosting, random_forest, linear_regression
- Model versions: `{'gradient_boosting': '1.0', 'linear_regression': '1.0', 'random_forest': '1.0'}`

## Metric summary

| Rank | Model | MAE | RMSE | MAPE | R² | Training Time | Prediction Time | Repository Version |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | gradient_boosting | 25.240395589345088 | 25.344655920480122 | 0.14154082507783158 | -84.09220273105274 | 0.03168520002509467 | 0.0007610000029671937 | 0.4.0 |
| 2 | random_forest | 26.25816666666705 | 26.371866450506563 | 0.14725083253325696 | -91.12949750883355 | 0.15125809999881312 | 0.08161469997139648 | 0.4.0 |
| 3 | linear_regression | 43.24327527788853 | 44.21482202185659 | 0.24224153388856481 | -257.97195139564326 | 0.0008636999991722405 | 0.0003572999848984182 | 0.4.0 |

## Experiment metadata

- Experiment ID: `forecastbench-baselines-20260801T053015551357Z-d6289b2c451f`
- Execution timestamp: `2026-08-01T05:30:15.551357+00:00`
- Repository version: `0.4.0`
- Random seed: `42`

## Key observations

- Rank 1 model: `gradient_boosting` with RMSE `25.34465592048012`.
- Lowest-ranked model: `linear_regression` with RMSE `44.21482202185659`.
- Observations describe the supplied benchmark artifacts; no metrics were recomputed.

## Reproducibility information

- Manifest: `artifacts/experiments/experiment_manifest.json`
- Git commit: `22f0da7efd3f5f782934e00d53b340fb3c1b416a`
- Configuration hash: `d6289b2c451f33a39392fc26b556bbfecc84ff9f52d9953bb94b9f2259902b8e`
- Dataset fingerprint: `be6b8dcca5d1bdd5472c64c5380f3ca548c91db5f3a03fdd87dbb6b50764d1ea`
- Python version: `3.12.10 (tags/v3.12.10:0cc8128, Apr  8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)]`

## Artifact references

- `artifacts/evaluation/metrics.csv`
- `artifacts/evaluation/benchmark_results.csv`
- `artifacts/evaluation/predictions.csv`
- `artifacts/evaluation/model_comparison.csv`
- `artifacts/experiments/experiment_manifest.json`
- `artifacts/tables/Table5_BenchmarkResults.csv`
- `artifacts/tables/Table5_BenchmarkResults.md`
- `artifacts/figures/Figure4_PerformanceComparison.png`
- `artifacts/figures/Figure5_ActualVsPredicted.png`
