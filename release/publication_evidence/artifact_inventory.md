# Artifact Inventory

This inventory records every generated repository artifact present at evidence-package generation time. README placeholder files are excluded. Timestamps are filesystem modification times in UTC.

| Category | Path | Size (bytes) | Generation timestamp (UTC) | Originating component |
| --- | --- | ---: | --- | --- |
| Dataset | `artifacts/profiles/dataset_statistics.csv` | 601 | 2026-08-01T04:04:50.712553+00:00 | `benchmark.preprocessing.profiling` |
| Dataset | `artifacts/profiles/dataset_statistics.md` | 1858 | 2026-08-01T04:04:50.640521+00:00 | `benchmark.preprocessing.profiling` |
| Dataset | `artifacts/splits/test.csv` | 267 | 2026-08-01T04:04:50.736270+00:00 | `benchmark.preprocessing.splitter` |
| Dataset | `artifacts/splits/train.csv` | 1037 | 2026-08-01T04:04:50.711551+00:00 | `benchmark.preprocessing.splitter` |
| Dataset | `artifacts/splits/validation.csv` | 259 | 2026-08-01T04:04:50.733943+00:00 | `benchmark.preprocessing.splitter` |
| Model | `artifacts/models/linear_regression-20260801T045212941487Z.joblib` | 833 | 2026-08-01T04:52:12.952011+00:00 | `benchmark.training.model_io` |
| Checkpoint | `artifacts/checkpoints/linear_regression-20260801T045212952850Z.checkpoint.joblib` | 817 | 2026-08-01T04:52:12.954855+00:00 | `benchmark.training.checkpoint` |
| Evaluation | `artifacts/evaluation/benchmark_results.csv` | 504 | 2026-08-01T05:33:01.592777+00:00 | `benchmark.evaluation` |
| Evaluation | `artifacts/evaluation/evaluation_log.json` | 461 | 2026-08-01T05:18:18.895001+00:00 | `benchmark.evaluation` |
| Evaluation | `artifacts/evaluation/metrics.csv` | 1346 | 2026-08-01T05:33:01.585478+00:00 | `benchmark.evaluation` |
| Evaluation | `artifacts/evaluation/metrics.json` | 426 | 2026-08-01T05:18:18.907645+00:00 | `benchmark.evaluation` |
| Evaluation | `artifacts/evaluation/model_comparison.csv` | 635 | 2026-08-01T05:33:01.634282+00:00 | `benchmark.evaluation` |
| Evaluation | `artifacts/evaluation/predictions.csv` | 6059 | 2026-08-01T05:33:01.524778+00:00 | `benchmark.evaluation` |
| Table | `artifacts/tables/Table5_BenchmarkResults.csv` | 507 | 2026-08-01T06:06:24.923351+00:00 | `benchmark.publication.tables` |
| Table | `artifacts/tables/Table5_BenchmarkResults.md` | 681 | 2026-08-01T06:06:25.078146+00:00 | `benchmark.publication.tables` |
| Figure | `artifacts/figures/Figure4_caption.md` | 218 | 2026-08-01T05:48:09.345183+00:00 | `benchmark.publication.figures` |
| Figure | `artifacts/figures/Figure4_PerformanceComparison.png` | 100320 | 2026-08-01T05:46:03.572152+00:00 | `benchmark.publication.figures` |
| Figure | `artifacts/figures/Figure4_PerformanceComparison.svg` | 34795 | 2026-08-01T05:48:10.457077+00:00 | `benchmark.publication.figures` |
| Figure | `artifacts/figures/Figure5_ActualVsPredicted.png` | 190114 | 2026-08-01T05:46:03.927183+00:00 | `benchmark.publication.figures` |
| Figure | `artifacts/figures/Figure5_ActualVsPredicted.svg` | 19853 | 2026-08-01T05:48:10.539120+00:00 | `benchmark.publication.figures` |
| Figure | `artifacts/figures/Figure5_caption.md` | 250 | 2026-08-01T05:48:09.347236+00:00 | `benchmark.publication.figures` |
| Report | `artifacts/reports/benchmark_report.md` | 2423 | 2026-08-01T05:48:09.275074+00:00 | `benchmark.publication.reports` |
| Report | `artifacts/reports/evaluation_summary.md` | 2445 | 2026-08-01T05:48:09.234322+00:00 | `benchmark.publication.reports` |
| Report | `artifacts/reports/experiment_report.md` | 2437 | 2026-08-01T05:48:09.403756+00:00 | `benchmark.publication.reports` |
| Report | `artifacts/reports/validation_report.json` | 98 | 2026-08-01T04:04:50.685478+00:00 | `benchmark.preprocessing.validator` |
| Report | `artifacts/reports/validation_report.md` | 139 | 2026-08-01T04:04:50.641538+00:00 | `benchmark.preprocessing.validator` |
| Experiment manifest | `artifacts/experiments/experiment_manifest.json` | 4118 | 2026-08-01T06:13:35.144907+00:00 | `benchmark.evaluation.manifest` |
| Training | `artifacts/training/training_history.json` | 675 | 2026-08-01T05:18:07.692621+00:00 | `benchmark.training.history` |
| Training | `artifacts/training/training_summary.md` | 322 | 2026-08-01T05:18:07.694132+00:00 | `benchmark.training.history` |
| Publication manifest | `release/publication_evidence/publication_manifest.json` | 2548 | 2026-08-01T06:19:05.037209+00:00 | `Commit 5.5 evidence packaging` |

## Counts

- Dataset: 5
- Model: 1
- Checkpoint: 1
- Evaluation: 6
- Table: 2
- Figure: 6
- Report: 5
- Experiment manifest: 1
- Training: 2
- Publication manifest: 1
- Total: 30
