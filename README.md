# BQEB ForecastBench

BQEB ForecastBench is an open benchmarking framework for evaluating artificial intelligence and machine learning models for smart grid forecasting using the BQEB-Data v1 benchmark dataset.

## Features

- Standardized benchmark protocol
- Public benchmark dataset
- Baseline machine learning models
- Reproducible evaluation
- Open research artifacts
- IEEE DataPort and Zenodo integration

## Baseline Models

- Linear Regression
- Random Forest
- Gradient Boosting

## Evaluation Metrics

- MAE
- RMSE
- MAPE
- R²

## Research Artifacts

- BQEB-Data v1
- IEEE DataPort
- Zenodo
- Benchmark documentation

## Repository Architecture

```text
benchmark/                 Python benchmark package
├── config/                Configuration loading and schemas
├── models/                Forecasting model interfaces and implementations
├── preprocessing/         Data validation and feature preparation
├── training/              Model training workflows
├── evaluation/            Metrics and benchmark evaluation
├── visualization/         Plots and result visualization
└── utils/                 Shared utilities
data/                      Dataset workspace
├── raw/                   Immutable source data
├── processed/             Reproducible transformed data
└── sample/                Small distributable examples
docs/                      Project documentation
├── design/                Design notes and benchmark specifications
├── figures/               Documentation figures
└── api/                   API reference
tests/                     Automated test suite
├── preprocessing/         Preprocessing tests
├── training/              Training tests
├── evaluation/            Evaluation tests
└── models/                Model tests
artifacts/                 Generated benchmark outputs
├── models/                Exported models
├── checkpoints/           Training checkpoints
├── evaluation/            Predictions and metric outputs
├── figures/               Generated visualizations
├── reports/               Generated reports
└── experiments/           Experiment metadata and logs
release/                   Release materials
publication_evidence/      Supporting publication evidence
publication_review/        Publication review materials
```

The repository currently contains architecture and tooling scaffolding only. Benchmark logic will be introduced separately.

## License

MIT License
