# v1.0.0 - BQEB ForecastBench

BQEB ForecastBench v1.0.0 is the first stable release of an open, reproducible benchmark framework for smart-grid forecasting with BQEB-Data v1.

## Installation

```shell
git clone https://github.com/RakeshKumarAgrawal/BQEB-ForecastBench.git
cd BQEB-ForecastBench
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On Linux or macOS, activate the environment with `source .venv/bin/activate`.

## Quick Start

```python
from benchmark.evaluation import BenchmarkRunner

runner = BenchmarkRunner()
runs = runner.run_all_models()
```

The configured runner evaluates all registered baselines and writes machine-readable artifacts under `artifacts/evaluation/` and `artifacts/experiments/`.

## Repository Highlights

- Python 3.12 with strict typing, Ruff, pytest, coverage, and cross-platform CI.
- Deterministic preprocessing and train/validation/test partitioning.
- Registry-driven model, metric, training, and evaluation architecture.
- Versioned artifacts and reproducibility manifests.
- 132 automated tests with 91.25% coverage at release preparation.

## Scientific Contributions

- A reusable benchmark protocol for comparing smart-grid load forecasting models.
- Consistent MAE, RMSE, MAPE, and R-squared evaluation across dataset partitions.
- Traceable row-level predictions and model rankings.
- A publication pipeline that consumes frozen benchmark artifacts without recomputing metrics.
- A structured evidence package for repository and publication auditing.

## Artifacts

Machine-readable metrics, predictions, comparisons, dataset profiles, split files, model metadata, and experiment provenance are available under `artifacts/`.

### Tables

- `artifacts/tables/Table5_BenchmarkResults.csv`
- `artifacts/tables/Table5_BenchmarkResults.md`

### Figures

- `artifacts/figures/Figure4_PerformanceComparison.png`
- `artifacts/figures/Figure4_PerformanceComparison.svg`
- `artifacts/figures/Figure5_ActualVsPredicted.png`
- `artifacts/figures/Figure5_ActualVsPredicted.svg`
- `artifacts/figures/Figure4_caption.md`
- `artifacts/figures/Figure5_caption.md`

### Reports

- `artifacts/reports/evaluation_summary.md`
- `artifacts/reports/experiment_report.md`
- `artifacts/reports/benchmark_report.md`
- `release/publication_evidence/evidence_index.md`

## Citation

Citation metadata is provided in `CITATION.cff` for Rakesh Kumar Agrawal (ORCID: `0009-0009-7113-5539`).

## DOI

Official Software DOI: https://doi.org/10.5281/zenodo.21735978

The Research Square preprint remains a distinct scientific publication:
https://doi.org/10.21203/rs.3.rs-10484554/v1

## Checksums

SHA-256 checksums for frozen scientific and publication artifacts:

```text
d85a24c4a83338571ccbc935444d71e35dac7b855b704b285c0ec886227f6b9c  artifacts/evaluation/benchmark_results.csv
951ebdc4090c8b949a698bb4c5ce9cb56f3dce80ac69e798f120a3b2be0b113b  artifacts/evaluation/metrics.csv
3bbaed77717575b7c60ad2bf22aff86155c9e169eb70fefdf509adf049cc5576  artifacts/evaluation/model_comparison.csv
47570fe508bc8f4432dbe135d15c7bd07b8d01eb7faffeba3048e502c218d0c6  artifacts/evaluation/predictions.csv
c8f5807bd11bec78871156f24fefe4b56a62e1908c3252383b23f50d1cd95a6b  artifacts/experiments/experiment_manifest.json
6a7ce8d2a194b3ddb4872fb8f0d757fe627c78091f8ef6ba9a30e42851d76b30  artifacts/tables/Table5_BenchmarkResults.csv
42508f8b94ba786092216cd629d88f80e7cac5ac1c045843054e181e541edb50  artifacts/tables/Table5_BenchmarkResults.md
6a5e8156c8a074c94113dbb1a0c8b959a3912ec6d5ffe14db7e5288e3495cfa2  artifacts/figures/Figure4_PerformanceComparison.png
cc041661562cafb7478c476dadfa3de047d629f22b7e8d4a77403e1c5d5820fc  artifacts/figures/Figure4_PerformanceComparison.svg
eb1338a7a7328b3299e1979532d6e05e70158d71e4050c28bb28b7ba6abd7c50  artifacts/figures/Figure4_caption.md
cc6b7c619690c9b2887170666ced20954ce75076090e93555986f1137edebbb7  artifacts/figures/Figure5_ActualVsPredicted.png
429cf7c616842e464c3f401cd8e51207f6a8eb7a8d5cfef3d268398ad1a0d596  artifacts/figures/Figure5_ActualVsPredicted.svg
bc6f4bcef4e8823c1f2027a5209bcb27edda77d99ded0afd69733c220810e200  artifacts/figures/Figure5_caption.md
caec551b3fdb733945fea77bafc7674652609224db3389c18b79b40d716bfcac  artifacts/reports/evaluation_summary.md
2cef38d8c427fb9dea3d779ad09b2f5aee49b410b60c888a32a3eb3bfe21d019  artifacts/reports/experiment_report.md
8bf0d916b09577519eec408f1321b0fba4ab4b5631e84bfb17c646d11cf49681  artifacts/reports/benchmark_report.md
03bd26550d30e0f0d08512c4b8c9bc4f3e97b921294cf23d352fec284d9b0df4  release/publication_evidence/publication_manifest.json
```
