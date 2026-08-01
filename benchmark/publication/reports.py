"""Traceable Markdown reports rendered from Batch 2 benchmark artifacts."""

from __future__ import annotations

import logging
from pathlib import Path

from benchmark.publication.inputs import PublicationData
from benchmark.publication.tables import benchmark_rows, render_benchmark_markdown
from benchmark.utils.filesystem import ensure_directory

LOGGER = logging.getLogger(__name__)

REPORT_SECTIONS = (
    "Overview",
    "Dataset summary",
    "Model summary",
    "Metric summary",
    "Experiment metadata",
    "Key observations",
    "Reproducibility information",
    "Artifact references",
)


def generate_reports(data: PublicationData, directory: Path) -> tuple[Path, ...]:
    """Generate evaluation, experiment, and benchmark Markdown reports."""
    destination = ensure_directory(directory)
    specifications = (
        (
            "evaluation_summary.md",
            "Evaluation Summary",
            "This report summarizes partition-level evaluation metrics and "
            "prediction coverage.",
        ),
        (
            "experiment_report.md",
            "Experiment Report",
            "This report documents the execution environment and "
            "reproducibility record.",
        ),
        (
            "benchmark_report.md",
            "Benchmark Report",
            "This report summarizes the ranked baseline benchmark outcomes.",
        ),
    )
    outputs = tuple(
        _write_report(data, destination / filename, title, overview)
        for filename, title, overview in specifications
    )
    LOGGER.info("Generated publication reports count=%d", len(outputs))
    return outputs


def _write_report(
    data: PublicationData,
    path: Path,
    title: str,
    overview: str,
) -> Path:
    manifest = data.manifest
    benchmark = data.benchmark_results
    predictions = data.predictions
    models = tuple(benchmark["Model"].astype(str))
    unique_samples = predictions.drop_duplicates("SampleID")
    partitions = tuple(
        unique_samples["SampleID"].astype(str).str.split("-").str[0].drop_duplicates()
    )
    best = benchmark.sort_values("Rank", kind="stable").iloc[0]
    worst = benchmark.sort_values("Rank", kind="stable").iloc[-1]
    lines = [
        f"# {title}",
        "",
        "## Overview",
        "",
        overview,
        "",
        "## Dataset summary",
        "",
        f"- Dataset: `{manifest['configuration']['dataset']}`",
        f"- Unique samples: {len(unique_samples)}",
        f"- Partitions: {', '.join(partitions)}",
        f"- Prediction records: {len(predictions)}",
        "",
        "## Model summary",
        "",
        f"- Models: {', '.join(models)}",
        f"- Model versions: `{manifest['model_versions']}`",
        "",
        "## Metric summary",
        "",
        render_benchmark_markdown(benchmark_rows(data)).rstrip(),
        "",
        "## Experiment metadata",
        "",
        f"- Experiment ID: `{manifest['experiment_id']}`",
        f"- Execution timestamp: `{manifest['execution_timestamp']}`",
        f"- Repository version: `{manifest['repository_version']}`",
        f"- Random seed: `{manifest['random_seed']}`",
        "",
        "## Key observations",
        "",
        f"- Rank 1 model: `{best['Model']}` with RMSE `{best['RMSE']}`.",
        f"- Lowest-ranked model: `{worst['Model']}` with RMSE `{worst['RMSE']}`.",
        "- Observations describe the supplied benchmark artifacts; no metrics "
        "were recomputed.",
        "",
        "## Reproducibility information",
        "",
        "- Manifest: `artifacts/experiments/experiment_manifest.json`",
        f"- Git commit: `{manifest['git_commit']}`",
        f"- Configuration hash: `{manifest['configuration_hash']}`",
        f"- Dataset fingerprint: `{manifest['dataset_fingerprint']}`",
        f"- Python version: `{manifest['python_version']}`",
        "",
        "## Artifact references",
        "",
        "- `artifacts/evaluation/metrics.csv`",
        "- `artifacts/evaluation/benchmark_results.csv`",
        "- `artifacts/evaluation/predictions.csv`",
        "- `artifacts/evaluation/model_comparison.csv`",
        "- `artifacts/experiments/experiment_manifest.json`",
        "- `artifacts/tables/Table5_BenchmarkResults.csv`",
        "- `artifacts/tables/Table5_BenchmarkResults.md`",
        "- `artifacts/figures/Figure4_PerformanceComparison.png`",
        "- `artifacts/figures/Figure5_ActualVsPredicted.png`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
