"""Tests for split loading and model experiment execution."""

from pathlib import Path

import pandas as pd

from benchmark.evaluation import BenchmarkEvaluator
from benchmark.evaluation.experiment import ExperimentExecutor, load_benchmark_dataset
from benchmark.models import LinearRegressionModel


def _write_split(path: Path, offset: int) -> Path:
    frame = pd.DataFrame(
        {
            "timestamp": [
                f"2026-01-01T0{offset}:00:00Z",
                f"2026-01-01T0{offset + 1}:00:00Z",
            ],
            "target": [float(offset * 2 + 1), float((offset + 1) * 2 + 1)],
            "feature": [float(offset), float(offset + 1)],
        }
    )
    frame.to_csv(path, index=False)
    return path


def test_experiment_executes_all_dataset_partitions(
    tmp_path: Path, evaluation_config: Path
) -> None:
    """One fitted model should emit metrics and predictions for every split."""
    dataset = load_benchmark_dataset(
        {
            "train": _write_split(tmp_path / "train.csv", 0),
            "validation": _write_split(tmp_path / "validation.csv", 2),
            "test": _write_split(tmp_path / "test.csv", 4),
        },
        target_column="target",
        timestamp_column="timestamp",
    )

    run = ExperimentExecutor(BenchmarkEvaluator(evaluation_config)).execute(
        LinearRegressionModel(), dataset, fit_model=True
    )

    assert [result.dataset for result in run.results] == [
        "train",
        "validation",
        "test",
    ]
    assert len(run.predictions) == 6
    assert run.test_result.r2 == 1.0
    assert len(dataset.fingerprint) == 64
