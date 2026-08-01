"""Shared fixtures for evaluation framework tests."""

from pathlib import Path

import pytest
import yaml


@pytest.fixture
def evaluation_config(tmp_path: Path) -> Path:
    """Write a minimal evaluation configuration with persistence disabled."""
    path = tmp_path / "evaluation.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "evaluation": {
                    "dataset": "unit_test",
                    "enabled_metrics": ["mae", "rmse", "mape", "r2"],
                    "options": {
                        "primary_metric": "rmse",
                        "persist_artifacts": False,
                    },
                    "prediction_output": {
                        "enabled": False,
                        "include_in_evaluation_log": False,
                    },
                    "artifact_locations": {
                        "evaluation": str(tmp_path / "evaluation"),
                        "metrics_filename": "metrics.json",
                        "log_filename": "evaluation_log.json",
                    },
                    "random_seed": 42,
                    "logging": {"level": "INFO", "console": True},
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path
