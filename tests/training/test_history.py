"""Tests for training history records and exports."""

import json
from pathlib import Path

from benchmark.training.history import (
    TrainingHistory,
    TrainingRecord,
    configuration_hash,
)


def test_history_exports_and_loads_required_artifacts(tmp_path: Path) -> None:
    """History should round-trip through JSON and produce a Markdown summary."""
    record = TrainingRecord(
        model_name="linear_regression",
        parameters={"positive": False},
        start_time="2026-01-01T00:00:00+00:00",
        end_time="2026-01-01T00:00:01+00:00",
        duration_seconds=1.0,
        status="completed",
        configuration_hash=configuration_hash({"random_seed": 42}),
        checkpoint_location="checkpoint.joblib",
    )
    history = TrainingHistory()
    history.add(record)

    history_path, summary_path = history.export(tmp_path)
    restored = TrainingHistory.load(history_path)

    assert restored.records == (record,)
    assert json.loads(history_path.read_text(encoding="utf-8"))["schema_version"] == 1
    assert "| linear_regression | completed |" in summary_path.read_text(
        encoding="utf-8"
    )
    assert configuration_hash({"random_seed": 42}) == configuration_hash(
        {"random_seed": 42}
    )
