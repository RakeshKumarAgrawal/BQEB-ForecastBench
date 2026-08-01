"""Tests for experiment reproducibility manifests."""

import json
from datetime import UTC, datetime
from pathlib import Path

from benchmark.evaluation.manifest import create_manifest


def test_manifest_records_reproducibility_metadata(tmp_path: Path) -> None:
    """Manifest JSON should include code, data, configuration, and environment."""
    manifest = create_manifest(
        experiment_name="unit-test",
        dataset_fingerprint="a" * 64,
        configuration_hash="b" * 64,
        configuration={"random_seed": 42},
        model_versions={"linear": "1.0"},
        random_seed=42,
        execution_timestamp=datetime(2026, 8, 1, tzinfo=UTC),
    )

    path = manifest.to_json(tmp_path / "experiment_manifest.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["manifest"]["git_commit"]
    assert payload["manifest"]["python_version"]
    assert payload["manifest"]["package_versions"]["numpy"]
    assert payload["manifest"]["random_seed"] == 42
