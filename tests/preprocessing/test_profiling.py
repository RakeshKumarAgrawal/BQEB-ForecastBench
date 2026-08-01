"""Tests for dataset profiling and exports."""

from pathlib import Path

import pandas as pd

from benchmark.preprocessing.profiling import DatasetProfiler, dataset_fingerprint


def test_profiler_captures_dimensions_missing_values_and_statistics() -> None:
    """Profiles should include structural and descriptive column information."""
    dataset = pd.DataFrame({"load": [1.0, None, 3.0], "site": ["a", "a", "b"]})

    profile = DatasetProfiler().profile(dataset)
    load = profile.statistics.set_index("column").loc["load"]

    assert (profile.row_count, profile.column_count) == (3, 2)
    assert profile.total_memory_bytes > 0
    assert load["missing_count"] == 1
    assert load["stat_mean"] == 2.0


def test_dataset_fingerprint_is_deterministic_and_content_sensitive() -> None:
    """Equal tables should match while changed values should alter the digest."""
    dataset = pd.DataFrame({"load": [1.0, 2.0]})
    changed = pd.DataFrame({"load": [1.0, 3.0]})

    assert dataset_fingerprint(dataset) == dataset_fingerprint(dataset.copy())
    assert dataset_fingerprint(dataset) != dataset_fingerprint(changed)
    assert len(dataset_fingerprint(dataset)) == 64


def test_profile_exports_required_files(tmp_path: Path) -> None:
    """Profile exports should use the required stable filenames."""
    profile = DatasetProfiler().profile(pd.DataFrame({"load": [1.0, 2.0]}))

    csv_path, markdown_path = profile.export(tmp_path / "profiles")

    assert csv_path.name == "dataset_statistics.csv"
    assert markdown_path.name == "dataset_statistics.md"
    assert "Dataset Statistics" in markdown_path.read_text(encoding="utf-8")
