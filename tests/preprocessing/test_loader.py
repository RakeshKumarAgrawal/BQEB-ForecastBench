"""Tests for CSV dataset loading."""

from pathlib import Path

import pandas as pd
import pytest

from benchmark.preprocessing.loader import DatasetLoader, DatasetLoadError


def test_loader_supports_encoding_and_delimiter(tmp_path: Path) -> None:
    """Configured CSV text settings should be passed to pandas."""
    source = tmp_path / "dataset.csv"
    source.write_text("site;load\nMontréal;10.5\n", encoding="utf-16")

    dataset = DatasetLoader(encoding="utf-16", delimiter=";").load(source)

    pd.testing.assert_frame_equal(
        dataset,
        pd.DataFrame({"site": ["Montréal"], "load": [10.5]}),
    )


def test_loader_rejects_missing_file(tmp_path: Path) -> None:
    """Missing input files should raise the domain-specific loader error."""
    with pytest.raises(DatasetLoadError, match="does not exist"):
        DatasetLoader().load(tmp_path / "missing.csv")


def test_loader_wraps_csv_errors(tmp_path: Path) -> None:
    """Decoding and parsing failures should preserve a stable public error."""
    source = tmp_path / "invalid.csv"
    source.write_bytes(b"\xff\xfe\x00")

    with pytest.raises(DatasetLoadError, match="Unable to load dataset"):
        DatasetLoader(encoding="utf-8").load(source)
