"""Tests for lossless publication table generation."""

import csv
from pathlib import Path

from benchmark.publication import PublicationData, generate_table5


def _rows(path: Path) -> list[list[str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.reader(stream))


def test_table5_values_exactly_match_benchmark_results(tmp_path: Path) -> None:
    """Table 5 must preserve every source value and row ordering."""
    data = PublicationData.load()

    csv_path, markdown_path = generate_table5(data, tmp_path)

    assert _rows(csv_path)[1:] == _rows(data.paths.benchmark_results)[1:]
    markdown = markdown_path.read_text(encoding="utf-8")
    for source_row in _rows(data.paths.benchmark_results)[1:]:
        assert "| " + " | ".join(source_row) + " |" in markdown
