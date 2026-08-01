"""Publication-ready tables generated without benchmark recomputation."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from benchmark.publication.inputs import BENCHMARK_COLUMNS, PublicationData
from benchmark.utils.filesystem import ensure_directory

LOGGER = logging.getLogger(__name__)

TABLE5_COLUMNS = (
    "Rank",
    "Model",
    "MAE",
    "RMSE",
    "MAPE",
    "R²",
    "Training Time",
    "Prediction Time",
    "Repository Version",
)


def generate_table5(data: PublicationData, directory: Path) -> tuple[Path, Path]:
    """Copy benchmark result values into CSV and manuscript Markdown formats."""
    destination = ensure_directory(directory)
    csv_path = destination / "Table5_BenchmarkResults.csv"
    markdown_path = destination / "Table5_BenchmarkResults.md"
    source_rows = benchmark_rows(data)
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(TABLE5_COLUMNS)
        writer.writerows(source_rows)
    markdown_path.write_text(
        "# Table 5. Benchmark Results\n\n" + render_benchmark_markdown(source_rows),
        encoding="utf-8",
    )
    LOGGER.info("Generated Table 5 rows=%d", len(source_rows))
    return csv_path, markdown_path


def render_benchmark_markdown(rows: list[list[str]]) -> str:
    """Render raw benchmark strings as a lossless Markdown table."""
    lines = [
        "| " + " | ".join(TABLE5_COLUMNS) + " |",
        "| "
        + " | ".join(
            "---:" if index != 1 else "---" for index in range(len(TABLE5_COLUMNS))
        )
        + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def benchmark_rows(data: PublicationData) -> list[list[str]]:
    """Return source benchmark rows without numeric parsing or formatting."""
    with data.paths.benchmark_results.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        if tuple(header) != BENCHMARK_COLUMNS:
            raise ValueError("Benchmark result columns changed after input validation")
        return [row for row in reader]
