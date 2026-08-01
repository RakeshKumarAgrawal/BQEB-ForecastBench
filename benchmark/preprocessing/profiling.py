"""Dataset profiling and statistics exports."""

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from benchmark.utils.filesystem import ensure_directory

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DatasetProfile:
    """Dimensions, memory usage, and per-column dataset statistics."""

    row_count: int
    column_count: int
    total_memory_bytes: int
    fingerprint: str
    statistics: pd.DataFrame

    def export(self, output_dir: Path) -> tuple[Path, Path]:
        """Export statistics to the required CSV and Markdown files."""
        destination = ensure_directory(output_dir)
        csv_path = destination / "dataset_statistics.csv"
        markdown_path = destination / "dataset_statistics.md"
        self.statistics.to_csv(csv_path, index=False)
        markdown = [
            "# Dataset Statistics",
            "",
            f"- Rows: {self.row_count}",
            f"- Columns: {self.column_count}",
            f"- Memory usage (bytes): {self.total_memory_bytes}",
            f"- SHA-256 fingerprint: `{self.fingerprint}`",
            "",
            self.statistics.to_markdown(index=False),
            "",
        ]
        markdown_path.write_text("\n".join(markdown), encoding="utf-8")
        return csv_path, markdown_path


@dataclass(frozen=True, slots=True)
class DatasetProfiler:
    """Generate reproducible descriptive statistics for a DataFrame."""

    include_percentiles: bool = True

    def profile(self, dataset: pd.DataFrame) -> DatasetProfile:
        """Build a profile containing dimensions and per-column summaries."""
        LOGGER.info(
            "Profiling dataset rows=%d columns=%d",
            len(dataset),
            len(dataset.columns),
        )
        records: list[dict[str, object]] = []
        for column in dataset.columns:
            series = dataset[column]
            record: dict[str, object] = {
                "column": str(column),
                "dtype": str(series.dtype),
                "non_null_count": int(series.count()),
                "missing_count": int(series.isna().sum()),
                "missing_percent": float(series.isna().mean() * 100),
                "unique_count": int(series.nunique(dropna=True)),
                "memory_bytes": int(series.memory_usage(index=False, deep=True)),
            }
            description = series.describe(
                percentiles=[0.25, 0.5, 0.75] if self.include_percentiles else []
            )
            for statistic, value in description.items():
                key = f"stat_{statistic}"
                record[key] = value
            records.append(record)

        statistics = pd.DataFrame.from_records(records)
        profile = DatasetProfile(
            row_count=len(dataset),
            column_count=len(dataset.columns),
            total_memory_bytes=int(dataset.memory_usage(index=True, deep=True).sum()),
            fingerprint=dataset_fingerprint(dataset),
            statistics=statistics,
        )
        LOGGER.info(
            "Dataset profile complete memory_bytes=%d", profile.total_memory_bytes
        )
        return profile


def dataset_fingerprint(dataset: pd.DataFrame) -> str:
    """Return a deterministic SHA-256 fingerprint for values and table structure."""
    digest = hashlib.sha256()
    digest.update("\x1f".join(str(column) for column in dataset.columns).encode())
    digest.update("\x1f".join(str(dtype) for dtype in dataset.dtypes).encode())
    row_hashes = pd.util.hash_pandas_object(dataset, index=True)
    digest.update(row_hashes.to_numpy(dtype="uint64", copy=False).tobytes())
    return digest.hexdigest()
