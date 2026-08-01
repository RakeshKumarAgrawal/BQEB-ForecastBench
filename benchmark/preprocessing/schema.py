"""Dataset schema definitions and structural validation."""

from collections.abc import Mapping
from dataclasses import dataclass, field

import pandas as pd
from pandas.api import types as pandas_types


@dataclass(frozen=True, slots=True)
class DatasetSchema:
    """Describe required dataset columns and their expected data types."""

    target_column: str
    timestamp_column: str | None = None
    required_columns: tuple[str, ...] = ()
    optional_columns: tuple[str, ...] = ()
    data_types: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and freeze the schema definition."""
        if not self.target_column.strip():
            raise ValueError("target_column cannot be empty")
        if self.timestamp_column is not None and not self.timestamp_column.strip():
            raise ValueError("timestamp_column cannot be empty")

        required = set(self.all_required_columns())
        overlap = required.intersection(self.optional_columns)
        if overlap:
            columns = ", ".join(sorted(overlap))
            raise ValueError(f"Columns cannot be both required and optional: {columns}")
        object.__setattr__(self, "data_types", dict(self.data_types))

    def all_required_columns(self) -> tuple[str, ...]:
        """Return required columns including target and timestamp without duplicates."""
        columns = [self.target_column]
        if self.timestamp_column is not None:
            columns.append(self.timestamp_column)
        columns.extend(self.required_columns)
        return tuple(dict.fromkeys(columns))

    def missing_columns(self, dataset: pd.DataFrame) -> tuple[str, ...]:
        """Return required columns that are absent from ``dataset``."""
        return tuple(
            column
            for column in self.all_required_columns()
            if column not in dataset.columns
        )

    def invalid_data_types(self, dataset: pd.DataFrame) -> dict[str, str]:
        """Return columns whose pandas data type does not match the schema."""
        invalid: dict[str, str] = {}
        for column, expected in self.data_types.items():
            if column not in dataset.columns:
                continue
            if not _matches_data_type(dataset[column], expected):
                invalid[column] = str(dataset[column].dtype)
        return invalid


def _matches_data_type(series: pd.Series, expected: str) -> bool:
    normalized = expected.strip().lower()
    if normalized == "numeric":
        return pandas_types.is_numeric_dtype(series.dtype)
    if normalized == "datetime":
        return pandas_types.is_datetime64_any_dtype(series.dtype)
    if normalized == "string":
        return pandas_types.is_string_dtype(series.dtype)
    if normalized == "boolean":
        return pandas_types.is_bool_dtype(series.dtype)
    return str(series.dtype).lower() == normalized
