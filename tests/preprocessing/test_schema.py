"""Tests for dataset schema validation."""

import pandas as pd
import pytest

from benchmark.preprocessing.schema import DatasetSchema


def test_schema_reports_missing_required_columns() -> None:
    """Target, timestamp, and explicit required columns should be enforced."""
    schema = DatasetSchema(
        target_column="load",
        timestamp_column="timestamp",
        required_columns=("temperature",),
    )

    assert schema.missing_columns(pd.DataFrame({"load": [1.0]})) == (
        "timestamp",
        "temperature",
    )


def test_schema_reports_invalid_data_types() -> None:
    """Semantic and concrete pandas data types should be supported."""
    schema = DatasetSchema(
        target_column="load",
        data_types={"load": "numeric", "site": "string", "count": "int64"},
    )
    dataset = pd.DataFrame({"load": ["high"], "site": ["north"], "count": [1.5]})

    assert schema.invalid_data_types(dataset) == {
        "load": "object",
        "count": "float64",
    }


def test_schema_rejects_conflicting_columns() -> None:
    """A column cannot be both required and optional."""
    with pytest.raises(ValueError, match="both required and optional"):
        DatasetSchema(
            target_column="load",
            required_columns=("site",),
            optional_columns=("site",),
        )
