"""Tests for dataset validation and report exports."""

import json
from pathlib import Path

import pandas as pd

from benchmark.preprocessing.schema import DatasetSchema
from benchmark.preprocessing.validator import (
    DatasetValidator,
    ValidationOptions,
    ValidationSeverity,
)


def test_validator_detects_all_supported_quality_issues() -> None:
    """Validation should aggregate structural and record-level issues."""
    dataset = pd.DataFrame(
        {
            "timestamp": ["invalid", "2026-01-01", "2026-01-01"],
            "load": ["10", None, None],
        }
    )
    schema = DatasetSchema(
        target_column="load",
        timestamp_column="timestamp",
        required_columns=("temperature",),
        data_types={"load": "numeric"},
    )

    report = DatasetValidator(schema).validate(dataset)
    codes = {issue.code for issue in report.issues}

    assert not report.valid
    assert codes == {
        "missing_required_columns",
        "missing_values",
        "duplicate_records",
        "invalid_timestamps",
        "duplicate_timestamps",
        "invalid_data_types",
    }


def test_validator_reports_missing_target_column() -> None:
    """A missing target should have a dedicated validation code."""
    report = DatasetValidator(DatasetSchema(target_column="load")).validate(
        pd.DataFrame({"temperature": [20.0]})
    )

    assert report.issues[0].code == "missing_target_column"


def test_allowed_findings_are_warnings() -> None:
    """Allowed missing values and duplicates should not invalidate a dataset."""
    dataset = pd.DataFrame({"load": [1.0, 1.0], "value": [None, None]})
    options = ValidationOptions(
        allow_missing_values=True,
        allow_duplicate_records=True,
    )

    report = DatasetValidator(DatasetSchema("load"), options).validate(dataset)

    assert report.valid
    assert all(issue.severity is ValidationSeverity.WARNING for issue in report.issues)


def test_validation_report_exports_json_and_markdown(tmp_path: Path) -> None:
    """Both report formats should be deterministic and human-readable."""
    report = DatasetValidator(DatasetSchema("load")).validate(
        pd.DataFrame({"load": [1.0]})
    )

    json_path = report.save_json(tmp_path / "reports" / "validation.json")
    markdown_path = report.save_markdown(tmp_path / "reports" / "validation.md")

    assert json.loads(json_path.read_text(encoding="utf-8"))["valid"] is True
    assert "No validation issues found." in markdown_path.read_text(encoding="utf-8")
