"""Dataset quality validation and report generation."""

import json
import logging
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import pandas as pd

from benchmark.preprocessing.schema import DatasetSchema
from benchmark.utils.filesystem import ensure_directory

LOGGER = logging.getLogger(__name__)


class ValidationSeverity(StrEnum):
    """Severity assigned to a dataset validation issue."""

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class ValidationOptions:
    """Enable checks and choose whether quality findings are fatal."""

    check_missing_values: bool = True
    allow_missing_values: bool = False
    check_duplicate_records: bool = True
    allow_duplicate_records: bool = False
    check_duplicate_timestamps: bool = True
    allow_duplicate_timestamps: bool = False
    check_data_types: bool = True


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One actionable dataset validation finding."""

    code: str
    message: str
    count: int
    columns: tuple[str, ...] = ()
    severity: ValidationSeverity = ValidationSeverity.ERROR


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Immutable summary of all checks performed on a dataset."""

    row_count: int
    column_count: int
    issues: tuple[ValidationIssue, ...] = ()

    @property
    def valid(self) -> bool:
        """Return whether the report contains no error-level findings."""
        return all(
            issue.severity is not ValidationSeverity.ERROR for issue in self.issues
        )

    def summary(self) -> dict[str, int | bool]:
        """Return compact counts suitable for logs and programmatic checks."""
        return {
            "valid": self.valid,
            "rows": self.row_count,
            "columns": self.column_count,
            "errors": sum(
                issue.severity is ValidationSeverity.ERROR for issue in self.issues
            ),
            "warnings": sum(
                issue.severity is ValidationSeverity.WARNING for issue in self.issues
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the report."""
        return {
            **self.summary(),
            "issues": [asdict(issue) for issue in self.issues],
        }

    def save_json(self, path: Path) -> Path:
        """Write the validation report as deterministic JSON."""
        destination = ensure_directory(path.parent) / path.name
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return destination

    def save_markdown(self, path: Path) -> Path:
        """Write a human-readable Markdown validation report."""
        destination = ensure_directory(path.parent) / path.name
        summary = self.summary()
        lines = [
            "# Dataset Validation Report",
            "",
            f"- Valid: **{summary['valid']}**",
            f"- Rows: {self.row_count}",
            f"- Columns: {self.column_count}",
            f"- Errors: {summary['errors']}",
            f"- Warnings: {summary['warnings']}",
            "",
            "## Findings",
            "",
        ]
        if not self.issues:
            lines.append("No validation issues found.")
        else:
            lines.extend(
                [
                    "| Severity | Code | Count | Columns | Message |",
                    "| --- | --- | ---: | --- | --- |",
                ]
            )
            lines.extend(_issue_markdown(issue) for issue in self.issues)
        destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return destination


@dataclass(frozen=True, slots=True)
class DatasetValidator:
    """Validate a DataFrame against a schema and quality options."""

    schema: DatasetSchema
    options: ValidationOptions = field(default_factory=ValidationOptions)

    def validate(self, dataset: pd.DataFrame) -> ValidationReport:
        """Run configured checks and return a complete validation report."""
        LOGGER.info(
            "Validating dataset rows=%d columns=%d",
            len(dataset),
            len(dataset.columns),
        )
        issues: list[ValidationIssue] = []
        issues.extend(self._validate_columns(dataset))
        issues.extend(self._validate_missing_values(dataset))
        issues.extend(self._validate_duplicate_records(dataset))
        issues.extend(self._validate_timestamps(dataset))
        issues.extend(self._validate_data_types(dataset))
        report = ValidationReport(len(dataset), len(dataset.columns), tuple(issues))
        LOGGER.info("Dataset validation complete summary=%s", report.summary())
        return report

    def _validate_columns(self, dataset: pd.DataFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        missing = self.schema.missing_columns(dataset)
        if self.schema.target_column in missing:
            issues.append(
                ValidationIssue(
                    "missing_target_column",
                    f"Target column {self.schema.target_column!r} is missing.",
                    1,
                    (self.schema.target_column,),
                )
            )
        other_missing = tuple(
            column for column in missing if column != self.schema.target_column
        )
        if other_missing:
            issues.append(
                ValidationIssue(
                    "missing_required_columns",
                    "One or more required columns are missing.",
                    len(other_missing),
                    other_missing,
                )
            )
        return issues

    def _validate_missing_values(self, dataset: pd.DataFrame) -> list[ValidationIssue]:
        if not self.options.check_missing_values:
            return []
        missing = dataset.isna().sum()
        affected = tuple(str(column) for column in missing[missing > 0].index)
        if not affected:
            return []
        return [
            ValidationIssue(
                "missing_values",
                "Dataset contains missing values.",
                int(missing.sum()),
                affected,
                _severity(self.options.allow_missing_values),
            )
        ]

    def _validate_duplicate_records(
        self, dataset: pd.DataFrame
    ) -> list[ValidationIssue]:
        if not self.options.check_duplicate_records:
            return []
        count = int(dataset.duplicated().sum())
        if count == 0:
            return []
        return [
            ValidationIssue(
                "duplicate_records",
                "Dataset contains duplicate records.",
                count,
                severity=_severity(self.options.allow_duplicate_records),
            )
        ]

    def _validate_timestamps(self, dataset: pd.DataFrame) -> list[ValidationIssue]:
        column = self.schema.timestamp_column
        if column is None or column not in dataset.columns:
            return []
        converted = pd.to_datetime(
            dataset[column], errors="coerce", format="mixed", utc=True
        )
        invalid_count = int((converted.isna() & dataset[column].notna()).sum())
        issues: list[ValidationIssue] = []
        if invalid_count:
            issues.append(
                ValidationIssue(
                    "invalid_timestamps",
                    f"Timestamp column {column!r} contains invalid values.",
                    invalid_count,
                    (column,),
                )
            )
        if self.options.check_duplicate_timestamps:
            duplicate_count = int(converted.dropna().duplicated().sum())
            if duplicate_count:
                issues.append(
                    ValidationIssue(
                        "duplicate_timestamps",
                        f"Timestamp column {column!r} contains duplicates.",
                        duplicate_count,
                        (column,),
                        _severity(self.options.allow_duplicate_timestamps),
                    )
                )
        return issues

    def _validate_data_types(self, dataset: pd.DataFrame) -> list[ValidationIssue]:
        if not self.options.check_data_types:
            return []
        invalid = self.schema.invalid_data_types(dataset)
        if not invalid:
            return []
        details = ", ".join(
            f"{column} ({actual})" for column, actual in sorted(invalid.items())
        )
        return [
            ValidationIssue(
                "invalid_data_types",
                f"Columns have invalid data types: {details}.",
                len(invalid),
                tuple(sorted(invalid)),
            )
        ]


def _severity(allowed: bool) -> ValidationSeverity:
    return ValidationSeverity.WARNING if allowed else ValidationSeverity.ERROR


def _issue_markdown(issue: ValidationIssue) -> str:
    columns = ", ".join(issue.columns) or "-"
    message = issue.message.replace("|", "\\|")
    return (
        f"| {issue.severity.value} | {issue.code} | {issue.count} | "
        f"{columns} | {message} |"
    )
