"""Structured records and artifact exports for training runs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

from benchmark.utils.filesystem import ensure_directory

HISTORY_FILENAME = "training_history.json"
SUMMARY_FILENAME = "training_summary.md"


@dataclass(frozen=True, slots=True)
class TrainingRecord:
    """Capture the outcome and provenance of one model training run."""

    model_name: str
    parameters: dict[str, Any]
    start_time: str
    end_time: str
    duration_seconds: float
    status: str
    configuration_hash: str
    checkpoint_location: str | None


class TrainingHistory:
    """Collect training records and export operational history artifacts."""

    def __init__(self, records: list[TrainingRecord] | None = None) -> None:
        """Initialize history from an optional record list."""
        self._records = list(records or [])

    @property
    def records(self) -> tuple[TrainingRecord, ...]:
        """Return an immutable snapshot of recorded runs."""
        return tuple(self._records)

    def add(self, record: TrainingRecord) -> None:
        """Append one training record."""
        self._records.append(record)

    def export(self, directory: Path) -> tuple[Path, Path]:
        """Write JSON history and a concise Markdown operational summary."""
        destination = ensure_directory(directory)
        history_path = destination / HISTORY_FILENAME
        summary_path = destination / SUMMARY_FILENAME
        payload = {
            "schema_version": 1,
            "records": [asdict(record) for record in self._records],
        }
        history_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        summary_path.write_text(self._summary(), encoding="utf-8")
        return history_path, summary_path

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load training history from a previously exported JSON artifact."""
        values = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(values, dict) or values.get("schema_version") != 1:
            raise ValueError("Unsupported training history schema")
        records = values.get("records")
        if not isinstance(records, list):
            raise ValueError("Training history records must be a list")
        return cls([TrainingRecord(**record) for record in records])

    def _summary(self) -> str:
        lines = [
            "# Training Summary",
            "",
            "| Model | Status | Started | Duration (s) | Checkpoint |",
            "| --- | --- | --- | ---: | --- |",
        ]
        lines.extend(
            "| {model} | {status} | {started} | {duration:.6f} | {checkpoint} |".format(
                model=record.model_name,
                status=record.status,
                started=record.start_time,
                duration=record.duration_seconds,
                checkpoint=record.checkpoint_location or "",
            )
            for record in self._records
        )
        return "\n".join(lines) + "\n"


def configuration_hash(configuration: Mapping[str, Any]) -> str:
    """Return a deterministic SHA-256 hash for a configuration snapshot."""
    encoded = json.dumps(
        configuration, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
