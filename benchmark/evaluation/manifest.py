"""Reproducibility manifest generation for benchmark experiments."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import distributions
from pathlib import Path
from typing import Any

from benchmark.constants import PROJECT_ROOT, VERSION
from benchmark.utils.filesystem import ensure_directory

MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ExperimentManifest:
    """Describe the code, data, configuration, and environment for one experiment."""

    experiment_id: str
    experiment_name: str
    repository_version: str
    git_commit: str
    dataset_fingerprint: str
    configuration_hash: str
    configuration: dict[str, Any]
    model_versions: dict[str, str]
    execution_timestamp: str
    python_version: str
    package_versions: dict[str, str]
    random_seed: int
    platform: str

    def to_json(self, path: Path) -> Path:
        """Write the manifest as a versioned JSON artifact."""
        destination = ensure_directory(path.expanduser().parent) / path.name
        destination.write_text(
            json.dumps(
                {"schema_version": MANIFEST_SCHEMA_VERSION, "manifest": asdict(self)},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return destination


def create_manifest(
    *,
    experiment_name: str,
    dataset_fingerprint: str,
    configuration_hash: str,
    configuration: dict[str, Any],
    model_versions: dict[str, str],
    random_seed: int,
    execution_timestamp: datetime | None = None,
) -> ExperimentManifest:
    """Capture reproducibility metadata from the current execution environment."""
    timestamp = execution_timestamp or datetime.now(UTC)
    experiment_id = (
        f"{experiment_name}-{timestamp.strftime('%Y%m%dT%H%M%S%fZ')}-"
        f"{configuration_hash[:12]}"
    )
    return ExperimentManifest(
        experiment_id=experiment_id,
        experiment_name=experiment_name,
        repository_version=VERSION,
        git_commit=_git_commit(),
        dataset_fingerprint=dataset_fingerprint,
        configuration_hash=configuration_hash,
        configuration=configuration,
        model_versions=dict(sorted(model_versions.items())),
        execution_timestamp=timestamp.isoformat(),
        python_version=sys.version,
        package_versions=_package_versions(),
        random_seed=random_seed,
        platform=platform.platform(),
    )


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("Unable to determine Git commit") from error
    return result.stdout.strip()


def _package_versions() -> dict[str, str]:
    packages = {
        distribution.metadata["Name"]: distribution.version
        for distribution in distributions()
        if distribution.metadata["Name"]
    }
    return dict(sorted(packages.items(), key=lambda item: item[0].lower()))
