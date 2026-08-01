"""Tests for project constants."""

from benchmark.constants import (
    ARTIFACTS_DIR,
    DATA_DIR,
    PACKAGE_NAME,
    PROJECT_ROOT,
    VERSION,
)


def test_project_paths_are_derived_from_repository_root() -> None:
    """Project paths should resolve consistently from the package location."""
    assert (PROJECT_ROOT / "benchmark").is_dir()
    assert DATA_DIR == PROJECT_ROOT / "data"
    assert ARTIFACTS_DIR == PROJECT_ROOT / "artifacts"


def test_project_metadata_is_defined() -> None:
    """Package metadata should be available to infrastructure modules."""
    assert PACKAGE_NAME == "bqeb-forecastbench"
    assert VERSION == "0.2.0"
