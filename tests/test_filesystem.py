"""Tests for filesystem utilities."""

from pathlib import Path

from benchmark.utils.filesystem import ensure_directory, resolve_path


def test_ensure_directory_creates_nested_path(tmp_path: Path) -> None:
    """Directory creation should include missing parents and return an absolute path."""
    target = tmp_path / "nested" / "output"

    result = ensure_directory(target)

    assert result == target.resolve()
    assert result.is_dir()


def test_resolve_path_uses_base_for_relative_path(tmp_path: Path) -> None:
    """Relative paths should be anchored to the provided base directory."""
    assert resolve_path("data/raw", tmp_path) == (tmp_path / "data" / "raw").resolve()


def test_resolve_path_preserves_absolute_path(tmp_path: Path) -> None:
    """Absolute paths should not be re-anchored."""
    target = (tmp_path / "absolute").resolve()
    assert resolve_path(target, tmp_path / "other") == target
