"""Filesystem helpers shared by ForecastBench infrastructure."""

from pathlib import Path


def ensure_directory(path: Path) -> Path:
    """Create a directory and its parents, then return the resolved path."""
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def resolve_path(path: str | Path, base_dir: Path) -> Path:
    """Resolve a path relative to ``base_dir`` when it is not absolute."""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate.resolve()
