"""Tests for runtime environment handling."""

import pytest

from benchmark.config.environment import Environment, get_environment


def test_get_environment_defaults_to_development() -> None:
    """An absent environment variable should select development mode."""
    assert get_environment({}) is Environment.DEVELOPMENT


def test_get_environment_is_case_insensitive() -> None:
    """Environment names should accept surrounding whitespace and mixed case."""
    assert (
        get_environment({"BQEB_ENVIRONMENT": " Production "}) is Environment.PRODUCTION
    )


def test_get_environment_rejects_unknown_value() -> None:
    """Unsupported environment names should produce an actionable error."""
    with pytest.raises(ValueError, match="Unsupported environment"):
        get_environment({"BQEB_ENVIRONMENT": "staging"})
