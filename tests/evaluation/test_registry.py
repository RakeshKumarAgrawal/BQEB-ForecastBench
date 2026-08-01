"""Tests for metric registration lifecycle behavior."""

import numpy as np
import pytest

from benchmark.evaluation.registry import (
    DuplicateMetricError,
    MetricNotRegisteredError,
    MetricRegistry,
    MetricRegistryError,
)
from benchmark.models.base_model import TargetVector


def sample_metric(actual: TargetVector, predicted: TargetVector) -> float:
    """Return a deterministic difference for registry tests."""
    return float(np.mean(np.asarray(actual) - np.asarray(predicted)))


def test_register_get_list_and_unregister() -> None:
    """A registered metric should support the complete lifecycle."""
    registry = MetricRegistry()
    registry.register(" Sample ", sample_metric)

    assert registry.get("sample") is sample_metric
    assert registry.list_metrics() == ("sample",)
    assert registry.unregister("SAMPLE") is sample_metric
    assert registry.list_metrics() == ()


def test_registry_rejects_duplicate_unknown_and_invalid_entries() -> None:
    """Registry failures should use stable exception categories."""
    registry = MetricRegistry()
    registry.register("sample", sample_metric)

    with pytest.raises(DuplicateMetricError, match="already registered"):
        registry.register("SAMPLE", sample_metric)
    with pytest.raises(MetricNotRegisteredError, match="not registered"):
        registry.get("missing")
    with pytest.raises(MetricNotRegisteredError, match="not registered"):
        registry.unregister("missing")
    with pytest.raises(MetricRegistryError, match="non-empty"):
        registry.register(" ", sample_metric)
    with pytest.raises(MetricRegistryError, match="callable"):
        registry.register("invalid", None)  # type: ignore[arg-type]
