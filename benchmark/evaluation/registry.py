"""Thread-safe registry for forecasting metric callables."""

import logging
from collections.abc import Callable
from threading import RLock

from benchmark.models.base_model import TargetVector

LOGGER = logging.getLogger(__name__)

type MetricFunction = Callable[[TargetVector, TargetVector], float]


class MetricRegistryError(RuntimeError):
    """Base exception for metric registry operations."""


class DuplicateMetricError(MetricRegistryError):
    """Raised when a metric name is already registered."""


class MetricNotRegisteredError(MetricRegistryError, KeyError):
    """Raised when a requested metric is unavailable."""


class MetricRegistry:
    """Map stable configuration names to forecasting metric callables."""

    def __init__(self) -> None:
        """Initialize an empty registry protected by a reentrant lock."""
        self._metrics: dict[str, MetricFunction] = {}
        self._lock = RLock()

    def register(self, name: str, metric: MetricFunction) -> None:
        """Register a callable under a unique non-empty name."""
        normalized = _normalize_name(name)
        if not callable(metric):
            raise MetricRegistryError("metric must be callable")
        with self._lock:
            if normalized in self._metrics:
                raise DuplicateMetricError(
                    f"Metric {normalized!r} is already registered"
                )
            self._metrics[normalized] = metric
        LOGGER.info("Registered metric name=%s", normalized)

    def unregister(self, name: str) -> MetricFunction:
        """Remove and return the metric registered under ``name``."""
        normalized = _normalize_name(name)
        with self._lock:
            try:
                metric = self._metrics.pop(normalized)
            except KeyError as error:
                raise MetricNotRegisteredError(
                    f"Metric {normalized!r} is not registered"
                ) from error
        LOGGER.info("Unregistered metric name=%s", normalized)
        return metric

    def get(self, name: str) -> MetricFunction:
        """Return the metric registered under ``name``."""
        normalized = _normalize_name(name)
        with self._lock:
            try:
                return self._metrics[normalized]
            except KeyError as error:
                raise MetricNotRegisteredError(
                    f"Metric {normalized!r} is not registered"
                ) from error

    def list_metrics(self) -> tuple[str, ...]:
        """Return registered metric names in deterministic order."""
        with self._lock:
            return tuple(sorted(self._metrics))


def _normalize_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise MetricRegistryError("Metric registry name must be a non-empty string")
    return name.strip().lower()


METRIC_REGISTRY = MetricRegistry()
