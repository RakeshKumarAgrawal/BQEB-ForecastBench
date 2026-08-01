"""Thread-safe registry for forecasting model implementation classes."""

import inspect
import logging
from threading import RLock

from benchmark.models.base_model import BaseForecastModel

LOGGER = logging.getLogger(__name__)


class ModelRegistryError(RuntimeError):
    """Base exception for model registry operations."""


class DuplicateModelError(ModelRegistryError):
    """Raised when a registry name is already registered."""


class ModelNotRegisteredError(ModelRegistryError, KeyError):
    """Raised when a requested registry name is unavailable."""


class ModelRegistry:
    """Map stable configuration names to forecasting model classes."""

    def __init__(self) -> None:
        """Initialize an empty registry protected by a reentrant lock."""
        self._models: dict[str, type[BaseForecastModel]] = {}
        self._lock = RLock()

    def register(self, name: str, model_class: type[BaseForecastModel]) -> None:
        """Register ``model_class`` under a unique non-empty ``name``."""
        normalized = _normalize_name(name)
        if not isinstance(model_class, type) or not issubclass(
            model_class, BaseForecastModel
        ):
            raise ModelRegistryError("model_class must inherit from BaseForecastModel")
        if inspect.isabstract(model_class):
            raise ModelRegistryError("model_class must be instantiable, not abstract")
        with self._lock:
            if normalized in self._models:
                LOGGER.error("Duplicate model registration name=%s", normalized)
                raise DuplicateModelError(f"Model {normalized!r} is already registered")
            self._models[normalized] = model_class
        LOGGER.info(
            "Registered model name=%s class=%s", normalized, model_class.__name__
        )

    def unregister(self, name: str) -> type[BaseForecastModel]:
        """Remove and return the model class registered under ``name``."""
        normalized = _normalize_name(name)
        with self._lock:
            try:
                model_class = self._models.pop(normalized)
            except KeyError as error:
                LOGGER.error("Cannot unregister unknown model name=%s", normalized)
                raise ModelNotRegisteredError(
                    f"Model {normalized!r} is not registered"
                ) from error
        LOGGER.info("Unregistered model name=%s", normalized)
        return model_class

    def get(self, name: str) -> type[BaseForecastModel]:
        """Return the model class registered under ``name``."""
        normalized = _normalize_name(name)
        with self._lock:
            try:
                model_class = self._models[normalized]
            except KeyError as error:
                LOGGER.error("Unknown model requested name=%s", normalized)
                raise ModelNotRegisteredError(
                    f"Model {normalized!r} is not registered"
                ) from error
        LOGGER.debug("Resolved model registration name=%s", normalized)
        return model_class

    def list_models(self) -> tuple[str, ...]:
        """Return registered names in deterministic sorted order."""
        with self._lock:
            names = tuple(sorted(self._models))
        LOGGER.debug("Listed model registrations count=%d", len(names))
        return names

    def clear(self) -> None:
        """Remove every model registration."""
        with self._lock:
            count = len(self._models)
            self._models.clear()
        LOGGER.info("Cleared model registry count=%d", count)


def _normalize_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ModelRegistryError("Model registry name must be a non-empty string")
    return name.strip().lower()


MODEL_REGISTRY = ModelRegistry()
