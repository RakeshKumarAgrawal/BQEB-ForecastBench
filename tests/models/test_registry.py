"""Tests for model registration lifecycle behavior."""

import pytest
from conftest import RegisteredStubModel

from benchmark.models.base_model import BaseForecastModel
from benchmark.models.registry import (
    DuplicateModelError,
    ModelNotRegisteredError,
    ModelRegistry,
    ModelRegistryError,
)


def test_register_get_and_list_models(registry: ModelRegistry) -> None:
    """Registered model classes should be retrievable in sorted order."""
    registry.register("Stub", RegisteredStubModel)

    assert registry.get("stub") is RegisteredStubModel
    assert registry.list_models() == ("stub",)


def test_register_prevents_duplicate_names(registry: ModelRegistry) -> None:
    """Duplicate normalized names should fail without replacing the class."""
    registry.register("stub", RegisteredStubModel)

    with pytest.raises(DuplicateModelError, match="already registered"):
        registry.register(" STUB ", RegisteredStubModel)

    assert registry.get("stub") is RegisteredStubModel


def test_unregister_returns_class_and_removes_name(registry: ModelRegistry) -> None:
    """Unregister should return the removed implementation class."""
    registry.register("stub", RegisteredStubModel)

    assert registry.unregister("stub") is RegisteredStubModel
    assert registry.list_models() == ()
    with pytest.raises(ModelNotRegisteredError, match="not registered"):
        registry.get("stub")


def test_clear_removes_all_registrations(registry: ModelRegistry) -> None:
    """Clear should reset the registry to an empty deterministic state."""
    registry.register("stub", RegisteredStubModel)

    registry.clear()

    assert registry.list_models() == ()


def test_unregister_rejects_unknown_name(registry: ModelRegistry) -> None:
    """Removing an unknown model should produce a stable registry error."""
    with pytest.raises(ModelNotRegisteredError, match="not registered"):
        registry.unregister("missing")


@pytest.mark.parametrize("name", ["", "   "])
def test_registry_rejects_empty_names(registry: ModelRegistry, name: str) -> None:
    """Registry names should be non-empty after normalization."""
    with pytest.raises(ModelRegistryError, match="non-empty"):
        registry.register(name, RegisteredStubModel)


def test_registry_rejects_unrelated_classes(registry: ModelRegistry) -> None:
    """Only BaseForecastModel subclasses should be accepted."""
    with pytest.raises(ModelRegistryError, match="must inherit"):
        registry.register("invalid", dict)  # type: ignore[arg-type]


def test_registry_rejects_abstract_model_classes(
    registry: ModelRegistry,
) -> None:
    """Factory registries should contain only instantiable model classes."""
    with pytest.raises(ModelRegistryError, match="not abstract"):
        registry.register("base", BaseForecastModel)
