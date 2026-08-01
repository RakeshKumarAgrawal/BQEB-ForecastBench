"""Tests for registry-driven model construction from YAML."""

from pathlib import Path
from typing import Any

import pytest
import yaml
from conftest import RegisteredStubModel

from benchmark.constants import CONFIG_DIR
from benchmark.models.factory import ModelFactoryError, create_model
from benchmark.models.registry import ModelRegistry


def _write_config(path: Path, values: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    return path


def test_factory_creates_default_registered_model(
    tmp_path: Path, registry: ModelRegistry
) -> None:
    """Default YAML selection should instantiate its registered class."""
    registry.register("registered_stub", RegisteredStubModel)
    path = _write_config(
        tmp_path / "models.yaml",
        {
            "default_model": "registered_stub",
            "models": {
                "registered_stub": {
                    "enabled": True,
                    "parameters": {"alpha": 1},
                }
            },
        },
    )

    model = create_model(config_path=path, registry=registry)

    assert isinstance(model, RegisteredStubModel)
    assert model.get_parameters() == {"alpha": 1}


def test_factory_supports_explicit_configuration_key(
    tmp_path: Path, registry: ModelRegistry
) -> None:
    """An explicit name should override the configured default."""
    registry.register("registered_stub", RegisteredStubModel)
    path = _write_config(
        tmp_path / "models.yaml",
        {
            "default_model": "other",
            "models": {
                "registered_stub": {"parameters": {}},
                "other": {"enabled": False},
            },
        },
    )

    model = create_model("registered_stub", config_path=path, registry=registry)

    assert isinstance(model, RegisteredStubModel)


def test_factory_rejects_disabled_model(
    tmp_path: Path, registry: ModelRegistry
) -> None:
    """Disabled configuration entries should not be constructible."""
    path = _write_config(
        tmp_path / "models.yaml",
        {
            "default_model": "disabled",
            "models": {"disabled": {"enabled": False}},
        },
    )

    with pytest.raises(ModelFactoryError, match="is disabled"):
        create_model(config_path=path, registry=registry)


def test_factory_rejects_unregistered_model(
    tmp_path: Path, registry: ModelRegistry
) -> None:
    """YAML entries should not bypass registry ownership."""
    path = _write_config(
        tmp_path / "models.yaml",
        {
            "default_model": "missing",
            "models": {"missing": {"parameters": {}}},
        },
    )

    with pytest.raises(ModelFactoryError, match="not registered"):
        create_model(config_path=path, registry=registry)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([], "root must be a mapping"),
        ({"models": {}}, "default_model"),
        ({"default_model": "stub", "models": []}, "must be a mapping"),
        (
            {
                "default_model": "stub",
                "models": {"stub": {"enabled": "yes"}},
            },
            "enabled must be boolean",
        ),
        (
            {
                "default_model": "stub",
                "models": {"stub": {"parameters": []}},
            },
            "parameters must be a string-keyed mapping",
        ),
    ],
)
def test_factory_rejects_invalid_configuration(
    tmp_path: Path,
    registry: ModelRegistry,
    values: Any,
    message: str,
) -> None:
    """Malformed YAML structures should produce actionable errors."""
    path = _write_config(tmp_path / "invalid.yaml", values)

    with pytest.raises(ModelFactoryError, match=message):
        create_model(config_path=path, registry=registry)


def test_factory_rejects_missing_configuration(registry: ModelRegistry) -> None:
    """Missing configuration paths should fail before registry lookup."""
    with pytest.raises(ModelFactoryError, match="does not exist"):
        create_model(config_path=Path("missing-models.yaml"), registry=registry)


def test_factory_wraps_malformed_yaml(tmp_path: Path, registry: ModelRegistry) -> None:
    """YAML parser failures should produce a stable factory error."""
    path = tmp_path / "malformed.yaml"
    path.write_text("models: [unterminated", encoding="utf-8")

    with pytest.raises(ModelFactoryError, match="Unable to read"):
        create_model(config_path=path, registry=registry)


def test_isolated_registry_rejects_repository_default(
    registry: ModelRegistry,
) -> None:
    """Factory injection should not silently fall back to global registrations."""
    with pytest.raises(ModelFactoryError, match="not registered"):
        create_model(registry=registry)


def test_repository_parameters_include_required_metadata() -> None:
    """Every baseline parameter should document its default and supported type."""
    configuration = yaml.safe_load(
        (CONFIG_DIR / "models.yaml").read_text(encoding="utf-8")
    )

    for model in configuration["models"].values():
        for parameter in model["parameters"].values():
            assert set(parameter) == {"default", "description", "type"}
            assert isinstance(parameter["description"], str)
            assert parameter["description"]
            assert isinstance(parameter["type"], str)
            assert parameter["type"]
