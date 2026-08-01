"""Tests for stateful dataset transformations."""

import pandas as pd
import pytest

from benchmark.preprocessing.schema import DatasetSchema
from benchmark.preprocessing.transformer import (
    DatasetTransformer,
    MissingValueStrategy,
    TransformationConfig,
)


def test_transformer_imputes_converts_selects_and_scales() -> None:
    """Configured transformations should compose without mutating input data."""
    dataset = pd.DataFrame(
        {
            "timestamp": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "load": [1.0, None, 3.0],
            "site": ["a", None, "b"],
            "unused": [1, 2, 3],
        }
    )
    transformer = DatasetTransformer(
        DatasetSchema("load", "timestamp"),
        TransformationConfig(
            missing_values=MissingValueStrategy.MEAN,
            selected_columns=("timestamp", "load", "site"),
            scale_columns=("load",),
        ),
    )

    transformed = transformer.fit_transform(dataset)

    assert list(transformed.columns) == ["timestamp", "load", "site"]
    assert str(transformed["timestamp"].dtype) == "datetime64[ns, UTC]"
    assert transformed.isna().sum().sum() == 0
    assert transformed["load"].mean() == pytest.approx(0.0)
    assert dataset["load"].isna().sum() == 1


def test_transformer_requires_fit() -> None:
    """Transform should reject use before parameters have been fitted."""
    transformer = DatasetTransformer(DatasetSchema("load"))

    with pytest.raises(RuntimeError, match="must be fitted"):
        transformer.transform(pd.DataFrame({"load": [1.0]}))


@pytest.mark.parametrize(
    "strategy",
    [
        MissingValueStrategy.DROP,
        MissingValueStrategy.FORWARD_FILL,
        MissingValueStrategy.BACKWARD_FILL,
    ],
)
def test_transformer_supports_row_based_missing_strategies(
    strategy: MissingValueStrategy,
) -> None:
    """Drop and directional fill strategies should remove missing values."""
    dataset = pd.DataFrame({"load": [1.0, None, 3.0]})
    transformer = DatasetTransformer(
        DatasetSchema("load"),
        TransformationConfig(missing_values=strategy, convert_timestamp=False),
    )

    assert not transformer.fit_transform(dataset)["load"].isna().any()


def test_transformer_rejects_missing_configured_columns() -> None:
    """Configuration mistakes should fail before transformation begins."""
    transformer = DatasetTransformer(
        DatasetSchema("load"),
        TransformationConfig(scale_columns=("temperature",)),
    )

    with pytest.raises(ValueError, match="columns are missing"):
        transformer.fit(pd.DataFrame({"load": [1.0]}))
