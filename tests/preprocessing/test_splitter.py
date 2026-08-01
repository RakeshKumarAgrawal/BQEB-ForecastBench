"""Tests for deterministic dataset splitting."""

import pandas as pd
import pytest

from benchmark.preprocessing.splitter import DatasetSplitter, SplitConfig


def test_time_series_split_orders_chronologically() -> None:
    """Time-series partitions should preserve chronological boundaries."""
    dataset = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-04", "2026-01-01", "2026-01-03", "2026-01-02"]
            ),
            "load": [4, 1, 3, 2],
        }
    )
    splitter = DatasetSplitter(SplitConfig(0.5, 0.25, 0.25))

    splits = splitter.split(dataset, "timestamp")

    assert splits.train["load"].tolist() == [1, 2]
    assert splits.validation["load"].tolist() == [3]
    assert splits.test["load"].tolist() == [4]


def test_random_split_is_reproducible() -> None:
    """Seeded shuffling should return identical partitions across runs."""
    dataset = pd.DataFrame({"load": range(10)})
    splitter = DatasetSplitter(
        SplitConfig(0.6, 0.2, 0.2, random_seed=7, shuffle=True, time_series=False)
    )

    first = splitter.split(dataset)
    second = splitter.split(dataset)

    pd.testing.assert_frame_equal(first.train, second.train)
    assert set(first.train["load"]).isdisjoint(first.test["load"])


def test_split_config_rejects_invalid_ratios() -> None:
    """Ratios must be positive and sum exactly to one."""
    with pytest.raises(ValueError, match=r"sum to 1\.0"):
        SplitConfig(0.8, 0.15, 0.15)
    with pytest.raises(ValueError, match="greater than zero"):
        SplitConfig(1.0, 0.0, 0.0)


def test_time_series_split_requires_timestamp() -> None:
    """Chronological splitting should require an available timestamp column."""
    with pytest.raises(ValueError, match="requires a timestamp"):
        DatasetSplitter().split(pd.DataFrame({"load": [1, 2, 3]}))
