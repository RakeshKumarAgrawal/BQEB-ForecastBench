"""Deterministic train, validation, and test dataset splitting."""

import logging
from dataclasses import dataclass, field

import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SplitConfig:
    """Ratios and ordering behavior for dataset partitions."""

    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    shuffle: bool = False
    time_series: bool = True

    def __post_init__(self) -> None:
        """Validate that ratios form three non-empty conceptual partitions."""
        ratios = (self.train_ratio, self.validation_ratio, self.test_ratio)
        if any(ratio <= 0 for ratio in ratios):
            raise ValueError("Split ratios must be greater than zero")
        if abs(sum(ratios) - 1.0) > 1e-9:
            raise ValueError("Split ratios must sum to 1.0")
        if self.time_series and self.shuffle:
            raise ValueError("Time-series splitting cannot enable shuffle")


@dataclass(frozen=True, slots=True)
class DatasetSplits:
    """Train, validation, and test DataFrames."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True, slots=True)
class DatasetSplitter:
    """Partition datasets chronologically or with seeded random shuffling."""

    config: SplitConfig = field(default_factory=SplitConfig)

    def split(
        self, dataset: pd.DataFrame, timestamp_column: str | None = None
    ) -> DatasetSplits:
        """Split ``dataset`` according to the configured ratios and ordering."""
        if len(dataset) < 3:
            raise ValueError("Dataset must contain at least three rows")
        ordered = self._order(dataset, timestamp_column)
        train_end = int(len(ordered) * self.config.train_ratio)
        validation_end = train_end + int(len(ordered) * self.config.validation_ratio)
        if (
            train_end == 0
            or validation_end == train_end
            or validation_end >= len(ordered)
        ):
            raise ValueError("Split ratios produce an empty partition")

        splits = DatasetSplits(
            train=ordered.iloc[:train_end].reset_index(drop=True),
            validation=ordered.iloc[train_end:validation_end].reset_index(drop=True),
            test=ordered.iloc[validation_end:].reset_index(drop=True),
        )
        LOGGER.info(
            "Split dataset train=%d validation=%d test=%d",
            len(splits.train),
            len(splits.validation),
            len(splits.test),
        )
        return splits

    def _order(
        self, dataset: pd.DataFrame, timestamp_column: str | None
    ) -> pd.DataFrame:
        if self.config.time_series:
            if timestamp_column is None or timestamp_column not in dataset.columns:
                raise ValueError("Time-series splitting requires a timestamp column")
            return dataset.sort_values(timestamp_column, kind="stable")
        if self.config.shuffle:
            return dataset.sample(frac=1.0, random_state=self.config.random_seed)
        return dataset.copy()
