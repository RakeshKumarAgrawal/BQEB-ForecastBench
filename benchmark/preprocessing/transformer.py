"""Configuration-driven, stateful dataset transformations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum

import pandas as pd
from pandas.api import types as pandas_types

from benchmark.preprocessing.schema import DatasetSchema

LOGGER = logging.getLogger(__name__)


class MissingValueStrategy(StrEnum):
    """Supported missing-value handling strategies."""

    NONE = "none"
    DROP = "drop"
    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"
    MEAN = "mean"
    MEDIAN = "median"


@dataclass(frozen=True, slots=True)
class TransformationConfig:
    """Controls column selection, missing values, datetimes, and scaling."""

    missing_values: MissingValueStrategy = MissingValueStrategy.NONE
    convert_timestamp: bool = True
    selected_columns: tuple[str, ...] | None = None
    scale_columns: tuple[str, ...] = ()


@dataclass(slots=True)
class DatasetTransformer:
    """Fit and apply deterministic transformations to tabular datasets."""

    schema: DatasetSchema
    config: TransformationConfig = field(default_factory=TransformationConfig)
    _fill_values: dict[str, object] = field(default_factory=dict, init=False)
    _scale_parameters: dict[str, tuple[float, float]] = field(
        default_factory=dict, init=False
    )
    _fitted: bool = field(default=False, init=False)

    def fit(self, dataset: pd.DataFrame) -> DatasetTransformer:
        """Learn imputation and standard-scaling parameters from ``dataset``."""
        self._validate_columns(dataset)
        self._fill_values = self._fit_fill_values(dataset)
        self._scale_parameters = self._fit_scale_parameters(dataset)
        self._fitted = True
        LOGGER.info(
            "Fitted transformer fill_columns=%d scale_columns=%d",
            len(self._fill_values),
            len(self._scale_parameters),
        )
        return self

    def transform(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Apply learned transformations and return a new DataFrame."""
        if not self._fitted:
            raise RuntimeError("DatasetTransformer must be fitted before transform")
        self._validate_columns(dataset)
        transformed = dataset.copy(deep=True)

        timestamp = self.schema.timestamp_column
        if self.config.convert_timestamp and timestamp is not None:
            transformed[timestamp] = pd.to_datetime(
                transformed[timestamp], errors="raise", format="mixed", utc=True
            )

        strategy = self.config.missing_values
        if strategy is MissingValueStrategy.DROP:
            transformed = transformed.dropna()
        elif strategy is MissingValueStrategy.FORWARD_FILL:
            transformed = transformed.ffill()
        elif strategy is MissingValueStrategy.BACKWARD_FILL:
            transformed = transformed.bfill()
        elif self._fill_values:
            transformed = transformed.fillna(self._fill_values)

        for column, (mean, standard_deviation) in self._scale_parameters.items():
            transformed[column] = (transformed[column] - mean) / standard_deviation

        if self.config.selected_columns is not None:
            transformed = transformed.loc[:, list(self.config.selected_columns)]
        return transformed.reset_index(drop=True)

    def fit_transform(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Fit transformation state and transform ``dataset`` in one operation."""
        return self.fit(dataset).transform(dataset)

    def _validate_columns(self, dataset: pd.DataFrame) -> None:
        configured = set(self.config.selected_columns or ()) | set(
            self.config.scale_columns
        )
        timestamp = self.schema.timestamp_column
        if self.config.convert_timestamp and timestamp is not None:
            configured.add(timestamp)
        missing = sorted(configured.difference(dataset.columns))
        if missing:
            raise ValueError(
                f"Transformation columns are missing: {', '.join(missing)}"
            )
        selected = self.config.selected_columns
        if selected is not None and self.schema.target_column not in selected:
            raise ValueError("selected_columns must include the target column")

    def _fit_fill_values(self, dataset: pd.DataFrame) -> dict[str, object]:
        strategy = self.config.missing_values
        if strategy not in {MissingValueStrategy.MEAN, MissingValueStrategy.MEDIAN}:
            return {}
        fill_values: dict[str, object] = {}
        for column in dataset.columns[dataset.isna().any()]:
            series = dataset[column]
            if pandas_types.is_numeric_dtype(series.dtype):
                value = (
                    series.mean()
                    if strategy is MissingValueStrategy.MEAN
                    else series.median()
                )
            else:
                modes = series.mode(dropna=True)
                if modes.empty:
                    raise ValueError(f"Cannot impute entirely missing column: {column}")
                value = modes.iloc[0]
            if pd.isna(value):
                raise ValueError(f"Cannot impute entirely missing column: {column}")
            fill_values[str(column)] = value
        return fill_values

    def _fit_scale_parameters(
        self, dataset: pd.DataFrame
    ) -> dict[str, tuple[float, float]]:
        parameters: dict[str, tuple[float, float]] = {}
        prepared = dataset.fillna(self._fill_values)
        for column in self.config.scale_columns:
            series = prepared[column]
            if not pandas_types.is_numeric_dtype(series.dtype):
                raise ValueError(f"Scale column must be numeric: {column}")
            mean = float(series.mean())
            standard_deviation = float(series.std(ddof=0)) or 1.0
            parameters[column] = (mean, standard_deviation)
        return parameters
