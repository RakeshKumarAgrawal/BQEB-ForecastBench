"""CSV dataset loading with consistent errors and logging."""

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)


class DatasetLoadError(RuntimeError):
    """Raised when a dataset cannot be loaded from disk."""


@dataclass(frozen=True, slots=True)
class DatasetLoader:
    """Load CSV files using configurable text and delimiter settings."""

    encoding: str = "utf-8"
    delimiter: str = ","

    def load(self, path: Path) -> pd.DataFrame:
        """Load ``path`` as a pandas DataFrame or raise ``DatasetLoadError``."""
        source = path.expanduser().resolve()
        if not source.is_file():
            raise DatasetLoadError(f"Dataset file does not exist: {source}")

        LOGGER.info(
            "Loading dataset path=%s encoding=%s delimiter=%r",
            source,
            self.encoding,
            self.delimiter,
        )
        try:
            dataset = pd.read_csv(
                source,
                encoding=self.encoding,
                sep=self.delimiter,
            )
        except (OSError, UnicodeError, pd.errors.ParserError) as error:
            LOGGER.exception("Failed to load dataset path=%s", source)
            raise DatasetLoadError(f"Unable to load dataset: {source}") from error

        LOGGER.info(
            "Loaded dataset path=%s rows=%d columns=%d",
            source,
            len(dataset),
            len(dataset.columns),
        )
        return dataset
