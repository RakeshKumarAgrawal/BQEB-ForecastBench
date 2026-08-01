"""Project-wide metadata and filesystem constants."""

from pathlib import Path
from typing import Final

PROJECT_NAME: Final = "BQEB ForecastBench"
PACKAGE_NAME: Final = "bqeb-forecastbench"
VERSION: Final = "0.2.0"

PROJECT_ROOT: Final = Path(__file__).resolve().parent.parent
DATA_DIR: Final = PROJECT_ROOT / "data"
ARTIFACTS_DIR: Final = PROJECT_ROOT / "artifacts"
CONFIG_DIR: Final = PROJECT_ROOT / "benchmark" / "config"
LOGS_DIR: Final = ARTIFACTS_DIR / "logs"

DEFAULT_CONFIG_FILENAME: Final = "forecastbench.yaml"
DEFAULT_LOG_FORMAT: Final = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
ENV_PREFIX: Final = "BQEB_"
