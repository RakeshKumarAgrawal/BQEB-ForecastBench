"""Tests for central logging configuration."""

import logging
from pathlib import Path

from benchmark.config import LoggingSettings
from benchmark.utils.logging import configure_logging


def test_configure_logging_writes_utf8_file(tmp_path: Path) -> None:
    """File logging should create parent directories and persist messages."""
    log_file = tmp_path / "logs" / "forecastbench.log"
    logger = configure_logging(
        LoggingSettings(level="INFO", file=log_file, console=False)
    )

    logger.info("forecast ready")
    logging.shutdown()

    assert "forecast ready" in log_file.read_text(encoding="utf-8")


def test_configure_logging_supports_disabled_outputs() -> None:
    """Disabling console and file output should install a null handler."""
    logger = configure_logging(LoggingSettings(console=False))

    logger.info("discarded")

    assert isinstance(logging.getLogger().handlers[0], logging.NullHandler)
