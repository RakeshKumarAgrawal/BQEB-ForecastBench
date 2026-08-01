"""Central logging configuration for ForecastBench."""

import logging
from logging.config import dictConfig

from benchmark.config.config import LoggingSettings
from benchmark.constants import DEFAULT_LOG_FORMAT, PACKAGE_NAME
from benchmark.utils.filesystem import ensure_directory


def configure_logging(settings: LoggingSettings) -> logging.Logger:
    """Configure application logging and return the package logger."""
    handlers: dict[str, dict[str, object]] = {}
    root_handlers: list[str] = []

    if settings.console:
        handlers["console"] = {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "stream": "ext://sys.stderr",
        }
        root_handlers.append("console")

    if settings.file is not None:
        ensure_directory(settings.file.parent)
        handlers["file"] = {
            "class": "logging.FileHandler",
            "encoding": "utf-8",
            "filename": str(settings.file),
            "formatter": "standard",
        }
        root_handlers.append("file")

    if not root_handlers:
        handlers["null"] = {"class": "logging.NullHandler"}
        root_handlers.append("null")

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {"standard": {"format": DEFAULT_LOG_FORMAT}},
            "handlers": handlers,
            "root": {"handlers": root_handlers, "level": settings.level},
        }
    )
    return logging.getLogger(PACKAGE_NAME)
