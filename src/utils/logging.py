"""
Structured logging with JSON output for production observability.
"""
from __future__ import annotations

import logging
import sys
import json
from datetime import datetime, timezone
from typing import Any, Optional

from src.config.settings import get_settings


class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter for k8s log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info),
            }
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)
        return json.dumps(log_entry, ensure_ascii=False, default=str)


class ContextLogger:
    """Logger wrapper with context injection."""

    def __init__(self, logger: logging.Logger):
        self._logger = logger
        self._context: dict[str, Any] = {}

    def bind(self, **kwargs: Any) -> ContextLogger:
        new = ContextLogger(self._logger)
        new._context = {**self._context, **kwargs}
        return new

    def _log(self, level: int, msg: str, **kwargs: Any) -> None:
        extra = {"extra_fields": {**self._context, **kwargs}}
        self._logger.log(level, msg, extra=extra)

    def debug(self, msg: str, **kwargs: Any) -> None:
        self._log(logging.DEBUG, msg, **kwargs)

    def info(self, msg: str, **kwargs: Any) -> None:
        self._log(logging.INFO, msg, **kwargs)

    def warning(self, msg: str, **kwargs: Any) -> None:
        self._log(logging.WARNING, msg, **kwargs)

    def error(self, msg: str, **kwargs: Any) -> None:
        self._log(logging.ERROR, msg, **kwargs)

    def critical(self, msg: str, **kwargs: Any) -> None:
        self._log(logging.CRITICAL, msg, **kwargs)

    def exception(self, msg: str, **kwargs: Any) -> None:
        self._logger.exception(msg, extra={"extra_fields": {**self._context, **kwargs}})


_loggers: dict[str, ContextLogger] = {}


def get_logger(name: str = "superai") -> ContextLogger:
    """Get or create a structured logger."""
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)

    if not logger.handlers:
        settings = get_settings()
        logger.setLevel(getattr(logging, settings.log_level.value))

        handler = logging.StreamHandler(sys.stdout)
        if settings.environment.value == "production":
            handler.setFormatter(JSONFormatter())
        else:
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
        logger.addHandler(handler)
        logger.propagate = False

    ctx_logger = ContextLogger(logger)
    _loggers[name] = ctx_logger
    return ctx_logger