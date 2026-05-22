"""Structured logging configuration for Baize using structlog.

Call :func:`setup_logging` once at application startup to configure JSON
structured logging output to stdout.
"""

import logging
import sys

import structlog


def setup_logging(log_level: str = "INFO") -> None:
    """Configure structlog for JSON structured output to stdout.

    Emits one JSON object per log record with at minimum these fields:
    - ``timestamp``: ISO 8601 UTC string (e.g. ``2024-01-01T10:00:00Z``)
    - ``level``: lowercase level name (``info``, ``debug``, ``error``, …)
    - ``event``: the log message

    This function is idempotent — calling it multiple times is safe.

    Args:
        log_level: The minimum log level to emit (e.g. "DEBUG", "INFO").
                   Defaults to "INFO".
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure stdlib logging — routes all loggers through structlog's output.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
        force=True,
    )

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
