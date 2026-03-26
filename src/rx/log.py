"""Structured logging configuration for rx.

Provides a single entry point to configure structlog for the entire application.
Call configure_logging() once at startup (CLI entry points, web server init).

Usage in modules::

    import structlog
    logger = structlog.get_logger()
    logger.info("search_started", path="/var/log", patterns=3)
"""

import logging
import sys

import structlog


class _StderrLoggerFactory:
    """Logger factory that always writes to the current sys.stderr.

    Unlike PrintLoggerFactory(file=sys.stderr), this resolves sys.stderr
    at log-time, not config-time. This is important for test frameworks
    like pytest's capsys that temporarily replace sys.stderr.
    """

    def __call__(self, *args: object, **kwargs: object) -> structlog.PrintLogger:
        return structlog.PrintLogger(file=sys.stderr)


def _apply_config(level: int, json_output: bool) -> None:
    """Internal: apply structlog configuration with given settings."""
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt='iso'),
        structlog.processors.CallsiteParameterAdder(
            parameters=[
                structlog.processors.CallsiteParameter.MODULE,
                structlog.processors.CallsiteParameter.FUNC_NAME,
            ]
        ),
    ]

    if json_output:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=_StderrLoggerFactory(),
        cache_logger_on_first_use=False,
    )


# Default configuration: WARNING level to stderr.
# Prevents structlog from writing to stdout before configure_logging() is called.
# CLI/web entry points call configure_logging() to override with desired level.
_apply_config(level=logging.WARNING, json_output=False)


def configure_logging(level: str = 'INFO', json_output: bool = False) -> None:
    """Configure structlog for the application.

    Args:
        level: Log level name (DEBUG, INFO, WARNING, ERROR).
        json_output: If True, output JSON lines (for production/machine parsing).
                     If False, use colored console output (for development).
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    _apply_config(level=log_level, json_output=json_output)


def get_logger(**initial_values: object) -> structlog.stdlib.BoundLogger:
    """Get a structlog logger, optionally with bound initial values.

    Args:
        **initial_values: Key-value pairs to bind to the logger.

    Returns:
        A bound structlog logger.
    """
    return structlog.get_logger(**initial_values)
