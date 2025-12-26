"""Utility functions for RX"""

import logging
import os
from pathlib import Path


# Newline symbol configuration (module-level constants)
_newline_env = os.getenv('NEWLINE_SYMBOL', '\\n')
# Handle escape sequences
NEWLINE_SYMBOL = _newline_env.replace('\\r', '\r').replace('\\n', '\n')
NEWLINE_SYMBOL_BYTES = NEWLINE_SYMBOL.encode('utf-8')


def get_int_env(key: str) -> int:
    """Get integer value from environment variable, return 0 if not set or invalid."""
    val = os.getenv(key)
    if val is None:
        return 0
    try:
        return int(val)
    except ValueError:
        return 0


def get_str_env(key: str, default: str) -> str:
    """
    Get string from environment variable, return default if not set.

    Args:
        key: Environment variable name
        default: Default value if not set

    Returns:
        String value or default
    """
    return os.getenv(key, default)


def get_bool_env(key: str, default: bool) -> bool:
    """
    Get boolean from environment variable, return default if not set.

    Recognizes: true/false, yes/no, 1/0 (case-insensitive)

    Args:
        key: Environment variable name
        default: Default value if not set

    Returns:
        Boolean value or default
    """
    val = os.getenv(key)
    if val is None:
        return default
    val_lower = val.lower()
    if val_lower in ('true', 'yes', '1'):
        return True
    elif val_lower in ('false', 'no', '0'):
        return False
    return default


class ShutdownFilter(logging.Filter):
    """
    Logging filter to suppress shutdown-related error tracebacks.

    Filters out KeyboardInterrupt, CancelledError, and SystemExit errors
    that occur during graceful shutdown of uvicorn/asyncio servers.
    """

    def filter(self, record):
        """Filter log records to suppress shutdown errors."""
        # Suppress error tracebacks related to shutdown
        if record.levelname == 'ERROR':
            msg = str(record.getMessage())
            # Check for shutdown-related errors in message
            if any(x in msg for x in ['KeyboardInterrupt', 'CancelledError', 'Shutting down']):
                return False
            # Check if the exception info contains shutdown-related exceptions
            if record.exc_info:
                exc_type = record.exc_info[0]
                if exc_type and exc_type.__name__ in ('KeyboardInterrupt', 'CancelledError', 'SystemExit'):
                    return False
        return True


def setup_shutdown_filter():
    """
    Apply ShutdownFilter to uvicorn and asyncio loggers.

    Call this before running uvicorn to suppress shutdown tracebacks.
    """
    shutdown_filter = ShutdownFilter()
    for logger_name in ['uvicorn.error', 'uvicorn', 'asyncio']:
        logger = logging.getLogger(logger_name)
        logger.addFilter(shutdown_filter)


def get_rx_cache_base() -> Path:
    """Get the base cache directory for RX.

    Priority:
    1. RX_CACHE_DIR environment variable (if set)
    2. XDG_CACHE_HOME environment variable (if set)
    3. ~/.cache (default)

    Returns:
        Path to the base cache directory (e.g., ~/.cache/rx)
    """
    # First check RX_CACHE_DIR for explicit override
    rx_cache = os.environ.get('RX_CACHE_DIR')
    if rx_cache:
        base = Path(rx_cache)
    else:
        # Fall back to XDG_CACHE_HOME or ~/.cache
        xdg_cache = os.environ.get('XDG_CACHE_HOME')
        if xdg_cache:
            base = Path(xdg_cache)
        else:
            base = Path.home() / '.cache'

    return base / 'rx'


def get_rx_cache_dir(subdir: str) -> Path:
    """Get a specific cache subdirectory for RX.

    Args:
        subdir: Subdirectory name (e.g., 'indexes', 'trace_cache')

    Returns:
        Path to the cache subdirectory, created if necessary
    """
    cache_dir = get_rx_cache_base() / subdir
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir
