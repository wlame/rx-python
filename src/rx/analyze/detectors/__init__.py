"""Anomaly detection modules.

This package contains all anomaly detectors for log analysis.
"""

from .base import AnomalyDetector, AnomalyRange, LineContext
from .error_keyword import ErrorKeywordDetector
from .format_deviation import FormatDeviationDetector
from .high_entropy import HighEntropyDetector
from .indentation import IndentationBlockDetector
from .json_dump import JsonDumpDetector
from .line_length import LineLengthSpikeDetector
from .timestamp_gap import TimestampGapDetector
from .traceback import TracebackDetector
from .warning_keyword import WarningKeywordDetector


__all__ = [
    # Base classes
    'AnomalyDetector',
    'AnomalyRange',
    'LineContext',
    # Detectors
    'ErrorKeywordDetector',
    'FormatDeviationDetector',
    'HighEntropyDetector',
    'IndentationBlockDetector',
    'JsonDumpDetector',
    'LineLengthSpikeDetector',
    'TimestampGapDetector',
    'TracebackDetector',
    'WarningKeywordDetector',
]


def default_detectors() -> list[AnomalyDetector]:
    """Get list of default anomaly detectors.

    Returns:
        List of instantiated detector objects.
    """
    return [
        TracebackDetector(),
        ErrorKeywordDetector(),
        WarningKeywordDetector(),
        LineLengthSpikeDetector(),
        IndentationBlockDetector(),
        TimestampGapDetector(),
        HighEntropyDetector(),
        JsonDumpDetector(),
        FormatDeviationDetector(),
    ]
