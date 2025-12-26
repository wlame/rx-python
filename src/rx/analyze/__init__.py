"""Unified analysis module for file indexing and anomaly detection.

This module provides:
- Anomaly detectors for log analysis
- Memory-efficient helpers for large files
- Parallel prescan using ripgrep
"""

from .detectors import (
    AnomalyDetector,
    AnomalyRange,
    ErrorKeywordDetector,
    FormatDeviationDetector,
    HighEntropyDetector,
    IndentationBlockDetector,
    JsonDumpDetector,
    LineContext,
    LineLengthSpikeDetector,
    TimestampGapDetector,
    TracebackDetector,
    WarningKeywordDetector,
    default_detectors,
)
from .helpers import BoundedAnomalyHeap, SparseLineOffsets
from .prescan import PrescanMatch, rg_prescan_all_detectors, rg_prescan_keywords


__all__ = [
    # Base classes and data models
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
    # Factory
    'default_detectors',
    # Helpers
    'BoundedAnomalyHeap',
    'SparseLineOffsets',
    # Prescan
    'PrescanMatch',
    'rg_prescan_all_detectors',
    'rg_prescan_keywords',
]
