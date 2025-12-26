"""Base classes and data models for anomaly detection."""

import re
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass


@dataclass
class AnomalyRange:
    """Represents a detected anomaly in a file.

    Anomalies are line ranges that have been flagged by detectors as
    interesting or potentially problematic (e.g., stack traces, errors).
    """

    start_line: int  # First line of anomaly (1-based)
    end_line: int  # Last line of anomaly (inclusive, 1-based)
    start_offset: int  # Byte offset of start
    end_offset: int  # Byte offset of end
    severity: float  # 0.0 to 1.0 (higher = more severe)
    category: str  # e.g., "traceback", "error", "format_deviation"
    description: str  # Human-readable description
    detector: str  # Name of detector that found it


@dataclass
class LineContext:
    """Context provided to each anomaly detector for a line.

    This provides both the current line and surrounding context
    to allow detectors to make informed decisions.
    """

    line: str  # Current line content
    line_number: int  # 1-based line number
    byte_offset: int  # Byte offset in file
    window: deque[str]  # Sliding window of previous N lines
    line_lengths: deque[int]  # Lengths of lines in window
    avg_line_length: float  # Running average line length
    stddev_line_length: float  # Running stddev of line length


class AnomalyDetector(ABC):
    """Base class for all anomaly detectors.

    Subclass this to create custom anomaly detectors. Each detector
    should focus on detecting a specific type of anomaly.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Detector identifier (e.g., 'traceback', 'error_keyword')."""
        pass

    @property
    @abstractmethod
    def category(self) -> str:
        """Anomaly category (e.g., 'traceback', 'error', 'format')."""
        pass

    @abstractmethod
    def check_line(self, ctx: LineContext) -> float | None:
        """Check if current line is anomalous.

        Args:
            ctx: Line context with current line and surrounding context.

        Returns:
            Severity score (0.0-1.0) if anomalous, None otherwise.
        """
        pass

    def should_merge_with_previous(self, ctx: LineContext, prev_severity: float) -> bool:
        """Return True if this line should be merged with previous anomaly.

        Override this for multi-line anomalies (e.g., stack traces).

        Args:
            ctx: Current line context.
            prev_severity: Severity of the previous anomaly line.

        Returns:
            True if this line should be merged with the previous anomaly.
        """
        return False

    def get_description(self, lines: list[str]) -> str:
        """Generate description for a detected anomaly range.

        Override to provide more specific descriptions.

        Args:
            lines: List of lines in the anomaly range.

        Returns:
            Human-readable description of the anomaly.
        """
        return f'Detected by {self.name}'

    def get_prescan_patterns(self) -> list[tuple[re.Pattern, float]]:
        """Return regex patterns for ripgrep prescan optimization.

        Override to enable fast prescan using ripgrep. Return a list of
        (pattern, severity) tuples that can identify potential anomaly lines.

        Returns:
            List of (compiled regex pattern, severity) tuples, or empty list
            if this detector doesn't support prescan.
        """
        return []
