"""Format deviation detector."""

import logging
import re

from .base import AnomalyDetector, LineContext, register_detector


logger = logging.getLogger(__name__)


@register_detector
class FormatDeviationDetector(AnomalyDetector):
    """Detects lines that deviate from the dominant log format pattern."""

    # Common log format components
    FORMAT_PATTERNS = [
        ('timestamp', re.compile(r'^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}')),
        ('timestamp', re.compile(r'^\[\d{4}-\d{2}-\d{2}')),
        ('timestamp', re.compile(r'^\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}')),  # Syslog
        ('level', re.compile(r'\b(DEBUG|INFO|WARN|WARNING|ERROR|FATAL|TRACE)\b', re.IGNORECASE)),
        ('bracketed', re.compile(r'^\[.*?\]')),
    ]

    # Minimum lines to establish a dominant pattern
    MIN_LINES_FOR_PATTERN = 100

    def __init__(self, filepath: str | None = None):
        """Initialize the detector.

        Args:
            filepath: Path to file being analyzed (for logging context).
        """
        self._filepath = filepath
        self._format_counts: dict[tuple[bool, ...], int] = {}
        self._total_lines = 0
        self._dominant_pattern: tuple[bool, ...] | None = None
        self._detection_count = 0
        self._pattern_locked = False
        logger.debug(f'[format_deviation] Initialized for file: {filepath}')

    @property
    def name(self) -> str:
        return 'format_deviation'

    @property
    def category(self) -> str:
        return 'format'

    @property
    def detector_description(self) -> str:
        return 'Detects lines that deviate from the dominant log format pattern in the file'

    @property
    def severity_min(self) -> float:
        return 0.3

    @property
    def severity_max(self) -> float:
        return 0.6

    @property
    def examples(self) -> list[str]:
        return ['Missing timestamp', 'Different log level format', 'Unstructured output']

    def _get_line_format(self, line: str) -> tuple[bool, ...]:
        """Get the format signature of a line."""
        return tuple(bool(p[1].search(line)) for p in self.FORMAT_PATTERNS)

    def check_line(self, ctx: LineContext) -> float | None:
        line = ctx.line

        # Skip empty lines
        if not line.strip():
            return None

        line_format = self._get_line_format(line)
        self._total_lines += 1

        # Count format occurrences
        self._format_counts[line_format] = self._format_counts.get(line_format, 0) + 1

        # Update dominant pattern periodically
        if self._total_lines >= self.MIN_LINES_FOR_PATTERN:
            if self._dominant_pattern is None or self._total_lines % 1000 == 0:
                # Find most common format
                max_count = 0
                for fmt, count in self._format_counts.items():
                    if count > max_count:
                        max_count = count
                        self._dominant_pattern = fmt

                if not self._pattern_locked and self._dominant_pattern is not None:
                    self._pattern_locked = True

        # Check for deviation from dominant pattern
        if self._dominant_pattern is not None:
            if line_format != self._dominant_pattern:
                # Calculate how different this format is
                diff_count = sum(1 for a, b in zip(line_format, self._dominant_pattern) if a != b)
                if diff_count >= 2:  # At least 2 components different
                    # Check if this is a rare format
                    format_frequency = self._format_counts.get(line_format, 0) / self._total_lines
                    if format_frequency < 0.05:  # Less than 5% of lines
                        severity = min(0.3 + (diff_count * 0.1), 0.6)
                        self._detection_count += 1
                        return severity

        return None

    def get_description(self, lines: list[str]) -> str:
        return 'Format deviation from dominant pattern'
