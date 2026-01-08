"""Prefix deviation detector.

Detects log lines that don't match the dominant prefix pattern established
for the file. This is useful for finding:
- Stack traces and exceptions
- Unformatted debug output
- Multi-line message continuations
- Corrupted or malformed log lines
"""

import logging
import re

from .base import AnomalyDetector, LineContext, register_detector


logger = logging.getLogger(__name__)


@register_detector
class PrefixDeviationDetector(AnomalyDetector):
    """Detects lines that don't match the dominant prefix pattern.

    This detector requires a prefix regex pattern to be set before use.
    The pattern is typically extracted using PrefixPatternExtractor during
    the analysis phase and cached in the file index.

    Lines that don't match the prefix are flagged as anomalies with
    relatively low severity (0.3) since many legitimate log entries
    may span multiple lines.
    """

    def __init__(
        self,
        prefix_regex: str | None = None,
        prefix_length: int = 0,
        severity: float = 0.3,
        filepath: str | None = None,
    ):
        """Initialize the detector.

        Args:
            prefix_regex: Regex pattern to match expected prefix.
                If None, detector will not flag any lines.
            prefix_length: Expected prefix length (for description).
            severity: Severity score for non-matching lines (0.0-1.0).
            filepath: Path to file being analyzed (for logging context).
        """
        self._filepath = filepath
        self._prefix_regex = prefix_regex
        self._prefix_length = prefix_length
        self._severity = severity
        self._pattern: re.Pattern | None = None
        self._detection_count = 0
        self._merge_count = 0
        self._continuation_count = 0

        if prefix_regex:
            try:
                self._pattern = re.compile(prefix_regex)
            except re.error as e:
                logger.warning(f'[prefix_deviation] {filepath}: Invalid regex pattern: {e}')
                self._pattern = None

    @property
    def name(self) -> str:
        return 'prefix_deviation'

    @property
    def category(self) -> str:
        return 'format'

    @property
    def detector_description(self) -> str:
        return "Detects lines that don't match the dominant prefix pattern (timestamp, level, etc.)"

    @property
    def severity_min(self) -> float:
        return 0.24

    @property
    def severity_max(self) -> float:
        return 0.3

    @property
    def examples(self) -> list[str]:
        return ['Stack trace continuation', 'Multi-line message', 'Unformatted debug output']

    @property
    def prefix_regex(self) -> str | None:
        """Get the current prefix regex pattern."""
        return self._prefix_regex

    @prefix_regex.setter
    def prefix_regex(self, value: str | None) -> None:
        """Set the prefix regex pattern."""
        self._prefix_regex = value
        if value:
            try:
                self._pattern = re.compile(value)
            except re.error:
                self._pattern = None
        else:
            self._pattern = None

    def is_configured(self) -> bool:
        """Check if detector has a valid prefix pattern configured."""
        return self._pattern is not None

    def check_line(self, ctx: LineContext) -> float | None:
        """Check if line matches the expected prefix pattern.

        Args:
            ctx: Line context.

        Returns:
            Severity if line doesn't match prefix, None if it matches.
        """
        if not self._pattern:
            return None

        line = ctx.line.rstrip()
        if not line:
            return None  # Empty lines are not anomalies

        # Check if line matches the prefix pattern
        if self._pattern.match(line):
            return None  # Matches expected format

        # Line doesn't match - but check if it might be a continuation
        # Continuation lines often start with whitespace
        if line[0].isspace():
            # Indented line - likely continuation, slightly lower severity
            self._continuation_count += 1
            severity = self._severity * 0.8
            return severity

        self._detection_count += 1
        return self._severity

    def should_merge_with_previous(self, ctx: LineContext, prev_severity: float) -> bool:
        """Merge consecutive non-matching lines.

        Non-matching lines are often part of multi-line content like
        stack traces, so we merge them together.

        Args:
            ctx: Current line context.
            prev_severity: Severity of previous anomaly.

        Returns:
            True if this line should merge with previous anomaly.
        """
        if not self._pattern:
            return False

        line = ctx.line.rstrip()
        if not line:
            return False  # Don't merge empty lines

        # If current line also doesn't match prefix, merge with previous
        if not self._pattern.match(line):
            self._merge_count += 1
            return True

        return False

    def get_description(self, lines: list[str]) -> str:
        """Generate description for prefix deviation anomaly.

        Args:
            lines: Lines in the anomaly range.

        Returns:
            Human-readable description.
        """
        if len(lines) == 1:
            # Single line - show preview
            preview = lines[0][:60]
            if len(lines[0]) > 60:
                preview += '...'
            return f"Line doesn't match expected prefix format: {preview}"
        else:
            # Multi-line block
            first_line = lines[0][:40]
            if len(lines[0]) > 40:
                first_line += '...'
            return f'Block of {len(lines)} lines without expected prefix: {first_line}'

    def get_prescan_patterns(self) -> list[tuple[re.Pattern, float]]:
        """Return patterns for ripgrep prescan.

        For prefix deviation, we can't easily prescan because we're looking
        for lines that DON'T match a pattern. The full line-by-line check
        is required.

        Returns:
            Empty list (prescan not supported).
        """
        return []
