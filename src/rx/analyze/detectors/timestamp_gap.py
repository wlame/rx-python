"""Timestamp gap detector."""

import os
import re
from datetime import datetime

from .base import AnomalyDetector, LineContext


class TimestampGapDetector(AnomalyDetector):
    """Detects unusual gaps in timestamps between log lines.

    Key behaviors:
    - Only searches for timestamps in the first N words of each line (default 5)
      to avoid matching timestamps in message payloads
    - Ignores gaps if the previous timestamp was too many lines ago (default 500)
    - Locks to a detected timestamp format after seeing it consistently (default 50 times)
    """

    # Common timestamp patterns
    TIMESTAMP_PATTERNS = [
        # ISO 8601: 2024-01-15T10:30:45.123Z or 2024-01-15 10:30:45
        re.compile(r'(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})'),
        # Common log format: 15/Jan/2024:10:30:45
        re.compile(r'(\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2})'),
        # Unix timestamp (10+ digits)
        re.compile(r'\b(\d{10,13})\b'),
        # HH:MM:SS format (with optional date prefix)
        re.compile(r'(\d{2}:\d{2}:\d{2})'),
    ]

    # Gap thresholds in seconds
    MIN_GAP_SECONDS = 300  # 5 minutes minimum gap to flag
    SEVERITY_SCALE = 3600  # 1 hour = max severity

    # Configuration defaults (can be overridden via env vars)
    DEFAULT_MAX_WORDS_TO_SEARCH = 5  # Only search first N words for timestamps
    DEFAULT_MAX_LINES_BETWEEN_TIMESTAMPS = 500  # Ignore gap if too many lines apart
    DEFAULT_FORMAT_LOCK_THRESHOLD = 50  # Lock to format after N consistent matches

    def __init__(self):
        self._last_timestamp: float | None = None
        self._last_timestamp_line: int = 0  # Line number of last timestamp
        self._timestamp_format: int | None = None  # Which pattern matched
        self._format_counts: dict[int, int] = {}  # Count per format
        self._locked_format: int | None = None  # Locked format after threshold

        # Load config from env vars
        self._max_words = int(os.environ.get('RX_TIMESTAMP_MAX_WORDS', str(self.DEFAULT_MAX_WORDS_TO_SEARCH)))
        self._max_lines_between = int(
            os.environ.get('RX_TIMESTAMP_MAX_LINES_BETWEEN', str(self.DEFAULT_MAX_LINES_BETWEEN_TIMESTAMPS))
        )
        self._format_lock_threshold = int(
            os.environ.get('RX_TIMESTAMP_FORMAT_LOCK_THRESHOLD', str(self.DEFAULT_FORMAT_LOCK_THRESHOLD))
        )

    @property
    def name(self) -> str:
        return 'timestamp_gap'

    @property
    def category(self) -> str:
        return 'timing'

    def _get_line_prefix(self, line: str) -> str:
        """Extract the first N words from a line for timestamp searching.

        This avoids matching timestamps that appear in message payloads
        rather than as the log line's actual timestamp.

        Args:
            line: The full log line

        Returns:
            The first N words of the line, joined by spaces
        """
        # Split by whitespace (spaces and tabs)
        words = line.split()[: self._max_words]
        return ' '.join(words)

    def _parse_timestamp(self, line: str) -> tuple[float | None, int | None]:
        """Try to parse a timestamp from the line prefix.

        Only searches the first N words of the line to avoid matching
        timestamps in message payloads.

        Returns:
            Tuple of (unix_timestamp, pattern_index) or (None, None)
        """
        # Only search in the line prefix (first N words)
        prefix = self._get_line_prefix(line)

        # If we've locked to a format, only try that format
        patterns_to_try = (
            [(self._locked_format, self.TIMESTAMP_PATTERNS[self._locked_format])]
            if self._locked_format is not None
            else enumerate(self.TIMESTAMP_PATTERNS)
        )

        for i, pattern in patterns_to_try:
            match = pattern.search(prefix)
            if match:
                ts_str = match.group(1)
                try:
                    # ISO 8601
                    if i == 0:
                        ts_str = ts_str.replace('T', ' ')
                        dt = datetime.strptime(ts_str[:19], '%Y-%m-%d %H:%M:%S')
                        return dt.timestamp(), i
                    # Common log format
                    elif i == 1:
                        dt = datetime.strptime(ts_str, '%d/%b/%Y:%H:%M:%S')
                        return dt.timestamp(), i
                    # Unix timestamp
                    elif i == 2:
                        ts = int(ts_str)
                        # Convert milliseconds to seconds if needed
                        if ts > 1e12:
                            ts = ts / 1000
                        return float(ts), i
                    # HH:MM:SS - can only detect gaps within same day
                    elif i == 3:
                        parts = ts_str.split(':')
                        seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                        return float(seconds), i
                except (ValueError, IndexError):
                    continue
        return None, None

    def _update_format_counts(self, fmt_idx: int) -> None:
        """Update format counts and potentially lock to a format.

        After seeing the same format N times, we lock to that format
        to avoid confusion from different timestamp styles in the file.

        Args:
            fmt_idx: The format index that was matched
        """
        self._format_counts[fmt_idx] = self._format_counts.get(fmt_idx, 0) + 1

        # Check if we should lock to this format
        if self._locked_format is None:
            if self._format_counts[fmt_idx] >= self._format_lock_threshold:
                self._locked_format = fmt_idx

    def check_line(self, ctx: LineContext) -> float | None:
        timestamp, fmt_idx = self._parse_timestamp(ctx.line)

        if timestamp is None:
            return None

        # Update format counts and potentially lock
        self._update_format_counts(fmt_idx)

        if self._last_timestamp is None:
            self._last_timestamp = timestamp
            self._last_timestamp_line = ctx.line_number
            self._timestamp_format = fmt_idx
            return None

        # Only compare timestamps of the same format
        if fmt_idx != self._timestamp_format:
            self._last_timestamp = timestamp
            self._last_timestamp_line = ctx.line_number
            self._timestamp_format = fmt_idx
            return None

        # Check if too many lines have passed since last timestamp
        lines_since_last = ctx.line_number - self._last_timestamp_line
        if lines_since_last > self._max_lines_between:
            # Too many lines apart - don't treat as a gap, just update state
            self._last_timestamp = timestamp
            self._last_timestamp_line = ctx.line_number
            return None

        # Calculate gap
        gap = abs(timestamp - self._last_timestamp)
        self._last_timestamp = timestamp
        self._last_timestamp_line = ctx.line_number

        # For HH:MM:SS format, handle day rollover
        if fmt_idx == 3 and gap > 43200:  # More than 12 hours
            gap = 86400 - gap  # Assume day rollover

        if gap >= self.MIN_GAP_SECONDS:
            # Severity scales with gap size, max at 1 hour
            severity = min(0.3 + (gap / self.SEVERITY_SCALE) * 0.5, 0.8)
            return severity

        return None

    def get_description(self, lines: list[str]) -> str:
        return 'Timestamp gap detected'
