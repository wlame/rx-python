"""Timestamp gap detector."""

import re
from datetime import datetime

from .base import AnomalyDetector, LineContext


class TimestampGapDetector(AnomalyDetector):
    """Detects unusual gaps in timestamps between log lines."""

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

    def __init__(self):
        self._last_timestamp: float | None = None
        self._timestamp_format: int | None = None  # Which pattern matched
        self._lines_without_timestamp = 0

    @property
    def name(self) -> str:
        return 'timestamp_gap'

    @property
    def category(self) -> str:
        return 'timing'

    def _parse_timestamp(self, line: str) -> tuple[float | None, int | None]:
        """Try to parse a timestamp from the line.

        Returns:
            Tuple of (unix_timestamp, pattern_index) or (None, None)
        """
        for i, pattern in enumerate(self.TIMESTAMP_PATTERNS):
            match = pattern.search(line)
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

    def check_line(self, ctx: LineContext) -> float | None:
        timestamp, fmt_idx = self._parse_timestamp(ctx.line)

        if timestamp is None:
            self._lines_without_timestamp += 1
            return None

        # Reset counter when we see a timestamp
        self._lines_without_timestamp = 0

        if self._last_timestamp is None:
            self._last_timestamp = timestamp
            self._timestamp_format = fmt_idx
            return None

        # Only compare timestamps of the same format
        if fmt_idx != self._timestamp_format:
            self._last_timestamp = timestamp
            self._timestamp_format = fmt_idx
            return None

        # Calculate gap
        gap = abs(timestamp - self._last_timestamp)
        self._last_timestamp = timestamp

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
