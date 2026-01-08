"""Experimental features!! File analysis module"""

import heapq
import json
import logging
import os
import random
import re
import statistics
import subprocess
import tempfile
from abc import ABC, abstractmethod
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from time import time
from typing import Any

from rx import seekable_index, seekable_zstd
from rx.compression import decompress_to_file, detect_compression, is_compressed
from rx.file_utils import MAX_SUBPROCESSES, FileTask, create_file_tasks, is_text_file


logger = logging.getLogger(__name__)


# =============================================================================
# Anomaly Detection Data Models
# =============================================================================


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


# =============================================================================
# Phase 1 Detectors
# =============================================================================


class TracebackDetector(AnomalyDetector):
    """Detects stack traces from Python, Java, JavaScript, Go, and Rust."""

    # Patterns for detecting start of tracebacks
    TRACEBACK_START_PATTERNS = {
        'python': [
            re.compile(r'^Traceback \(most recent call last\):'),
            re.compile(r'^\w+Error:'),
            re.compile(r'^\w+Exception:'),
        ],
        'java': [
            re.compile(r'^Exception in thread'),
            re.compile(r'^Caused by:'),
            re.compile(r'^\w+Exception:'),
            re.compile(r'^\w+Error:'),
        ],
        'javascript': [
            re.compile(r'^Error:'),
            re.compile(r'^TypeError:'),
            re.compile(r'^ReferenceError:'),
            re.compile(r'^SyntaxError:'),
        ],
        'go': [
            re.compile(r'^panic:'),
            re.compile(r'^goroutine \d+ \['),
        ],
        'rust': [
            re.compile(r'^thread .* panicked at'),
            re.compile(r'^stack backtrace:'),
        ],
    }

    # Patterns for continuation lines (part of a traceback)
    TRACEBACK_CONTINUATION_PATTERNS = [
        re.compile(r'^  File ".*", line \d+'),  # Python
        re.compile(r'^\s+at \w+'),  # Java/JavaScript
        re.compile(r'^\s+\.\.\. \d+ more'),  # Java truncated
        re.compile(r'^\s+raise \w+'),  # Python raise
        re.compile(r'^\s+\d+:\s+0x'),  # Rust/Go stack address
        re.compile(r'^\t'),  # Indented continuation
    ]

    @property
    def name(self) -> str:
        return 'traceback'

    @property
    def category(self) -> str:
        return 'traceback'

    def check_line(self, ctx: LineContext) -> float | None:
        line = ctx.line.rstrip()

        # Check for traceback start patterns
        for lang_patterns in self.TRACEBACK_START_PATTERNS.values():
            for pattern in lang_patterns:
                if pattern.match(line):
                    return 0.9

        return None

    def should_merge_with_previous(self, ctx: LineContext, prev_severity: float) -> bool:
        line = ctx.line.rstrip()

        # Check for continuation patterns
        for pattern in self.TRACEBACK_CONTINUATION_PATTERNS:
            if pattern.match(line):
                return True

        # Also merge if previous was a traceback and this looks like an error message
        if prev_severity >= 0.9:
            # Check if this is an exception message (starts with exception name)
            if re.match(r'^\w+(Error|Exception):', line):
                return True

        return False

    def get_description(self, lines: list[str]) -> str:
        # Try to identify the language and error type
        first_line = lines[0].strip() if lines else ''
        if 'Traceback' in first_line:
            return 'Python traceback'
        elif 'Exception in thread' in first_line:
            return 'Java exception'
        elif first_line.startswith('panic:'):
            return 'Go panic'
        elif 'panicked at' in first_line:
            return 'Rust panic'
        elif any(first_line.startswith(err) for err in ['TypeError:', 'ReferenceError:', 'SyntaxError:']):
            return 'JavaScript error'
        return f'Stack trace ({len(lines)} lines)'


class ErrorKeywordDetector(AnomalyDetector):
    """Detects lines containing error-related keywords."""

    # Keywords with their severity scores
    ERROR_KEYWORDS = [
        (re.compile(r'\bFATAL\b', re.IGNORECASE), 0.9),
        (re.compile(r'\bCRITICAL\b', re.IGNORECASE), 0.85),
        (re.compile(r'\bERROR\b', re.IGNORECASE), 0.7),
        (re.compile(r'\bException\b'), 0.7),
        (re.compile(r'\bFailed\b', re.IGNORECASE), 0.6),
        (re.compile(r'\bFailure\b', re.IGNORECASE), 0.6),
        (re.compile(r'\bAborted\b', re.IGNORECASE), 0.6),
        (re.compile(r'\bCrash(ed)?\b', re.IGNORECASE), 0.7),
        (re.compile(r'\bSegmentation fault\b', re.IGNORECASE), 0.9),
        (re.compile(r'\bOOM\b'), 0.8),  # Out of memory
        (re.compile(r'\bOut of memory\b', re.IGNORECASE), 0.8),
    ]

    @property
    def name(self) -> str:
        return 'error_keyword'

    @property
    def category(self) -> str:
        return 'error'

    def check_line(self, ctx: LineContext) -> float | None:
        line = ctx.line

        # Find the highest severity match
        max_severity = None
        for pattern, severity in self.ERROR_KEYWORDS:
            if pattern.search(line):
                if max_severity is None or severity > max_severity:
                    max_severity = severity

        return max_severity

    def get_description(self, lines: list[str]) -> str:
        if not lines:
            return 'Error keyword detected'

        # Find what keyword was matched
        first_line = lines[0]
        for pattern, _ in self.ERROR_KEYWORDS:
            match = pattern.search(first_line)
            if match:
                return f'Error: {match.group(0)}'
        return 'Error keyword detected'


class LineLengthSpikeDetector(AnomalyDetector):
    """Detects lines that are significantly longer than average."""

    # Minimum stddev threshold to avoid flagging in uniform files
    MIN_STDDEV_THRESHOLD = 40
    # How many standard deviations above mean to flag
    STDDEV_MULTIPLIER = 5.0

    @property
    def name(self) -> str:
        return 'line_length_spike'

    @property
    def category(self) -> str:
        return 'format'

    def check_line(self, ctx: LineContext) -> float | None:
        line_len = len(ctx.line.rstrip())

        # Need enough data and variance to make a judgment
        if ctx.stddev_line_length < self.MIN_STDDEV_THRESHOLD:
            return None

        if ctx.avg_line_length <= 0:
            return None

        # Check if line is significantly longer than average
        threshold = ctx.avg_line_length + (self.STDDEV_MULTIPLIER * ctx.stddev_line_length)
        if line_len > threshold:
            # Severity based on how far above threshold
            deviation = (line_len - ctx.avg_line_length) / ctx.stddev_line_length
            severity = min(0.3 + (deviation - self.STDDEV_MULTIPLIER) * 0.1, 0.7)
            return severity

        return None

    def get_description(self, lines: list[str]) -> str:
        if lines:
            length = len(lines[0].rstrip())
            return f'Unusually long line ({length} chars)'
        return 'Unusually long line'


class IndentationBlockDetector(AnomalyDetector):
    """Detects unusual indentation patterns that may indicate embedded data."""

    # Minimum lines to consider a block
    MIN_BLOCK_SIZE = 3
    # Minimum indentation to flag
    MIN_INDENTATION = 4

    @property
    def name(self) -> str:
        return 'indentation_block'

    @property
    def category(self) -> str:
        return 'multiline'

    def _get_indentation(self, line: str) -> int:
        """Get the number of leading spaces (tabs count as 4 spaces)."""
        count = 0
        for char in line:
            if char == ' ':
                count += 1
            elif char == '\t':
                count += 4
            else:
                break
        return count

    def check_line(self, ctx: LineContext) -> float | None:
        line = ctx.line.rstrip()

        # Empty lines don't trigger detection
        if not line.strip():
            return None

        indent = self._get_indentation(line)
        if indent < self.MIN_INDENTATION:
            return None

        # Check if previous lines in window also have unusual indentation
        if len(ctx.window) >= self.MIN_BLOCK_SIZE - 1:
            indented_count = 0
            for prev_line in list(ctx.window)[-self.MIN_BLOCK_SIZE + 1 :]:
                prev_stripped = prev_line.rstrip()
                if prev_stripped and self._get_indentation(prev_stripped) >= self.MIN_INDENTATION:
                    indented_count += 1

            # If we have a consistent indentation block
            if indented_count >= self.MIN_BLOCK_SIZE - 1:
                return 0.4

        return None

    def should_merge_with_previous(self, ctx: LineContext, prev_severity: float) -> bool:
        line = ctx.line.rstrip()
        if not line.strip():
            return False
        return self._get_indentation(line) >= self.MIN_INDENTATION

    def get_description(self, lines: list[str]) -> str:
        return f'Indented block ({len(lines)} lines)'


class WarningKeywordDetector(AnomalyDetector):
    """Detects lines containing warning-related keywords."""

    WARNING_KEYWORDS = [
        (re.compile(r'\bWARNING\b', re.IGNORECASE), 0.4),
        (re.compile(r'\bWARN\b', re.IGNORECASE), 0.4),
        (re.compile(r'\[W\]'), 0.4),  # Common log format [W]
        (re.compile(r'\bCaution\b', re.IGNORECASE), 0.35),
        (re.compile(r'\bDeprecated\b', re.IGNORECASE), 0.3),
        (re.compile(r'\bDeprecation\b', re.IGNORECASE), 0.3),
    ]

    @property
    def name(self) -> str:
        return 'warning_keyword'

    @property
    def category(self) -> str:
        return 'warning'

    def check_line(self, ctx: LineContext) -> float | None:
        line = ctx.line

        max_severity = None
        for pattern, severity in self.WARNING_KEYWORDS:
            if pattern.search(line):
                if max_severity is None or severity > max_severity:
                    max_severity = severity

        return max_severity

    def get_description(self, lines: list[str]) -> str:
        if not lines:
            return 'Warning keyword detected'

        first_line = lines[0]
        for pattern, _ in self.WARNING_KEYWORDS:
            match = pattern.search(first_line)
            if match:
                return f'Warning: {match.group(0)}'
        return 'Warning keyword detected'


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
                        from datetime import datetime

                        dt = datetime.strptime(ts_str[:19], '%Y-%m-%d %H:%M:%S')
                        return dt.timestamp(), i
                    # Common log format
                    elif i == 1:
                        from datetime import datetime

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


class HighEntropyDetector(AnomalyDetector):
    """Detects high-entropy strings that might be secrets, tokens, or API keys."""

    # Minimum length of high-entropy substring to flag
    MIN_TOKEN_LENGTH = 40
    # Entropy threshold (bits per character)
    ENTROPY_THRESHOLD = 4.0
    # Patterns that often contain secrets
    SECRET_CONTEXT_PATTERNS = [
        re.compile(r'(?:api[_-]?key|apikey|secret|password|token|auth|credential|private[_-]?key)', re.IGNORECASE),
        re.compile(r'(?:bearer|authorization)\s*[:=]', re.IGNORECASE),
        re.compile(r'-----BEGIN\s+\w+\s+PRIVATE\s+KEY-----'),
    ]
    # Patterns for base64/hex strings
    HIGH_ENTROPY_PATTERNS = [
        re.compile(r'[A-Za-z0-9+/]{40,}={0,2}'),  # Base64
        re.compile(r'[a-fA-F0-9]{32,}'),  # Hex
        re.compile(r'[A-Za-z0-9_-]{20,}'),  # URL-safe base64 / API keys
    ]

    @property
    def name(self) -> str:
        return 'high_entropy'

    @property
    def category(self) -> str:
        return 'security'

    def _calculate_entropy(self, s: str) -> float:
        """Calculate Shannon entropy of a string."""
        if not s:
            return 0.0

        import math
        from collections import Counter

        counts = Counter(s)
        length = len(s)
        entropy = 0.0

        for count in counts.values():
            if count > 0:
                p = count / length
                entropy -= p * math.log2(p)

        return entropy

    def check_line(self, ctx: LineContext) -> float | None:
        line = ctx.line

        # First check for secret context patterns (higher severity)
        for pattern in self.SECRET_CONTEXT_PATTERNS:
            if pattern.search(line):
                # Found secret context, look for high-entropy value
                for hp in self.HIGH_ENTROPY_PATTERNS:
                    match = hp.search(line)
                    if match:
                        token = match.group(0)
                        if len(token) >= self.MIN_TOKEN_LENGTH:
                            entropy = self._calculate_entropy(token)
                            if entropy >= self.ENTROPY_THRESHOLD:
                                return 0.85  # High severity for secrets in context
                return 0.6  # Medium severity for secret context without clear token

        # Check for high-entropy strings without context
        for pattern in self.HIGH_ENTROPY_PATTERNS:
            for match in pattern.finditer(line):
                token = match.group(0)
                if len(token) >= self.MIN_TOKEN_LENGTH:
                    entropy = self._calculate_entropy(token)
                    if entropy >= self.ENTROPY_THRESHOLD:
                        # Lower severity without secret context
                        return 0.5

        return None

    def get_description(self, lines: list[str]) -> str:
        if not lines:
            return 'Potential secret or token detected'

        first_line = lines[0]
        for pattern in self.SECRET_CONTEXT_PATTERNS:
            if pattern.search(first_line):
                return 'Potential secret in context'
        return 'High-entropy string detected'


class JsonDumpDetector(AnomalyDetector):
    """Detects embedded JSON objects in log lines."""

    # Patterns for JSON-like content
    JSON_START_PATTERNS = [
        re.compile(r'^\s*\{'),  # Line starts with {
        re.compile(r':\s*\{'),  # JSON object as value
        re.compile(r'^\s*\['),  # Line starts with [
        re.compile(r':\s*\['),  # JSON array as value
    ]

    # Minimum length to consider it a meaningful JSON dump
    MIN_JSON_LENGTH = 50

    @property
    def name(self) -> str:
        return 'json_dump'

    @property
    def category(self) -> str:
        return 'format'

    def check_line(self, ctx: LineContext) -> float | None:
        line = ctx.line.rstrip()

        # Skip short lines
        if len(line) < self.MIN_JSON_LENGTH:
            return None

        # Check for JSON patterns
        for pattern in self.JSON_START_PATTERNS:
            if pattern.search(line):
                # Verify it looks like actual JSON (has key-value pairs)
                if '":' in line or "': " in line:
                    # Longer JSON dumps get slightly higher severity
                    if len(line) > 500:
                        return 0.4
                    elif len(line) > 200:
                        return 0.35
                    else:
                        return 0.3

        return None

    def should_merge_with_previous(self, ctx: LineContext, prev_severity: float) -> bool:
        """Merge continuation of multi-line JSON."""
        line = ctx.line.rstrip()
        if not line:
            return False

        # Check if this looks like JSON continuation
        stripped = line.lstrip()
        if stripped.startswith(('"', '{', '}', '[', ']', ',')):
            return True

        return False

    def get_description(self, lines: list[str]) -> str:
        total_chars = sum(len(line) for line in lines)
        return f'Embedded JSON ({total_chars} chars)'


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

    def __init__(self):
        self._format_counts: dict[tuple[bool, ...], int] = {}
        self._total_lines = 0
        self._dominant_pattern: tuple[bool, ...] | None = None

    @property
    def name(self) -> str:
        return 'format_deviation'

    @property
    def category(self) -> str:
        return 'format'

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
                        return severity

        return None

    def get_description(self, lines: list[str]) -> str:
        return 'Format deviation from dominant pattern'


# =============================================================================
# Memory-Efficient Anomaly Collection
# =============================================================================


class BoundedAnomalyHeap:
    """A bounded priority queue for anomalies that prunes low-severity entries.

    Uses a min-heap based on severity, so we can efficiently remove the
    lowest-severity anomalies when the heap exceeds capacity.

    This prevents OOM on large files by maintaining at most max_size anomalies
    in memory at any time.
    """

    def __init__(self, max_size: int = 100_000):
        """Initialize the bounded heap.

        Args:
            max_size: Maximum number of anomalies to keep. When exceeded,
                     lowest-severity anomalies are pruned.
        """
        self.max_size = max_size
        # Min-heap: (severity, line_num, detector_name, byte_offset, line_text)
        # We use negative severity for max-heap behavior when we want highest severity
        # But for pruning we want min-heap to remove lowest severity
        self._heap: list[tuple[float, int, str, int, str]] = []
        self._min_severity = 0.0  # Track minimum severity in heap for fast rejection

    def push(self, detector_name: str, line_num: int, byte_offset: int, severity: float, line_text: str) -> bool:
        """Add an anomaly to the heap.

        Args:
            detector_name: Name of the detector that found the anomaly
            line_num: Line number (1-based)
            byte_offset: Byte offset in file
            severity: Severity score (0.0 to 1.0)
            line_text: The line content

        Returns:
            True if the anomaly was added, False if rejected (severity too low)
        """
        # Fast rejection: if heap is full and this severity is too low, skip
        if len(self._heap) >= self.max_size and severity <= self._min_severity:
            return False

        # Add to heap (min-heap by severity)
        entry = (severity, line_num, detector_name, byte_offset, line_text)
        heapq.heappush(self._heap, entry)

        # Prune if over capacity
        if len(self._heap) > self.max_size:
            # Remove lowest severity entry
            heapq.heappop(self._heap)
            # Update min severity threshold
            if self._heap:
                self._min_severity = self._heap[0][0]

        return True

    def prune_if_needed(self, total_lines: int, density_threshold: float = 0.01) -> None:
        """Prune anomalies if they exceed density threshold.

        Args:
            total_lines: Total lines processed so far
            density_threshold: Maximum fraction of lines that should be anomalies
        """
        if total_lines <= 0:
            return

        max_anomalies = max(100, int(total_lines * density_threshold))
        while len(self._heap) > max_anomalies:
            heapq.heappop(self._heap)

        if self._heap:
            self._min_severity = self._heap[0][0]

    def get_all(self) -> list[tuple[str, int, int, float, str]]:
        """Get all anomalies as a list of (detector_name, line_num, byte_offset, severity, line_text).

        Returns:
            List of anomalies sorted by line number
        """
        # Convert from heap format to expected format and sort by line number
        result = [(name, line, offset, sev, text) for sev, line, name, offset, text in self._heap]
        result.sort(key=lambda x: x[1])  # Sort by line number
        return result

    def __len__(self) -> int:
        return len(self._heap)


class SparseLineOffsets:
    """Memory-efficient storage for line offsets.

    Only stores offsets for lines that are actually referenced (anomaly lines).
    Uses a sliding window for recent offsets to support merging adjacent anomalies.
    """

    def __init__(self, window_size: int = 100):
        """Initialize sparse storage.

        Args:
            window_size: Number of recent line offsets to keep in window
        """
        self.window_size = window_size
        # Sparse storage for anomaly line offsets
        self._offsets: dict[int, int] = {}
        # Sliding window of recent (line_num, offset) pairs
        self._recent: deque[tuple[int, int]] = deque(maxlen=window_size)

    def record(self, line_num: int, offset: int) -> None:
        """Record a line offset in the sliding window."""
        self._recent.append((line_num, offset))

    def mark_anomaly(self, line_num: int) -> None:
        """Mark a line as containing an anomaly, preserving its offset."""
        # Check if it's in recent window
        for ln, off in self._recent:
            if ln == line_num:
                self._offsets[line_num] = off
                return
        # If not found, it's a programming error but don't crash
        logger.warning(f'Line {line_num} not found in recent window for offset tracking')

    def get(self, line_num: int, default: int = 0) -> int:
        """Get the byte offset for a line number."""
        # First check anomaly offsets
        if line_num in self._offsets:
            return self._offsets[line_num]
        # Then check recent window
        for ln, off in self._recent:
            if ln == line_num:
                return off
        return default

    def __contains__(self, line_num: int) -> bool:
        if line_num in self._offsets:
            return True
        for ln, _ in self._recent:
            if ln == line_num:
                return True
        return False

    def __len__(self) -> int:
        return len(self._offsets)


def rg_prescan_keywords(filepath: str, patterns: list[tuple[re.Pattern, float]]) -> set[int]:
    """Use ripgrep to quickly find lines matching error keywords.

    This is MUCH faster than checking each line with Python regex,
    especially for large files. Ripgrep uses SIMD and parallel processing.

    Args:
        filepath: Path to the file to scan
        patterns: List of (compiled_pattern, severity) tuples

    Returns:
        Set of line numbers (1-based) that match any keyword pattern
    """
    if not patterns:
        return set()

    # Build rg command with all patterns
    # Use -n for line numbers, -o for only matching (faster)
    rg_cmd = ['rg', '--line-number', '--no-heading', '--color=never']

    # Add each pattern - convert Python regex to rg pattern
    for pattern, _ in patterns:
        # Get the pattern string from compiled regex
        rg_cmd.extend(['-e', pattern.pattern])

    rg_cmd.append(filepath)

    try:
        result = subprocess.run(
            rg_cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout for very large files
        )

        matching_lines: set[int] = set()

        # Parse output: "line_num:matched_text"
        for line in result.stdout.splitlines():
            if ':' in line:
                try:
                    line_num_str = line.split(':', 1)[0]
                    matching_lines.add(int(line_num_str))
                except (ValueError, IndexError):
                    continue

        logger.info(f'rg prescan found {len(matching_lines)} lines with error keywords in {filepath}')
        return matching_lines

    except subprocess.TimeoutExpired:
        logger.warning(f'rg prescan timed out for {filepath}, falling back to line-by-line')
        return set()
    except FileNotFoundError:
        logger.warning('ripgrep (rg) not found, falling back to line-by-line keyword detection')
        return set()
    except Exception as e:
        logger.warning(f'rg prescan failed: {e}, falling back to line-by-line')
        return set()


@dataclass
class PrescanMatch:
    """A match from rg prescan with detector info."""

    line_num: int
    byte_offset: int
    detector_name: str
    severity: float
    line_text: str


def _prescan_chunk_worker(
    task: FileTask,
    patterns: list[str],
    pattern_to_detector: dict[str, tuple[str, float]],
) -> tuple[FileTask, list[PrescanMatch], float]:
    """Worker function to prescan a single file chunk with ripgrep.

    Uses dd | rg --json pipeline similar to rx trace for parallel processing.

    Args:
        task: FileTask defining the chunk to process
        patterns: List of regex pattern strings
        pattern_to_detector: Mapping from pattern -> (detector_name, severity)

    Returns:
        Tuple of (task, list_of_matches, execution_time)
    """
    start_time = time()
    thread_id = threading.current_thread().name

    logger.debug(f'[PRESCAN {thread_id}] Processing chunk {task.task_id}: offset={task.offset}, count={task.count}')

    try:
        # Calculate dd block parameters (same approach as rx trace)
        bs = 1024 * 1024  # 1MB block size
        skip_blocks = task.offset // bs
        skip_remainder = task.offset % bs
        actual_dd_offset = skip_blocks * bs
        count_blocks = (task.count + skip_remainder + bs - 1) // bs

        # Run dd | rg --json pipeline
        dd_proc = subprocess.Popen(
            ['dd', f'if={task.filepath}', f'bs={bs}', f'skip={skip_blocks}', f'count={count_blocks}', 'status=none'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Build ripgrep command with --json for structured output
        rg_cmd = ['rg', '--json', '--no-heading', '--color=never']
        for pattern in patterns:
            rg_cmd.extend(['-e', pattern])
        rg_cmd.append('-')  # Read from stdin

        rg_proc = subprocess.Popen(
            rg_cmd,
            stdin=dd_proc.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if dd_proc.stdout:
            dd_proc.stdout.close()

        # Parse JSON output
        matches: list[PrescanMatch] = []

        for line in rg_proc.stdout or []:
            try:
                data = json.loads(line)
                if data.get('type') != 'match':
                    continue

                match_data = data.get('data', {})
                rg_offset = match_data.get('absolute_offset', 0)

                # Convert to absolute offset in file
                absolute_offset = actual_dd_offset + rg_offset

                # Only include matches within this task's range
                if not (task.offset <= absolute_offset < task.offset + task.count):
                    continue

                line_text = match_data.get('lines', {}).get('text', '')

                # Find which detector this pattern belongs to
                for pattern, (detector_name, severity) in pattern_to_detector.items():
                    if re.search(pattern, line_text):
                        matches.append(
                            PrescanMatch(
                                line_num=-1,  # Will be calculated from offset
                                byte_offset=absolute_offset,
                                detector_name=detector_name,
                                severity=severity,
                                line_text=line_text.rstrip('\n\r'),
                            )
                        )
                        break

            except json.JSONDecodeError:
                continue

        rg_proc.wait()
        dd_proc.wait()

        elapsed = time() - start_time
        logger.debug(f'[PRESCAN {thread_id}] Chunk {task.task_id} completed: {len(matches)} matches in {elapsed:.3f}s')

        return (task, matches, elapsed)

    except Exception as e:
        elapsed = time() - start_time
        logger.error(f'[PRESCAN {thread_id}] Chunk {task.task_id} failed: {e}')
        return (task, [], elapsed)


def rg_prescan_all_detectors(
    filepath: str,
    detectors: list[AnomalyDetector],
) -> dict[str, list[PrescanMatch]]:
    """Use ripgrep to prescan file for ALL regex-based anomaly patterns in PARALLEL.

    This uses the same chunked parallel processing approach as rx trace:
    - Split file into chunks using create_file_tasks()
    - Process each chunk in parallel with ThreadPoolExecutor
    - Each worker runs dd | rg --json pipeline
    - Merge results from all workers

    This is MUCH faster than single-threaded rg for very large files (75GB+).

    Args:
        filepath: Path to the file to scan
        detectors: List of anomaly detectors to extract patterns from

    Returns:
        Dict mapping detector_name -> list of PrescanMatch objects
    """
    # Collect all patterns from regex-based detectors
    pattern_to_detector: dict[str, tuple[str, float]] = {}

    for detector in detectors:
        if isinstance(detector, TracebackDetector):
            for lang_patterns in detector.TRACEBACK_START_PATTERNS.values():
                for pattern in lang_patterns:
                    pattern_to_detector[pattern.pattern] = (detector.name, 0.9)
        elif isinstance(detector, ErrorKeywordDetector):
            for pattern, severity in detector.ERROR_KEYWORDS:
                pattern_to_detector[pattern.pattern] = (detector.name, severity)
        elif isinstance(detector, WarningKeywordDetector):
            for pattern, severity in detector.WARNING_KEYWORDS:
                pattern_to_detector[pattern.pattern] = (detector.name, severity)
        elif isinstance(detector, HighEntropyDetector):
            for pattern in detector.SECRET_CONTEXT_PATTERNS:
                pattern_to_detector[pattern.pattern] = (detector.name, 0.6)
        elif isinstance(detector, JsonDumpDetector):
            for pattern in detector.JSON_START_PATTERNS:
                pattern_to_detector[pattern.pattern] = (detector.name, 0.3)

    if not pattern_to_detector:
        return {}

    patterns = list(pattern_to_detector.keys())

    try:
        start_time = time()

        # Create file tasks for parallel processing
        tasks = create_file_tasks(filepath)
        logger.info(f'[PRESCAN] Created {len(tasks)} parallel tasks for {filepath}')

        # Process chunks in parallel
        all_matches: list[PrescanMatch] = []
        total_worker_time = 0.0

        with ThreadPoolExecutor(max_workers=MAX_SUBPROCESSES, thread_name_prefix='Prescan') as executor:
            future_to_task = {
                executor.submit(_prescan_chunk_worker, task, patterns, pattern_to_detector): task for task in tasks
            }

            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    _, matches, elapsed = future.result()
                    all_matches.extend(matches)
                    total_worker_time += elapsed
                except Exception as e:
                    logger.error(f'[PRESCAN] Task {task.task_id} failed: {e}')

        # Sort by byte offset
        all_matches.sort(key=lambda m: m.byte_offset)

        # Group by detector
        matches_by_detector: dict[str, list[PrescanMatch]] = {}
        for match in all_matches:
            if match.detector_name not in matches_by_detector:
                matches_by_detector[match.detector_name] = []
            matches_by_detector[match.detector_name].append(match)

        elapsed = time() - start_time
        total_matches = len(all_matches)
        logger.info(
            f'[PRESCAN] Completed in {elapsed:.1f}s (worker time: {total_worker_time:.1f}s): '
            f'{total_matches} matches across {len(matches_by_detector)} detectors'
        )

        return matches_by_detector

    except FileNotFoundError:
        logger.warning('ripgrep (rg) not found')
        return {}
    except Exception as e:
        logger.warning(f'rg prescan failed: {e}')
        return {}


def get_sample_size_lines() -> int:
    """Get the sample size for line length statistics from environment variable.

    Returns:
        Sample size in number of lines. Default is 1,000,000.
        Files with fewer non-empty lines than this will have exact statistics.
    """
    try:
        return int(os.environ.get('RX_SAMPLE_SIZE_LINES', '1000000'))
    except (ValueError, TypeError):
        logger.warning('Invalid RX_SAMPLE_SIZE_LINES value, using default 1000000')
        return 1000000


def human_readable_size(size_bytes: int) -> str:
    """Convert bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f'{size_bytes:.2f} {unit}'
        size_bytes /= 1024
    return f'{size_bytes:.2f} PB'


@dataclass
class FileAnalysisState:
    """Internal state for analyzing a single file.

    This is used during the analysis process and contains internal identifiers.
    For API responses, use the Pydantic FileAnalysisResult model from rx.models.
    """

    file_id: str
    filepath: str
    size_bytes: int
    size_human: str
    is_text: bool

    # Metadata
    created_at: str | None = None
    modified_at: str | None = None
    permissions: str | None = None
    owner: str | None = None

    # Text file metrics (only if is_text=True)
    line_count: int | None = None
    empty_line_count: int | None = None
    line_length_max: int | None = None
    line_length_avg: float | None = None
    line_length_median: float | None = None
    line_length_p95: float | None = None
    line_length_p99: float | None = None
    line_length_stddev: float | None = None

    # Longest line info
    line_length_max_line_number: int | None = None
    line_length_max_byte_offset: int | None = None

    # Line ending info
    line_ending: str | None = None  # "LF", "CRLF", "CR", or "mixed"

    # Compression information
    is_compressed: bool = False
    compression_format: str | None = None
    is_seekable_zstd: bool = False
    compressed_size: int | None = None
    decompressed_size: int | None = None
    compression_ratio: float | None = None

    # Index information
    has_index: bool = False
    index_path: str | None = None
    index_valid: bool = False
    index_checkpoint_count: int | None = None

    # Anomaly detection results (only populated when detect_anomalies=True)
    anomalies: list[AnomalyRange] = field(default_factory=list)
    anomaly_summary: dict[str, int] = field(default_factory=dict)  # category -> count

    # Prefix pattern detection (only populated when detect_anomalies=True)
    prefix_pattern: str | None = None  # Masked tokens: "<DATE> <TIME> <COMPONENT>"
    prefix_regex: str | None = None  # Regex to match the pattern
    prefix_coverage: float | None = None  # Fraction of lines matching (0.0-1.0)
    prefix_length: int | None = None  # Typical prefix length in characters

    # Additional metrics can be added by plugins
    custom_metrics: dict[str, Any] = field(default_factory=dict)


class FileAnalyzer:
    """
    Pluggable file analysis system.

    Supports custom analysis via overridable hook methods:
    - file_hook(): Processes file metadata after basic info is gathered
    - line_hook(): Processes each line during iteration
    - post_hook(): Runs after file processing is complete

    For large files (>= RX_LARGE_TEXT_FILE_MB), analysis results are cached
    in index files for faster subsequent access.

    To add custom analysis, subclass FileAnalyzer and override the hook methods.
    """

    # Default window size for anomaly detection context
    ANOMALY_WINDOW_SIZE = 10

    def __init__(
        self,
        use_index_cache: bool = True,
        detect_anomalies: bool = False,
        anomaly_detectors: list[AnomalyDetector] | None = None,
    ):
        """Initialize the analyzer.

        Args:
            use_index_cache: If True, use cached analysis from index files
                           when available. Default: True
            detect_anomalies: If True, run anomaly detection during analysis.
                            Default: False
            anomaly_detectors: Custom list of anomaly detectors. If None and
                             detect_anomalies is True, uses default detectors.
        """
        self.use_index_cache = use_index_cache
        self.detect_anomalies = detect_anomalies
        self._custom_detectors = anomaly_detectors  # Store custom detectors if provided
        self.anomaly_detectors: list[AnomalyDetector] = []  # Will be set per-file

    def _create_detectors_for_file(self, filepath: str) -> list[AnomalyDetector]:
        """Create anomaly detectors for a specific file.

        Args:
            filepath: Path to the file being analyzed.

        Returns:
            List of detector instances configured for this file.
        """
        if self._custom_detectors is not None:
            logger.debug(f'[DETECTOR] Using {len(self._custom_detectors)} custom detectors')
            return self._custom_detectors  # Use custom detectors as-is

        # Import from rx.analyze.detectors to get the versions with logging
        from rx.analyze.detectors import (
            ErrorKeywordDetector as EKD,
        )
        from rx.analyze.detectors import (
            FormatDeviationDetector as FDD,
        )
        from rx.analyze.detectors import (
            HighEntropyDetector as HED,
        )
        from rx.analyze.detectors import (
            IndentationBlockDetector as IBD,
        )
        from rx.analyze.detectors import (
            JsonDumpDetector as JDD,
        )
        from rx.analyze.detectors import (
            LineLengthSpikeDetector as LLSD,
        )
        from rx.analyze.detectors import (
            TimestampGapDetector as TGD,
        )
        from rx.analyze.detectors import (
            TracebackDetector as TBD,
        )
        from rx.analyze.detectors import (
            WarningKeywordDetector as WKD,
        )

        detectors = [
            TBD(filepath=filepath),
            EKD(filepath=filepath),
            WKD(filepath=filepath),
            LLSD(filepath=filepath),
            IBD(filepath=filepath),
            TGD(filepath=filepath),
            HED(filepath=filepath),
            JDD(filepath=filepath),
            FDD(filepath=filepath),
        ]
        logger.debug(f'[DETECTOR] Created {len(detectors)} default detectors: {[d.name for d in detectors]}')
        return detectors

    def file_hook(self, filepath: str, result: FileAnalysisState) -> None:
        """
        Hook that processes file metadata.

        Override this method to add custom file-level analysis.

        Args:
            filepath: Path to the file being analyzed
            result: FileAnalysisState object to update with custom metrics
        """
        pass

    def line_hook(self, line: str, line_num: int, result: FileAnalysisState) -> None:
        """
        Hook that processes each line.

        Override this method to add custom line-level analysis.

        Args:
            line: The current line content
            line_num: The line number (1-indexed)
            result: FileAnalysisState object to update with custom metrics
        """
        pass

    def post_hook(self, result: FileAnalysisState) -> None:
        """
        Hook that runs after file processing.

        Override this method to add custom post-processing analysis.

        Args:
            result: FileAnalysisState object to update with custom metrics
        """
        pass

    def _dict_to_state(self, data: dict, file_id: str, filepath: str) -> FileAnalysisState:
        """Convert cached dict to FileAnalysisState."""
        return FileAnalysisState(
            file_id=file_id,
            filepath=filepath,
            size_bytes=data.get('size_bytes', 0),
            size_human=data.get('size_human', '0 B'),
            is_text=data.get('is_text', False),
            created_at=data.get('created_at'),
            modified_at=data.get('modified_at'),
            permissions=data.get('permissions'),
            owner=data.get('owner'),
            line_count=data.get('line_count'),
            empty_line_count=data.get('empty_line_count'),
            line_length_max=data.get('line_length_max'),
            line_length_avg=data.get('line_length_avg'),
            line_length_median=data.get('line_length_median'),
            line_length_p95=data.get('line_length_p95'),
            line_length_p99=data.get('line_length_p99'),
            line_length_stddev=data.get('line_length_stddev'),
            line_length_max_line_number=data.get('line_length_max_line_number'),
            line_length_max_byte_offset=data.get('line_length_max_byte_offset'),
            line_ending=data.get('line_ending'),
            custom_metrics=data.get('custom_metrics', {}),
            # Compression fields
            is_compressed=data.get('is_compressed', False),
            compression_format=data.get('compression_format'),
            is_seekable_zstd=data.get('is_seekable_zstd', False),
            compressed_size=data.get('compressed_size'),
            decompressed_size=data.get('decompressed_size'),
            compression_ratio=data.get('compression_ratio'),
            # Index fields
            has_index=data.get('has_index', False),
            index_path=data.get('index_path'),
            index_valid=data.get('index_valid', False),
            index_checkpoint_count=data.get('index_checkpoint_count'),
            # Anomaly fields
            anomalies=self._deserialize_anomalies(data.get('anomalies', [])),
            anomaly_summary=data.get('anomaly_summary', {}),
        )

    def _serialize_anomalies(self, anomalies: list[AnomalyRange]) -> list[dict]:
        """Convert AnomalyRange objects to dicts for serialization."""
        return [
            {
                'start_line': a.start_line,
                'end_line': a.end_line,
                'start_offset': a.start_offset,
                'end_offset': a.end_offset,
                'severity': a.severity,
                'category': a.category,
                'description': a.description,
                'detector': a.detector,
            }
            for a in anomalies
        ]

    def _deserialize_anomalies(self, data: list[dict]) -> list[AnomalyRange]:
        """Convert dicts to AnomalyRange objects."""
        return [
            AnomalyRange(
                start_line=d['start_line'],
                end_line=d['end_line'],
                start_offset=d['start_offset'],
                end_offset=d['end_offset'],
                severity=d['severity'],
                category=d['category'],
                description=d['description'],
                detector=d['detector'],
            )
            for d in data
        ]

    def _state_to_dict(self, result: FileAnalysisState) -> dict:
        """Convert FileAnalysisState to dict for caching."""
        return {
            'file': result.file_id,
            'size_bytes': result.size_bytes,
            'size_human': result.size_human,
            'is_text': result.is_text,
            'created_at': result.created_at,
            'modified_at': result.modified_at,
            'permissions': result.permissions,
            'owner': result.owner,
            'line_count': result.line_count,
            'empty_line_count': result.empty_line_count,
            'line_length_max': result.line_length_max,
            'line_length_avg': result.line_length_avg,
            'line_length_median': result.line_length_median,
            'line_length_p95': result.line_length_p95,
            'line_length_p99': result.line_length_p99,
            'line_length_stddev': result.line_length_stddev,
            'line_length_max_line_number': result.line_length_max_line_number,
            'line_length_max_byte_offset': result.line_length_max_byte_offset,
            'line_ending': result.line_ending,
            'custom_metrics': result.custom_metrics,
            # Compression fields
            'is_compressed': result.is_compressed,
            'compression_format': result.compression_format,
            'is_seekable_zstd': result.is_seekable_zstd,
            'compressed_size': result.compressed_size,
            'decompressed_size': result.decompressed_size,
            'compression_ratio': result.compression_ratio,
            # Index fields
            'has_index': result.has_index,
            'index_path': result.index_path,
            'index_valid': result.index_valid,
            'index_checkpoint_count': result.index_checkpoint_count,
            # Anomaly fields
            'anomalies': self._serialize_anomalies(result.anomalies),
            'anomaly_summary': result.anomaly_summary,
        }

    def _index_to_state(self, idx: 'UnifiedFileIndex', file_id: str, filepath: str) -> FileAnalysisState:
        """Convert UnifiedFileIndex to FileAnalysisState."""

        return FileAnalysisState(
            file_id=file_id,
            filepath=filepath,
            size_bytes=idx.source_size_bytes,
            size_human=human_readable_size(idx.source_size_bytes),
            is_text=idx.is_text,
            modified_at=idx.source_modified_at,
            permissions=idx.permissions,
            owner=idx.owner,
            line_count=idx.line_count,
            empty_line_count=idx.empty_line_count,
            line_length_max=idx.line_length_max,
            line_length_avg=idx.line_length_avg,
            line_length_median=idx.line_length_median,
            line_length_p95=idx.line_length_p95,
            line_length_p99=idx.line_length_p99,
            line_length_stddev=idx.line_length_stddev,
            line_length_max_line_number=idx.line_length_max_line_number,
            line_length_max_byte_offset=idx.line_length_max_byte_offset,
            line_ending=idx.line_ending,
            # Compression fields
            is_compressed=idx.compression_format is not None,
            compression_format=idx.compression_format,
            decompressed_size=idx.decompressed_size_bytes,
            compression_ratio=idx.compression_ratio,
            # Index fields
            has_index=True,
            index_valid=True,
            index_checkpoint_count=len(idx.line_index) if idx.line_index else 0,
            # Anomaly fields
            anomalies=[
                AnomalyRange(
                    start_line=a.start_line,
                    end_line=a.end_line,
                    start_offset=a.start_offset,
                    end_offset=a.end_offset,
                    severity=a.severity,
                    category=a.category,
                    description=a.description,
                    detector=a.detector,
                )
                for a in (idx.anomalies or [])
            ],
            anomaly_summary=idx.anomaly_summary or {},
        )

    def _state_to_index(self, result: FileAnalysisState, filepath: str) -> 'UnifiedFileIndex':
        """Convert FileAnalysisState to UnifiedFileIndex for caching."""
        from datetime import datetime

        from rx.models import AnomalyRangeResult, FileType, UnifiedFileIndex
        from rx.unified_index import UNIFIED_INDEX_VERSION

        # Determine file type
        if result.is_compressed:
            file_type = FileType.COMPRESSED
        elif result.is_text:
            file_type = FileType.TEXT
        else:
            file_type = FileType.BINARY

        return UnifiedFileIndex(
            version=UNIFIED_INDEX_VERSION,
            source_path=filepath,
            source_modified_at=result.modified_at or datetime.now().isoformat(),
            source_size_bytes=result.size_bytes,
            created_at=datetime.now().isoformat(),
            build_time_seconds=0.0,
            file_type=file_type,
            compression_format=result.compression_format,
            is_text=result.is_text,
            permissions=result.permissions,
            owner=result.owner,
            line_index=[],  # Not available from FileAnalysisState
            line_count=result.line_count,
            empty_line_count=result.empty_line_count,
            line_length_max=result.line_length_max,
            line_length_avg=result.line_length_avg,
            line_length_median=result.line_length_median,
            line_length_p95=result.line_length_p95,
            line_length_p99=result.line_length_p99,
            line_length_stddev=result.line_length_stddev,
            line_length_max_line_number=result.line_length_max_line_number,
            line_length_max_byte_offset=result.line_length_max_byte_offset,
            line_ending=result.line_ending,
            decompressed_size_bytes=result.decompressed_size,
            compression_ratio=result.compression_ratio,
            analysis_performed=True,
            anomalies=[
                AnomalyRangeResult(
                    start_line=a.start_line,
                    end_line=a.end_line,
                    start_offset=a.start_offset,
                    end_offset=a.end_offset,
                    severity=a.severity,
                    category=a.category,
                    description=a.description,
                    detector=a.detector,
                )
                for a in (result.anomalies or [])
            ],
            anomaly_summary=result.anomaly_summary or {},
            # Prefix pattern detection
            prefix_pattern=result.prefix_pattern,
            prefix_regex=result.prefix_regex,
            prefix_coverage=result.prefix_coverage,
            prefix_length=result.prefix_length,
        )

    def _add_index_info(self, filepath: str, result: FileAnalysisState):
        """Add index information to analysis result."""
        from rx.unified_index import get_index_path as get_unified_index_path
        from rx.unified_index import load_index as load_unified_index

        try:
            # Check for seekable zstd index first (for .zst files)
            if seekable_zstd.is_seekable_zstd(filepath):
                index_path = seekable_index.get_index_path(filepath)
                if os.path.exists(str(index_path)):
                    result.has_index = True
                    result.index_path = str(index_path)
                    result.index_valid = seekable_index.is_index_valid(filepath)

                    if result.index_valid:
                        try:
                            index_data = seekable_index.load_index(index_path)
                            if index_data:
                                result.index_checkpoint_count = len(index_data.frames)
                        except Exception as e:
                            logger.warning(f'Failed to load seekable index: {e}')
                    return

            # Check for unified index
            index_path = get_unified_index_path(filepath)
            if os.path.exists(str(index_path)):
                result.has_index = True
                result.index_path = str(index_path)

                try:
                    unified_idx = load_unified_index(filepath)
                    if unified_idx:
                        result.index_valid = True
                        result.index_checkpoint_count = len(unified_idx.line_index)
                    else:
                        result.index_valid = False
                except Exception as e:
                    logger.warning(f'Failed to load unified index: {e}')
                    result.index_valid = False
        except Exception as e:
            logger.warning(f'Failed to add index info: {e}')

    def _analyze_compressed_file(self, filepath: str, result: FileAnalysisState):
        """Analyze compressed file by decompressing to /tmp and analyzing."""
        temp_file = None
        try:
            # Create temp file
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tf:
                temp_file = tf.name

            logger.info(f'Decompressing {filepath} to {temp_file}')

            # Decompress to temp file
            try:
                decompress_to_file(filepath, temp_file)
            except OSError as e:
                if 'No space left' in str(e) or 'Disk quota exceeded' in str(e):
                    logger.warning(f'No space left on device, skipping decompression of {filepath}')
                    return
                raise

            # Get decompressed size
            stat = os.stat(temp_file)
            result.decompressed_size = stat.st_size

            # Calculate compression ratio
            if result.compressed_size and result.decompressed_size:
                result.compression_ratio = result.decompressed_size / result.compressed_size

            # Check if decompressed file is text
            if is_text_file(temp_file):
                result.is_text = True
                # Analyze the decompressed content
                self._analyze_text_file(temp_file, result)
            else:
                logger.info(f'Decompressed file is not text: {filepath}')

        except Exception as e:
            logger.error(f'Failed to analyze compressed file {filepath}: {e}')
        finally:
            # IMPORTANT: Clean up temp file
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    logger.debug(f'Cleaned up temp file: {temp_file}')
                except OSError as e:
                    logger.warning(f'Failed to remove temp file {temp_file}: {e}')

    def _try_load_from_cache(self, filepath: str, file_id: str) -> FileAnalysisState | None:
        """Try to load analysis results from cached unified index.

        Args:
            filepath: Path to the file
            file_id: File ID for the result

        Returns:
            FileAnalysisState if valid cache exists with analysis, None otherwise
        """
        if not self.use_index_cache:
            return None

        from rx.unified_index import load_index

        unified_idx = load_index(filepath)
        if unified_idx is None:
            return None

        # Only use cache if analysis was performed
        if not unified_idx.analysis_performed:
            return None

        logger.debug(f'Using cached analysis for {filepath}')

        try:
            stat_info = os.stat(filepath)

            result = FileAnalysisState(
                file_id=file_id,
                filepath=filepath,
                size_bytes=unified_idx.source_size_bytes or stat_info.st_size,
                size_human=human_readable_size(unified_idx.source_size_bytes or stat_info.st_size),
                is_text=True,  # Index only exists for text files
            )

            # File metadata from stat
            result.created_at = datetime.fromtimestamp(stat_info.st_ctime).isoformat()
            result.modified_at = unified_idx.source_modified_at
            result.permissions = unified_idx.permissions or oct(stat_info.st_mode)[-3:]
            result.owner = unified_idx.owner

            if result.owner is None:
                try:
                    import pwd

                    result.owner = pwd.getpwuid(stat_info.st_uid).pw_name
                except (ImportError, KeyError):
                    result.owner = str(stat_info.st_uid)

            # Analysis data from unified index
            result.line_count = unified_idx.line_count
            result.empty_line_count = unified_idx.empty_line_count
            result.line_length_max = unified_idx.line_length_max
            result.line_length_avg = unified_idx.line_length_avg
            result.line_length_median = unified_idx.line_length_median
            result.line_length_p95 = unified_idx.line_length_p95
            result.line_length_p99 = unified_idx.line_length_p99
            result.line_length_stddev = unified_idx.line_length_stddev
            result.line_length_max_line_number = unified_idx.line_length_max_line_number
            result.line_length_max_byte_offset = unified_idx.line_length_max_byte_offset
            result.line_ending = unified_idx.line_ending

            # Anomaly data from unified index
            if unified_idx.anomalies:
                result.anomalies = unified_idx.anomalies
                result.anomaly_summary = unified_idx.anomaly_summary

            return result

        except Exception as e:
            logger.debug(f'Failed to load from cache for {filepath}: {e}')
            return None

    def analyze_file(self, filepath: str, file_id: str) -> FileAnalysisState:
        """Analyze a single file with all registered hooks.

        For large text files with valid cached indexes, analysis data is
        loaded from cache for better performance.
        """
        # STEP 1: Try unified index cache first (only if use_index_cache is True)
        from rx.unified_index import load_index

        cached_index = load_index(filepath) if self.use_index_cache else None
        if cached_index:
            # Check if anomalies were requested but cache has none
            cache_has_anomalies = bool(cached_index.anomalies)
            if self.detect_anomalies and not cache_has_anomalies:
                logger.info(f'Cache exists but has no anomalies, re-analyzing: {filepath}')
            else:
                logger.info(f'Loaded from unified index cache: {filepath}')
                # Convert UnifiedFileIndex to FileAnalysisState
                result = self._index_to_state(cached_index, file_id, filepath)
                # Still run hooks
                try:
                    self.file_hook(filepath, result)
                except Exception as e:
                    logger.warning(f'File hook failed: {e}')
                try:
                    self.post_hook(result)
                except Exception as e:
                    logger.warning(f'Post hook failed: {e}')
                return result

        # STEP 2: Try old index cache (only if use_index_cache is True)
        cached_result = self._try_load_from_cache(filepath, file_id) if self.use_index_cache else None
        if cached_result is not None:
            # Check if anomalies were requested but cache has none
            if self.detect_anomalies and not cached_result.anomalies:
                logger.info(f'Index cache exists but has no anomalies, re-analyzing: {filepath}')
            else:
                # Still run hooks on cached result
                try:
                    self.file_hook(filepath, cached_result)
                except Exception as e:
                    logger.warning(f'File hook failed for {filepath}: {e}')
                try:
                    self.post_hook(cached_result)
                except Exception as e:
                    logger.warning(f'Post hook failed for {filepath}: {e}')

                return cached_result

        # STEP 3: Fresh analysis
        try:
            stat_info = os.stat(filepath)
            size_bytes = stat_info.st_size

            # Initialize result
            result = FileAnalysisState(
                file_id=file_id,
                filepath=filepath,
                size_bytes=size_bytes,
                size_human=human_readable_size(size_bytes),
                is_text=is_text_file(filepath),
            )

            # File metadata
            result.created_at = datetime.fromtimestamp(stat_info.st_ctime).isoformat()
            result.modified_at = datetime.fromtimestamp(stat_info.st_mtime).isoformat()
            result.permissions = oct(stat_info.st_mode)[-3:]

            try:
                import pwd

                result.owner = pwd.getpwuid(stat_info.st_uid).pw_name
            except (ImportError, KeyError):
                result.owner = str(stat_info.st_uid)

            # STEP 4: Detect compression (NEW)
            if is_compressed(filepath):
                result.is_compressed = True
                comp_format = detect_compression(filepath)
                result.compression_format = comp_format.value if comp_format else None
                result.compressed_size = size_bytes

                if seekable_zstd.is_seekable_zstd(filepath):
                    result.is_seekable_zstd = True
                    # Get decompressed size from seekable index if available
                    try:
                        index_data = seekable_index.load_index(filepath)
                        if index_data:
                            result.decompressed_size = index_data.get('decompressed_size_bytes')
                            if result.decompressed_size and result.compressed_size:
                                result.compression_ratio = result.decompressed_size / result.compressed_size
                    except Exception as e:
                        logger.debug(f'Could not get seekable zstd info: {e}')

            # STEP 5: Create detectors for this file (with filepath for logging)
            if self.detect_anomalies:
                self.anomaly_detectors = self._create_detectors_for_file(filepath)

            # STEP 6: Run file-level hooks
            try:
                self.file_hook(filepath, result)
            except Exception as e:
                logger.warning(f'File hook failed for {filepath}: {e}')

            # STEP 7: Analyze content
            if result.is_text:
                self._analyze_text_file(filepath, result)
            elif result.is_compressed:
                # NEW: Handle compressed files
                self._analyze_compressed_file(filepath, result)

            # STEP 8: Add index information (NEW)
            self._add_index_info(filepath, result)

            # STEP 9: Run post-processing hooks
            try:
                self.post_hook(result)
            except Exception as e:
                logger.warning(f'Post hook failed for {filepath}: {e}')

            # STEP 10: Save to unified index cache
            try:
                unified_index = self._state_to_index(result, filepath)
                from rx.unified_index import save_index

                save_index(unified_index)
            except Exception as e:
                logger.warning(f'Failed to save cache: {e}')

            return result

        except Exception as e:
            logger.error(f'Failed to analyze {filepath}: {e}')
            # Return minimal result for failed files
            return FileAnalysisState(
                file_id=file_id,
                filepath=filepath,
                size_bytes=0,
                size_human='0 B',
                is_text=False,
            )

    def _analyze_text_file(self, filepath: str, result: FileAnalysisState):
        """Analyze text file content using streaming to avoid loading entire file in memory.

        Memory optimizations for large files:
        - Uses BoundedAnomalyHeap to limit anomalies to 100k entries max
        - Uses SparseLineOffsets to only track offsets for anomaly lines
        - Uses parallel rg prescan for fast keyword detection on large files (>100MB)
        - Periodic pruning based on anomaly density
        """
        try:
            file_size = result.size_bytes

            # Sample first chunk for line ending detection (up to 10MB)
            SAMPLE_SIZE = 10 * 1024 * 1024
            with open(filepath, 'rb') as f:
                sample = f.read(SAMPLE_SIZE)

            # Detect line endings from sample
            result.line_ending = self._detect_line_ending(sample)

            # Now process file line by line (streaming)
            # Use reservoir sampling for percentiles to avoid storing all line lengths
            # IMPORTANT: Use binary mode to count lines consistently with wc -l
            # Text mode treats \r, \n, and \r\n all as line separators, which can
            # double-count lines in files with CRLF that also have bare \r characters
            empty_line_count = 0
            byte_offset = 0
            max_line_length = 0
            max_line_number = 0
            max_line_offset = 0
            last_line_num = 0

            # Streaming statistics
            total_length = 0
            non_empty_count = 0
            sum_of_squares = 0.0

            # Reservoir sampling for percentiles
            # Files with fewer non-empty lines will have exact statistics
            sample_size = get_sample_size_lines()
            line_length_sample = []

            # Anomaly detection state - MEMORY OPTIMIZED
            keyword_prescan_lines: set[int] = set()
            warning_prescan_lines: set[int] = set()
            secret_prescan_lines: set[int] = set()
            anomaly_heap: BoundedAnomalyHeap | None = None
            line_offsets: SparseLineOffsets | None = None
            window: deque[str] | None = None
            line_lengths_window: deque[int] | None = None
            # Detectors that need line-by-line processing (require context)
            context_detectors: list[AnomalyDetector] = []
            # Detector name -> detector for keyword prescan results
            detector_by_name: dict[str, AnomalyDetector] = {}
            # Debug: timing and hit count per detector
            detector_timing: dict[str, float] = {}
            detector_hits: dict[str, int] = {}
            analysis_start_time = time()

            if self.detect_anomalies:
                window = deque(maxlen=self.ANOMALY_WINDOW_SIZE)
                line_lengths_window = deque(maxlen=1000)
                # Use bounded heap instead of unbounded list
                anomaly_heap = BoundedAnomalyHeap(max_size=100_000)
                # Use sparse line offsets instead of storing all offsets
                line_offsets = SparseLineOffsets(window_size=100)

                # Extract prefix pattern using Drain3 (for prefix deviation detection)
                try:
                    from rx.analyze.prefix_pattern import PrefixPatternExtractor

                    prefix_extractor = PrefixPatternExtractor()
                    prefix_result = prefix_extractor.extract_from_file(filepath)
                    if prefix_result and prefix_result.coverage >= 0.90:
                        result.prefix_pattern = prefix_result.pattern
                        result.prefix_regex = prefix_result.regex
                        result.prefix_coverage = prefix_result.coverage
                        result.prefix_length = prefix_result.prefix_length
                        logger.info(
                            f'Prefix pattern detected: {prefix_result.pattern} (coverage: {prefix_result.coverage:.1%})'
                        )

                        # Add PrefixDeviationDetector to detectors list
                        from rx.analyze.detectors.prefix_deviation import PrefixDeviationDetector

                        prefix_detector = PrefixDeviationDetector(
                            prefix_regex=prefix_result.regex,
                            prefix_length=prefix_result.prefix_length,
                        )
                        self.anomaly_detectors.append(prefix_detector)
                except Exception as e:
                    logger.debug(f'Prefix pattern extraction failed: {e}')

                # Separate detectors: keyword-based (can use rg) vs context-based (need line-by-line)
                # Collect all keyword patterns for combined rg prescan
                keyword_patterns: list[tuple[re.Pattern, float]] = []
                warning_patterns: list[tuple[re.Pattern, float]] = []
                secret_patterns: list[tuple[re.Pattern, float]] = []

                for detector in self.anomaly_detectors:
                    detector_by_name[detector.name] = detector
                    if isinstance(detector, ErrorKeywordDetector):
                        keyword_patterns.extend(detector.ERROR_KEYWORDS)
                    elif isinstance(detector, WarningKeywordDetector):
                        warning_patterns.extend(detector.WARNING_KEYWORDS)
                    elif isinstance(detector, HighEntropyDetector):
                        # Add secret context patterns for prescan
                        secret_patterns.extend((p, 0.8) for p in detector.SECRET_CONTEXT_PATTERNS)

                # Use rg prescan for keyword detection on files > 100MB
                if file_size > 100 * 1024 * 1024:  # 100MB threshold
                    # Prescan for error keywords
                    if keyword_patterns:
                        keyword_prescan_lines = rg_prescan_keywords(filepath, keyword_patterns)
                        logger.info(f'rg prescan: {len(keyword_prescan_lines)} error keyword lines')
                    # Prescan for warning keywords
                    if warning_patterns:
                        warning_prescan_lines = rg_prescan_keywords(filepath, warning_patterns)
                        logger.info(f'rg prescan: {len(warning_prescan_lines)} warning keyword lines')
                    # Prescan for secret patterns
                    if secret_patterns:
                        secret_prescan_lines = rg_prescan_keywords(filepath, secret_patterns)
                        logger.info(f'rg prescan: {len(secret_prescan_lines)} potential secret lines')

                # Add all detectors to context_detectors for line-by-line processing
                # (prescan just helps with fast rejection of non-matching lines)
                for detector in self.anomaly_detectors:
                    context_detectors.append(detector)
                    # Initialize timing and hit counters
                    detector_timing[detector.name] = 0.0
                    detector_hits[detector.name] = 0

                logger.debug(
                    f'[ANALYZE] Starting analysis of {filepath} ({file_size:,} bytes) '
                    f'with {len(self.anomaly_detectors)} detectors'
                )

            # Use binary mode to handle all line ending types correctly
            with open(filepath, 'rb') as f:
                for line_num, line_bytes in enumerate(f, 1):
                    last_line_num = line_num

                    # Decode line
                    try:
                        line = line_bytes.decode('utf-8', errors='ignore')
                    except Exception:
                        line = ''

                    # Calculate byte offset
                    line_byte_length = len(line_bytes)

                    # Record line offset in sliding window (memory efficient)
                    if self.detect_anomalies and line_offsets is not None:
                        line_offsets.record(line_num, byte_offset)

                    # Strip line ending for length calculation
                    # Remove \n and \r from the end
                    stripped = line.rstrip('\n\r')

                    if stripped.strip():  # non-empty line
                        line_len = len(stripped)
                        non_empty_count += 1
                        total_length += line_len
                        sum_of_squares += line_len * line_len

                        # Reservoir sampling: keep a random sample of line lengths
                        if len(line_length_sample) < sample_size:
                            line_length_sample.append(line_len)
                        else:
                            # Randomly replace an element with decreasing probability
                            j = random.randint(0, non_empty_count - 1)
                            if j < sample_size:
                                line_length_sample[j] = line_len

                        # Track longest line
                        if line_len > max_line_length:
                            max_line_length = line_len
                            max_line_number = line_num
                            max_line_offset = byte_offset
                    else:
                        empty_line_count += 1

                    # Run anomaly detectors
                    if self.detect_anomalies and anomaly_heap is not None and line_offsets is not None:
                        # Calculate running stats for anomaly context
                        running_avg = total_length / non_empty_count if non_empty_count > 0 else 0.0
                        running_variance = (
                            (sum_of_squares / non_empty_count) - (running_avg * running_avg)
                            if non_empty_count > 1
                            else 0.0
                        )
                        running_stddev = running_variance**0.5 if running_variance > 0 else 0.0

                        ctx = LineContext(
                            line=line,
                            line_number=line_num,
                            byte_offset=byte_offset,
                            window=window,
                            line_lengths=line_lengths_window,
                            avg_line_length=running_avg,
                            stddev_line_length=running_stddev,
                        )

                        # Check if this line was found by keyword prescan (for large files)
                        if keyword_prescan_lines and line_num in keyword_prescan_lines:
                            detector = detector_by_name.get('error_keyword')
                            if detector and isinstance(detector, ErrorKeywordDetector):
                                det_start = time()
                                severity = detector.check_line(ctx)
                                detector_timing[detector.name] += time() - det_start
                                if severity is not None:
                                    detector_hits[detector.name] += 1
                                    if anomaly_heap.push(detector.name, line_num, byte_offset, severity, line):
                                        line_offsets.mark_anomaly(line_num)

                        if warning_prescan_lines and line_num in warning_prescan_lines:
                            detector = detector_by_name.get('warning_keyword')
                            if detector and isinstance(detector, WarningKeywordDetector):
                                det_start = time()
                                severity = detector.check_line(ctx)
                                detector_timing[detector.name] += time() - det_start
                                if severity is not None:
                                    detector_hits[detector.name] += 1
                                    if anomaly_heap.push(detector.name, line_num, byte_offset, severity, line):
                                        line_offsets.mark_anomaly(line_num)

                        if secret_prescan_lines and line_num in secret_prescan_lines:
                            detector = detector_by_name.get('high_entropy')
                            if detector and isinstance(detector, HighEntropyDetector):
                                det_start = time()
                                severity = detector.check_line(ctx)
                                detector_timing[detector.name] += time() - det_start
                                if severity is not None:
                                    detector_hits[detector.name] += 1
                                    if anomaly_heap.push(detector.name, line_num, byte_offset, severity, line):
                                        line_offsets.mark_anomaly(line_num)

                        # Run context-dependent detectors line-by-line
                        # Skip detectors that were already handled by prescan
                        for detector in context_detectors:
                            # Skip if already processed via prescan
                            if keyword_prescan_lines and isinstance(detector, ErrorKeywordDetector):
                                continue
                            if warning_prescan_lines and isinstance(detector, WarningKeywordDetector):
                                continue
                            if secret_prescan_lines and isinstance(detector, HighEntropyDetector):
                                continue
                            det_start = time()
                            severity = detector.check_line(ctx)
                            detector_timing[detector.name] += time() - det_start
                            if severity is not None:
                                detector_hits[detector.name] += 1
                                if anomaly_heap.push(detector.name, line_num, byte_offset, severity, line):
                                    line_offsets.mark_anomaly(line_num)

                        # Update sliding window
                        if window is not None:
                            window.append(line)
                        if line_lengths_window is not None:
                            line_lengths_window.append(len(stripped))

                        # Periodic pruning: every 1M lines, check density
                        if line_num % 1_000_000 == 0:
                            anomaly_heap.prune_if_needed(line_num, density_threshold=0.01)
                            if len(anomaly_heap) > 0:
                                logger.debug(
                                    f'Anomaly detection progress: {line_num:,} lines, {len(anomaly_heap):,} anomalies'
                                )

                    # Run line-level hooks on the fly
                    try:
                        self.line_hook(line, line_num, result)
                    except Exception as e:
                        logger.warning(f'Line hook failed at {filepath}:{line_num}: {e}')

                    byte_offset += line_byte_length

            # Set basic line metrics
            result.line_count = last_line_num
            result.empty_line_count = empty_line_count

            # Calculate statistics from streaming data and sample
            if non_empty_count > 0:
                result.line_length_max = max_line_length
                result.line_length_avg = total_length / non_empty_count
                result.line_length_max_line_number = max_line_number
                result.line_length_max_byte_offset = max_line_offset

                # Calculate stddev from sum of squares
                mean = result.line_length_avg
                variance = (sum_of_squares / non_empty_count) - (mean * mean)
                result.line_length_stddev = variance**0.5 if variance > 0 else 0.0

                # Use sample for percentiles
                if line_length_sample:
                    result.line_length_median = statistics.median(line_length_sample)
                    result.line_length_p95 = self._percentile(line_length_sample, 95)
                    result.line_length_p99 = self._percentile(line_length_sample, 99)
                else:
                    result.line_length_median = 0.0
                    result.line_length_p95 = 0.0
                    result.line_length_p99 = 0.0
            else:
                result.line_length_max = 0
                result.line_length_avg = 0.0
                result.line_length_median = 0.0
                result.line_length_p95 = 0.0
                result.line_length_p99 = 0.0
                result.line_length_stddev = 0.0

            # Post-process anomalies: merge and filter
            if self.detect_anomalies and anomaly_heap is not None and len(anomaly_heap) > 0:
                # Convert heap to list format expected by _merge_and_filter_anomalies
                raw_anomalies = anomaly_heap.get_all()
                result.anomalies = self._merge_and_filter_anomalies_v2(
                    raw_anomalies, last_line_num, line_offsets, byte_offset, detector_by_name
                )
                # Build anomaly summary
                result.anomaly_summary = {}
                for anomaly in result.anomalies:
                    category = anomaly.category
                    result.anomaly_summary[category] = result.anomaly_summary.get(category, 0) + 1

            # Log debug summary for anomaly detection
            if self.detect_anomalies and detector_timing:
                total_analysis_time = time() - analysis_start_time
                lines_per_sec = last_line_num / total_analysis_time if total_analysis_time > 0 else 0

                logger.debug(
                    f'[ANALYZE] Completed {filepath}: {last_line_num:,} lines in {total_analysis_time:.2f}s '
                    f'({lines_per_sec:,.0f} lines/sec)'
                )
                logger.debug(f'[ANALYZE] Anomaly heap size: {len(anomaly_heap) if anomaly_heap else 0}')
                logger.debug(f'[ANALYZE] Final anomalies after merge: {len(result.anomalies)}')

                # Log per-detector timing and hits
                logger.debug('[ANALYZE] Detector statistics:')
                for det_name in sorted(detector_timing.keys()):
                    det_time = detector_timing[det_name]
                    det_hits = detector_hits.get(det_name, 0)
                    pct_time = (det_time / total_analysis_time * 100) if total_analysis_time > 0 else 0
                    logger.debug(
                        f'[ANALYZE]   {det_name}: {det_time:.3f}s ({pct_time:.1f}%), '
                        f'{det_hits} hits'
                    )

                # Log anomaly summary by category
                if result.anomaly_summary:
                    logger.debug(f'[ANALYZE] Anomaly summary: {result.anomaly_summary}')

        except Exception as e:
            logger.error(f'Failed to analyze text content of {filepath}: {e}')

    def _merge_and_filter_anomalies(
        self,
        raw_anomalies: list[tuple[AnomalyDetector, int, int, float, str]],
        total_lines: int,
        line_offsets: dict[int, int],
        file_size: int,
    ) -> list[AnomalyRange]:
        """Merge adjacent anomalies and filter to maintain density limits.

        Density is kept between 0.01% and 5% of total lines.

        Args:
            raw_anomalies: List of (detector, line_num, byte_offset, severity, line_text)
            total_lines: Total line count in file
            line_offsets: Mapping of line number to byte offset
            file_size: Total file size in bytes

        Returns:
            List of merged and filtered AnomalyRange objects
        """
        if not raw_anomalies:
            return []

        # Get anomaly line limit from env var (default 1000)
        anomaly_line_limit = int(os.environ.get('RX_ANOMALY_LINE_LIMIT', '1000'))

        # Group by detector
        by_detector: dict[str, list[tuple[int, int, float, str]]] = {}
        for detector, line_num, byte_offset, severity, line_text in raw_anomalies:
            name = detector.name
            if name not in by_detector:
                by_detector[name] = []
            by_detector[name].append((line_num, byte_offset, severity, line_text))

        merged: list[AnomalyRange] = []

        # Process each detector's anomalies
        for detector in self.anomaly_detectors:
            name = detector.name
            if name not in by_detector:
                continue

            # Sort by line number
            entries = sorted(by_detector[name], key=lambda x: x[0])

            # Merge adjacent lines, but cap at anomaly_line_limit
            i = 0
            while i < len(entries):
                start_line, start_offset, max_severity, first_line = entries[i]
                end_line = start_line
                lines_in_range = [first_line]

                # Look ahead for adjacent lines to merge
                j = i + 1
                while j < len(entries):
                    next_line = entries[j][0]
                    # Check adjacency AND line limit
                    range_size = next_line - start_line + 1
                    if next_line <= end_line + 2 and range_size <= anomaly_line_limit:
                        end_line = next_line
                        max_severity = max(max_severity, entries[j][2])
                        lines_in_range.append(entries[j][3])
                        j += 1
                    else:
                        break

                # Calculate end offset
                end_offset = line_offsets.get(end_line, start_offset)
                # Approximate end offset at end of line
                if end_line + 1 in line_offsets:
                    end_offset = line_offsets[end_line + 1] - 1
                elif end_line == total_lines:
                    end_offset = file_size

                # Create anomaly range
                merged.append(
                    AnomalyRange(
                        start_line=start_line,
                        end_line=end_line,
                        start_offset=start_offset,
                        end_offset=end_offset,
                        severity=max_severity,
                        category=detector.category,
                        description=detector.get_description(lines_in_range),
                        detector=detector.name,
                    )
                )

                i = j

        # Apply density limits: keep between 0.01% and 5% of lines
        # For small files (< 1000 lines), allow up to 50 anomalies minimum
        # This prevents overly aggressive filtering on small test files
        if total_lines > 0 and len(merged) > 0:
            # Minimum: at least 50 anomalies or 0.01% of lines, whichever is larger
            min_anomalies = max(50, int(total_lines * 0.0001))
            # Maximum: 5% of lines, but at least min_anomalies
            max_anomalies = max(min_anomalies, int(total_lines * 0.05))

            if len(merged) > max_anomalies:
                # Sort by severity (descending) and take top N
                merged.sort(key=lambda a: a.severity, reverse=True)
                merged = merged[:max_anomalies]

        # Sort final result by line number
        merged.sort(key=lambda a: a.start_line)

        return merged

    def _merge_and_filter_anomalies_v2(
        self,
        raw_anomalies: list[tuple[str, int, int, float, str]],
        total_lines: int,
        line_offsets: SparseLineOffsets | None,
        file_size: int,
        detector_by_name: dict[str, AnomalyDetector],
    ) -> list[AnomalyRange]:
        """Merge adjacent anomalies and filter to maintain density limits.

        This is the memory-efficient version that works with BoundedAnomalyHeap output.

        Args:
            raw_anomalies: List of (detector_name, line_num, byte_offset, severity, line_text)
            total_lines: Total line count in file
            line_offsets: SparseLineOffsets for getting byte offsets
            file_size: Total file size in bytes
            detector_by_name: Mapping from detector name to detector instance

        Returns:
            List of merged and filtered AnomalyRange objects
        """
        if not raw_anomalies:
            return []

        # Get anomaly line limit from env var (default 1000)
        anomaly_line_limit = int(os.environ.get('RX_ANOMALY_LINE_LIMIT', '1000'))

        # Group by detector name
        by_detector: dict[str, list[tuple[int, int, float, str]]] = {}
        for detector_name, line_num, byte_offset, severity, line_text in raw_anomalies:
            if detector_name not in by_detector:
                by_detector[detector_name] = []
            by_detector[detector_name].append((line_num, byte_offset, severity, line_text))

        merged: list[AnomalyRange] = []

        # Process each detector's anomalies
        for detector_name, entries in by_detector.items():
            detector = detector_by_name.get(detector_name)
            if detector is None:
                continue

            # Sort by line number
            entries = sorted(entries, key=lambda x: x[0])

            # Merge adjacent lines, but cap at anomaly_line_limit
            i = 0
            while i < len(entries):
                start_line, start_offset, max_severity, first_line = entries[i]
                end_line = start_line
                lines_in_range = [first_line]

                # Look ahead for adjacent lines to merge
                j = i + 1
                while j < len(entries):
                    next_line = entries[j][0]
                    # Check adjacency AND line limit
                    range_size = next_line - start_line + 1
                    if next_line <= end_line + 2 and range_size <= anomaly_line_limit:
                        end_line = next_line
                        max_severity = max(max_severity, entries[j][2])
                        lines_in_range.append(entries[j][3])
                        j += 1
                    else:
                        break

                # Calculate end offset using sparse offsets
                end_offset = start_offset
                if line_offsets is not None:
                    end_offset = line_offsets.get(end_line, start_offset)
                    # Try to get the next line's offset for more accurate end
                    next_line_offset = line_offsets.get(end_line + 1, 0)
                    if next_line_offset > 0:
                        end_offset = next_line_offset - 1
                    elif end_line == total_lines:
                        end_offset = file_size

                # Create anomaly range
                merged.append(
                    AnomalyRange(
                        start_line=start_line,
                        end_line=end_line,
                        start_offset=start_offset,
                        end_offset=end_offset,
                        severity=max_severity,
                        category=detector.category,
                        description=detector.get_description(lines_in_range),
                        detector=detector.name,
                    )
                )

                i = j

        # Apply density limits: keep between 0.01% and 5% of lines
        # For small files (< 1000 lines), allow up to 50 anomalies minimum
        # This prevents overly aggressive filtering on small test files
        if total_lines > 0 and len(merged) > 0:
            # Minimum: at least 50 anomalies or 0.01% of lines, whichever is larger
            min_anomalies = max(50, int(total_lines * 0.0001))
            # Maximum: 5% of lines, but at least min_anomalies
            max_anomalies = max(min_anomalies, int(total_lines * 0.05))

            if len(merged) > max_anomalies:
                # Sort by severity (descending) and take top N
                merged.sort(key=lambda a: a.severity, reverse=True)
                merged = merged[:max_anomalies]

        # Sort final result by line number
        merged.sort(key=lambda a: a.start_line)

        return merged

    @staticmethod
    def _percentile(data: list[int | float], p: float) -> float:
        """Calculate the p-th percentile of data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        n = len(sorted_data)
        k = (n - 1) * p / 100
        f = int(k)
        c = f + 1 if f + 1 < n else f
        return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])

    @staticmethod
    def _detect_line_ending(content: bytes) -> str:
        """Detect the line ending style used in the file."""
        crlf_count = content.count(b'\r\n')
        # Count standalone CR (not followed by LF)
        cr_count = content.count(b'\r') - crlf_count
        # Count standalone LF (not preceded by CR)
        lf_count = content.count(b'\n') - crlf_count

        endings = []
        if crlf_count > 0:
            endings.append(('CRLF', crlf_count))
        if lf_count > 0:
            endings.append(('LF', lf_count))
        if cr_count > 0:
            endings.append(('CR', cr_count))

        if len(endings) == 0:
            return 'LF'  # Default for single-line files
        elif len(endings) == 1:
            return endings[0][0]
        else:
            return 'mixed'


def analyze_path(paths: list[str], max_workers: int = 10, detect_anomalies: bool = False) -> dict[str, Any]:
    """
    Analyze files at given paths.

    For directories, only text files are analyzed (binary files are skipped).

    Args:
        paths: List of file or directory paths
        max_workers: Maximum number of parallel workers
        detect_anomalies: If True, run anomaly detection on text files

    Returns:
        Dictionary with analysis results in ID-based format
    """
    start_time = time()

    # Collect all files to analyze
    files_to_analyze = []
    skipped_binary_files = []
    for path in paths:
        if os.path.isfile(path):
            # Single file - always analyze (even if binary)
            files_to_analyze.append(path)
        elif os.path.isdir(path):
            # Scan directory for text files only
            for root, dirs, files in os.walk(path):
                for file in files:
                    filepath = os.path.join(root, file)
                    if is_text_file(filepath):
                        files_to_analyze.append(filepath)
                    else:
                        skipped_binary_files.append(filepath)
        else:
            logger.warning(f'Path not found: {path}')

    # Create file IDs
    file_ids = {f'f{i + 1}': filepath for i, filepath in enumerate(files_to_analyze)}

    # Analyze files in parallel
    analyzer = FileAnalyzer(detect_anomalies=detect_anomalies)
    results = []
    scanned_files = []
    skipped_files = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(analyzer.analyze_file, filepath, file_id): (file_id, filepath)
            for file_id, filepath in file_ids.items()
        }

        for future in as_completed(future_to_file):
            file_id, filepath = future_to_file[future]
            try:
                result = future.result()
                results.append(result)
                scanned_files.append(filepath)
            except Exception as e:
                logger.error(f'Analysis failed for {filepath}: {e}')
                skipped_files.append(filepath)

    elapsed_time = time() - start_time

    # Build response in ID-based format
    return {
        'path': ', '.join(paths) if len(paths) > 1 else paths[0],
        'time': elapsed_time,
        'files': file_ids,
        'results': [
            {
                'file': r.file_id,
                'size_bytes': r.size_bytes,
                'size_human': r.size_human,
                'is_text': r.is_text,
                'created_at': r.created_at,
                'modified_at': r.modified_at,
                'permissions': r.permissions,
                'owner': r.owner,
                'line_count': r.line_count,
                'empty_line_count': r.empty_line_count,
                'line_length_max': r.line_length_max,
                'line_length_avg': r.line_length_avg,
                'line_length_median': r.line_length_median,
                'line_length_p95': r.line_length_p95,
                'line_length_p99': r.line_length_p99,
                'line_length_stddev': r.line_length_stddev,
                'line_length_max_line_number': r.line_length_max_line_number,
                'line_length_max_byte_offset': r.line_length_max_byte_offset,
                'line_ending': r.line_ending,
                'custom_metrics': r.custom_metrics,
                # Compression fields
                'is_compressed': r.is_compressed,
                'compression_format': r.compression_format,
                'is_seekable_zstd': r.is_seekable_zstd,
                'compressed_size': r.compressed_size,
                'decompressed_size': r.decompressed_size,
                'compression_ratio': r.compression_ratio,
                # Index fields
                'has_index': r.has_index,
                'index_path': r.index_path,
                'index_valid': r.index_valid,
                'index_checkpoint_count': r.index_checkpoint_count,
                # Anomaly fields (only if detect_anomalies was enabled)
                'anomalies': [
                    {
                        'start_line': a.start_line,
                        'end_line': a.end_line,
                        'start_offset': a.start_offset,
                        'end_offset': a.end_offset,
                        'severity': a.severity,
                        'category': a.category,
                        'description': a.description,
                        'detector': a.detector,
                    }
                    for a in r.anomalies
                ]
                if r.anomalies
                else None,
                'anomaly_summary': r.anomaly_summary if r.anomaly_summary else None,
            }
            for r in results
        ],
        'scanned_files': scanned_files,
        'skipped_files': skipped_files + skipped_binary_files,
    }


__all__ = [
    'AnomalyDetector',
    'AnomalyRange',
    'ErrorKeywordDetector',
    'FileAnalysisState',
    'FileAnalyzer',
    'IndentationBlockDetector',
    'LineContext',
    'LineLengthSpikeDetector',
    'TracebackDetector',
    'analyze_path',
]
