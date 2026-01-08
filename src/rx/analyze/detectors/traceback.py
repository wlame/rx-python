"""Traceback detector for stack traces from multiple languages."""

import logging
import re

from .base import AnomalyDetector, LineContext, register_detector


logger = logging.getLogger(__name__)


@register_detector
class TracebackDetector(AnomalyDetector):
    """Detects stack traces from Python, Java, JavaScript, Go, and Rust."""

    def __init__(self, filepath: str | None = None):
        """Initialize the detector.

        Args:
            filepath: Path to file being analyzed (for logging context).
        """
        self._filepath = filepath
        self._detection_count = 0
        self._merge_count = 0
        logger.debug(f'[traceback] Initialized for file: {filepath}')

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

    @property
    def detector_description(self) -> str:
        return 'Detects stack traces and exception backtraces from Python, Java, JavaScript, Go, and Rust'

    @property
    def severity_min(self) -> float:
        return 0.9

    @property
    def severity_max(self) -> float:
        return 0.9

    @property
    def examples(self) -> list[str]:
        return [
            'Traceback (most recent call last):',
            'Exception in thread',
            'panic:',
            "thread 'main' panicked at",
        ]

    def check_line(self, ctx: LineContext) -> float | None:
        line = ctx.line.rstrip()

        # Check for traceback start patterns
        for lang, lang_patterns in self.TRACEBACK_START_PATTERNS.items():
            for pattern in lang_patterns:
                if pattern.match(line):
                    self._detection_count += 1
                    return 0.9

        return None

    def should_merge_with_previous(self, ctx: LineContext, prev_severity: float) -> bool:
        line = ctx.line.rstrip()

        # Check for continuation patterns
        for pattern in self.TRACEBACK_CONTINUATION_PATTERNS:
            if pattern.match(line):
                self._merge_count += 1
                return True

        # Also merge if previous was a traceback and this looks like an error message
        if prev_severity >= 0.9:
            # Check if this is an exception message (starts with exception name)
            if re.match(r'^\w+(Error|Exception):', line):
                self._merge_count += 1
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

    def get_prescan_patterns(self) -> list[tuple[re.Pattern, float]]:
        """Return patterns for ripgrep prescan."""
        return [
            (re.compile(r'^Traceback \(most recent call last\):'), 0.9),
            (re.compile(r'^Exception in thread'), 0.9),
            (re.compile(r'^panic:'), 0.9),
            (re.compile(r'^thread .* panicked at'), 0.9),
        ]
