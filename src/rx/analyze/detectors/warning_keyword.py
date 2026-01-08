"""Warning keyword detector."""

import logging
import re

from .base import AnomalyDetector, LineContext, register_detector


logger = logging.getLogger(__name__)


@register_detector
class WarningKeywordDetector(AnomalyDetector):
    """Detects lines containing warning-related keywords."""

    def __init__(self, filepath: str | None = None):
        """Initialize the detector.

        Args:
            filepath: Path to file being analyzed (for logging context).
        """
        self._filepath = filepath
        self._detection_count = 0
        self._keyword_counts: dict[str, int] = {}
        logger.debug(f'[warning_keyword] Initialized for file: {filepath}')

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

    @property
    def detector_description(self) -> str:
        return 'Detects warning-level keywords: WARNING, WARN, Deprecated, Caution'

    @property
    def severity_min(self) -> float:
        return 0.3

    @property
    def severity_max(self) -> float:
        return 0.4

    @property
    def examples(self) -> list[str]:
        return ['WARNING', 'WARN', '[W]', 'Deprecated', 'Caution']

    def check_line(self, ctx: LineContext) -> float | None:
        line = ctx.line

        max_severity = None
        matched_keyword = None
        for pattern, severity in self.WARNING_KEYWORDS:
            match = pattern.search(line)
            if match:
                if max_severity is None or severity > max_severity:
                    max_severity = severity
                    matched_keyword = match.group(0)

        if max_severity is not None and matched_keyword:
            self._detection_count += 1
            self._keyword_counts[matched_keyword] = self._keyword_counts.get(matched_keyword, 0) + 1

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

    def get_prescan_patterns(self) -> list[tuple[re.Pattern, float]]:
        """Return patterns for ripgrep prescan."""
        return self.WARNING_KEYWORDS
