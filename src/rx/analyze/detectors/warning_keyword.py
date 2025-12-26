"""Warning keyword detector."""

import re

from .base import AnomalyDetector, LineContext


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

    def get_prescan_patterns(self) -> list[tuple[re.Pattern, float]]:
        """Return patterns for ripgrep prescan."""
        return self.WARNING_KEYWORDS
