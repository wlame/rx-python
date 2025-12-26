"""Error keyword detector."""

import re

from .base import AnomalyDetector, LineContext


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

    def get_prescan_patterns(self) -> list[tuple[re.Pattern, float]]:
        """Return patterns for ripgrep prescan."""
        return self.ERROR_KEYWORDS
