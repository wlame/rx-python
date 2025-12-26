"""JSON dump detector."""

import re

from .base import AnomalyDetector, LineContext


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
