"""JSON dump detector."""

import re

from .base import AnomalyDetector, LineContext


class JsonDumpDetector(AnomalyDetector):
    """Detects embedded JSON objects in log lines.

    Only triggers for substantial multiline JSON structures (>10 lines).
    Single-line JSON or small structures are not flagged.

    To avoid false positives, we require:
    - JSON-like structure with proper key-value syntax ("key": value)
    - Multiple consecutive lines that look like JSON
    - Minimum line length of 100 characters
    """

    # Patterns for JSON-like content - require proper JSON syntax
    # These patterns look for actual JSON structure, not just brackets
    JSON_START_PATTERNS = [
        # Line starts with { followed by quoted key
        re.compile(r'^\s*\{\s*"[^"]+"\s*:'),
        # JSON object as value: key: {"
        re.compile(r':\s*\{\s*"[^"]+"\s*:'),
        # Line starts with [ followed by { or " (array of objects/strings)
        re.compile(r'^\s*\[\s*[\{"]'),
        # JSON array as value: key: [{ or key: ["
        re.compile(r':\s*\[\s*[\{"]'),
    ]

    # Pattern to verify JSON-like key-value pairs
    JSON_KEY_VALUE_PATTERN = re.compile(r'"[^"]+"\s*:\s*(?:"[^"]*"|[\d.]+|true|false|null|\{|\[)')

    # Minimum length to consider it a meaningful JSON dump
    MIN_JSON_LENGTH = 100

    # Minimum number of JSON key-value pairs to detect
    MIN_KEY_VALUE_PAIRS = 3

    # Minimum number of lines in window to consider multiline JSON
    MIN_MULTILINE_LINES = 10

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
                # Verify it has multiple JSON key-value pairs
                key_value_matches = self.JSON_KEY_VALUE_PATTERN.findall(line)
                if len(key_value_matches) < self.MIN_KEY_VALUE_PAIRS:
                    continue  # Not enough key-value pairs

                # Check if this is part of a multiline JSON structure
                # by looking at the surrounding window
                # Require multiline context - if no window or not enough JSON-like lines, skip
                json_like_lines = self._count_json_like_lines(ctx.window, line) if ctx.window else 1
                if json_like_lines < self.MIN_MULTILINE_LINES:
                    return None  # Not enough lines to be a significant JSON dump

                # Longer JSON dumps get slightly higher severity
                if len(line) > 500:
                    return 0.4
                elif len(line) > 200:
                    return 0.35
                else:
                    return 0.3

        return None

    def _count_json_like_lines(self, window: 'deque[str]', current_line: str) -> int:
        """Count lines that look like actual JSON in the window plus current line.

        A line is considered JSON-like if it contains JSON key-value patterns
        or is a JSON structure line (just braces/brackets).
        """
        count = 0

        for line in window:
            if self._is_json_like_line(line):
                count += 1

        # Check current line too
        if self._is_json_like_line(current_line):
            count += 1

        return count

    def _is_json_like_line(self, line: str) -> bool:
        """Check if a line looks like JSON content."""
        stripped = line.strip()
        if not stripped:
            return False

        # Pure structure lines (just braces/brackets with optional comma)
        if stripped in ('{', '}', '[', ']', '{,', '},', '[,', '],', '{}', '[]'):
            return True

        # Lines with JSON key-value pairs
        if self.JSON_KEY_VALUE_PATTERN.search(stripped):
            return True

        # Lines that are just closing with values: "},", "],"
        if re.match(r'^[\}\]]\s*,?\s*$', stripped):
            return True

        return False

    def should_merge_with_previous(self, ctx: LineContext, prev_severity: float) -> bool:
        """Merge continuation of multi-line JSON."""
        line = ctx.line.rstrip()
        if not line:
            return False

        return self._is_json_like_line(line)

    def get_description(self, lines: list[str]) -> str:
        total_chars = sum(len(line) for line in lines)
        return f'Embedded JSON ({total_chars} chars, {len(lines)} lines)'
