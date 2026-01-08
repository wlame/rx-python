"""Indentation block detector."""

import logging

from .base import AnomalyDetector, LineContext, register_detector


logger = logging.getLogger(__name__)


@register_detector
class IndentationBlockDetector(AnomalyDetector):
    """Detects unusual indentation patterns that may indicate embedded data."""

    def __init__(self, filepath: str | None = None):
        """Initialize the detector.

        Args:
            filepath: Path to file being analyzed (for logging context).
        """
        self._filepath = filepath
        self._detection_count = 0
        self._merge_count = 0
        logger.debug(f'[indentation_block] Initialized for file: {filepath}')

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

    @property
    def detector_description(self) -> str:
        return 'Detects blocks of consistently indented lines that may indicate embedded data or configurations'

    @property
    def severity_min(self) -> float:
        return 0.4

    @property
    def severity_max(self) -> float:
        return 0.4

    @property
    def examples(self) -> list[str]:
        return ['3+ consecutive lines with 4+ spaces indentation']

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
                self._detection_count += 1
                return 0.4

        return None

    def should_merge_with_previous(self, ctx: LineContext, prev_severity: float) -> bool:
        line = ctx.line.rstrip()
        if not line.strip():
            return False
        if self._get_indentation(line) >= self.MIN_INDENTATION:
            self._merge_count += 1
            return True
        return False

    def get_description(self, lines: list[str]) -> str:
        return f'Indented block ({len(lines)} lines)'
