"""Indentation block detector."""

from .base import AnomalyDetector, LineContext


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
