"""Line length spike detector."""

from .base import AnomalyDetector, LineContext


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
