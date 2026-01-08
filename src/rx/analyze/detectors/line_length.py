"""Line length spike detector."""

import logging

from .base import AnomalyDetector, LineContext, register_detector


logger = logging.getLogger(__name__)


@register_detector
class LineLengthSpikeDetector(AnomalyDetector):
    """Detects lines that are significantly longer than average."""

    def __init__(self, filepath: str | None = None):
        """Initialize the detector.

        Args:
            filepath: Path to file being analyzed (for logging context).
        """
        self._filepath = filepath
        self._detection_count = 0
        self._max_detected_length = 0
        logger.debug(f'[line_length_spike] Initialized for file: {filepath}')

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

    @property
    def detector_description(self) -> str:
        return 'Detects lines significantly longer than average (>5 standard deviations)'

    @property
    def severity_min(self) -> float:
        return 0.3

    @property
    def severity_max(self) -> float:
        return 0.7

    @property
    def examples(self) -> list[str]:
        return ['Lines >5 stddev above mean length']

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
            self._detection_count += 1
            if line_len > self._max_detected_length:
                self._max_detected_length = line_len
            return severity

        return None

    def get_description(self, lines: list[str]) -> str:
        if lines:
            length = len(lines[0].rstrip())
            return f'Unusually long line ({length} chars)'
        return 'Unusually long line'
