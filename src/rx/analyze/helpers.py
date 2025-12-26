"""Helper classes for memory-efficient anomaly collection."""

import heapq
import logging
from collections import deque


logger = logging.getLogger(__name__)


class BoundedAnomalyHeap:
    """A bounded priority queue for anomalies that prunes low-severity entries.

    Uses a min-heap based on severity, so we can efficiently remove the
    lowest-severity anomalies when the heap exceeds capacity.

    This prevents OOM on large files by maintaining at most max_size anomalies
    in memory at any time.
    """

    def __init__(self, max_size: int = 100_000):
        """Initialize the bounded heap.

        Args:
            max_size: Maximum number of anomalies to keep. When exceeded,
                     lowest-severity anomalies are pruned.
        """
        self.max_size = max_size
        # Min-heap: (severity, line_num, detector_name, byte_offset, line_text)
        # We use negative severity for max-heap behavior when we want highest severity
        # But for pruning we want min-heap to remove lowest severity
        self._heap: list[tuple[float, int, str, int, str]] = []
        self._min_severity = 0.0  # Track minimum severity in heap for fast rejection

    def push(self, detector_name: str, line_num: int, byte_offset: int, severity: float, line_text: str) -> bool:
        """Add an anomaly to the heap.

        Args:
            detector_name: Name of the detector that found the anomaly
            line_num: Line number (1-based)
            byte_offset: Byte offset in file
            severity: Severity score (0.0 to 1.0)
            line_text: The line content

        Returns:
            True if the anomaly was added, False if rejected (severity too low)
        """
        # Fast rejection: if heap is full and this severity is too low, skip
        if len(self._heap) >= self.max_size and severity <= self._min_severity:
            return False

        # Add to heap (min-heap by severity)
        entry = (severity, line_num, detector_name, byte_offset, line_text)
        heapq.heappush(self._heap, entry)

        # Prune if over capacity
        if len(self._heap) > self.max_size:
            # Remove lowest severity entry
            heapq.heappop(self._heap)
            # Update min severity threshold
            if self._heap:
                self._min_severity = self._heap[0][0]

        return True

    def prune_if_needed(self, total_lines: int, density_threshold: float = 0.01) -> None:
        """Prune anomalies if they exceed density threshold.

        Args:
            total_lines: Total lines processed so far
            density_threshold: Maximum fraction of lines that should be anomalies
        """
        if total_lines <= 0:
            return

        max_anomalies = max(100, int(total_lines * density_threshold))
        while len(self._heap) > max_anomalies:
            heapq.heappop(self._heap)

        if self._heap:
            self._min_severity = self._heap[0][0]

    def get_all(self) -> list[tuple[str, int, int, float, str]]:
        """Get all anomalies as a list of (detector_name, line_num, byte_offset, severity, line_text).

        Returns:
            List of anomalies sorted by line number
        """
        # Convert from heap format to expected format and sort by line number
        result = [(name, line, offset, sev, text) for sev, line, name, offset, text in self._heap]
        result.sort(key=lambda x: x[1])  # Sort by line number
        return result

    def __len__(self) -> int:
        return len(self._heap)


class SparseLineOffsets:
    """Memory-efficient storage for line offsets.

    Only stores offsets for lines that are actually referenced (anomaly lines).
    Uses a sliding window for recent offsets to support merging adjacent anomalies.
    """

    def __init__(self, window_size: int = 100):
        """Initialize sparse storage.

        Args:
            window_size: Number of recent line offsets to keep in window
        """
        self.window_size = window_size
        # Sparse storage for anomaly line offsets
        self._offsets: dict[int, int] = {}
        # Sliding window of recent (line_num, offset) pairs
        self._recent: deque[tuple[int, int]] = deque(maxlen=window_size)

    def record(self, line_num: int, offset: int) -> None:
        """Record a line offset in the sliding window."""
        self._recent.append((line_num, offset))

    def mark_anomaly(self, line_num: int) -> None:
        """Mark a line as containing an anomaly, preserving its offset."""
        # Check if it's in recent window
        for ln, off in self._recent:
            if ln == line_num:
                self._offsets[line_num] = off
                return
        # If not found, it's a programming error but don't crash
        logger.warning(f'Line {line_num} not found in recent window for offset tracking')

    def get(self, line_num: int, default: int = 0) -> int:
        """Get the byte offset for a line number."""
        # First check anomaly offsets
        if line_num in self._offsets:
            return self._offsets[line_num]
        # Then check recent window
        for ln, off in self._recent:
            if ln == line_num:
                return off
        return default

    def __contains__(self, line_num: int) -> bool:
        if line_num in self._offsets:
            return True
        for ln, _ in self._recent:
            if ln == line_num:
                return True
        return False

    def __len__(self) -> int:
        return len(self._offsets)
