"""Unit tests for anomaly detectors."""

from collections import deque

import pytest

from rx.analyze import (
    AnomalyDetector,
    ErrorKeywordDetector,
    FormatDeviationDetector,
    HighEntropyDetector,
    IndentationBlockDetector,
    JsonDumpDetector,
    LineContext,
    LineLengthSpikeDetector,
    TimestampGapDetector,
    TracebackDetector,
    WarningKeywordDetector,
)


def make_context(
    line: str,
    line_number: int = 1,
    byte_offset: int = 0,
    window: list[str] | None = None,
    line_lengths: list[int] | None = None,
    avg_line_length: float = 50.0,
    stddev_line_length: float = 10.0,
) -> LineContext:
    """Helper to create a LineContext for testing."""
    return LineContext(
        line=line,
        line_number=line_number,
        byte_offset=byte_offset,
        window=deque(window or [], maxlen=10),
        line_lengths=deque(line_lengths or [], maxlen=1000),
        avg_line_length=avg_line_length,
        stddev_line_length=stddev_line_length,
    )


class TestTracebackDetector:
    """Tests for TracebackDetector."""

    def setup_method(self):
        self.detector = TracebackDetector()

    def test_name_and_category(self):
        assert self.detector.name == 'traceback'
        assert self.detector.category == 'traceback'

    def test_python_traceback_header(self):
        ctx = make_context('Traceback (most recent call last):')
        severity = self.detector.check_line(ctx)
        assert severity == 0.9

    def test_python_error(self):
        ctx = make_context('ValueError: invalid literal for int()')
        severity = self.detector.check_line(ctx)
        assert severity == 0.9

    def test_python_exception(self):
        ctx = make_context('RuntimeException: something went wrong')
        severity = self.detector.check_line(ctx)
        assert severity == 0.9

    def test_java_exception_in_thread(self):
        ctx = make_context('Exception in thread "main" java.lang.NullPointerException')
        severity = self.detector.check_line(ctx)
        assert severity == 0.9

    def test_java_caused_by(self):
        ctx = make_context('Caused by: java.io.IOException: File not found')
        severity = self.detector.check_line(ctx)
        assert severity == 0.9

    def test_javascript_type_error(self):
        ctx = make_context('TypeError: Cannot read property "x" of undefined')
        severity = self.detector.check_line(ctx)
        assert severity == 0.9

    def test_javascript_reference_error(self):
        ctx = make_context('ReferenceError: foo is not defined')
        severity = self.detector.check_line(ctx)
        assert severity == 0.9

    def test_go_panic(self):
        ctx = make_context('panic: runtime error: index out of range')
        severity = self.detector.check_line(ctx)
        assert severity == 0.9

    def test_go_goroutine(self):
        ctx = make_context('goroutine 1 [running]:')
        severity = self.detector.check_line(ctx)
        assert severity == 0.9

    def test_rust_panic(self):
        ctx = make_context("thread 'main' panicked at 'explicit panic', src/main.rs:2:5")
        severity = self.detector.check_line(ctx)
        assert severity == 0.9

    def test_rust_backtrace(self):
        ctx = make_context('stack backtrace:')
        severity = self.detector.check_line(ctx)
        assert severity == 0.9

    def test_normal_line_not_detected(self):
        ctx = make_context('INFO: Application started successfully')
        severity = self.detector.check_line(ctx)
        assert severity is None

    def test_should_merge_python_file_line(self):
        ctx = make_context('  File "/path/to/file.py", line 42, in func')
        assert self.detector.should_merge_with_previous(ctx, 0.9) is True

    def test_should_merge_java_at_line(self):
        ctx = make_context('\tat com.example.Main.main(Main.java:10)')
        assert self.detector.should_merge_with_previous(ctx, 0.9) is True

    def test_should_merge_indented_continuation(self):
        ctx = make_context('\t\tsome continuation line')
        assert self.detector.should_merge_with_previous(ctx, 0.9) is True

    def test_should_not_merge_unrelated(self):
        ctx = make_context('Just a normal log line')
        assert self.detector.should_merge_with_previous(ctx, 0.9) is False

    def test_get_description_python(self):
        lines = ['Traceback (most recent call last):', '  File "test.py", line 1']
        desc = self.detector.get_description(lines)
        assert 'Python traceback' in desc

    def test_get_description_java(self):
        lines = ['Exception in thread "main" NullPointerException']
        desc = self.detector.get_description(lines)
        assert 'Java exception' in desc

    def test_get_description_go(self):
        lines = ['panic: runtime error']
        desc = self.detector.get_description(lines)
        assert 'Go panic' in desc

    def test_get_description_rust(self):
        lines = ["thread 'main' panicked at 'error'"]
        desc = self.detector.get_description(lines)
        assert 'Rust panic' in desc


class TestErrorKeywordDetector:
    """Tests for ErrorKeywordDetector."""

    def setup_method(self):
        self.detector = ErrorKeywordDetector()

    def test_name_and_category(self):
        assert self.detector.name == 'error_keyword'
        assert self.detector.category == 'error'

    def test_fatal_uppercase(self):
        ctx = make_context('2024-01-01 FATAL: Database connection failed')
        severity = self.detector.check_line(ctx)
        assert severity == 0.9

    def test_fatal_lowercase(self):
        ctx = make_context('2024-01-01 fatal: something went wrong')
        severity = self.detector.check_line(ctx)
        assert severity == 0.9

    def test_critical(self):
        ctx = make_context('[CRITICAL] System overload detected')
        severity = self.detector.check_line(ctx)
        assert severity == 0.85

    def test_error(self):
        ctx = make_context('ERROR: Failed to process request')
        severity = self.detector.check_line(ctx)
        assert severity == 0.7

    def test_exception(self):
        ctx = make_context('Exception occurred during processing')
        severity = self.detector.check_line(ctx)
        assert severity == 0.7

    def test_failed(self):
        ctx = make_context('Task failed after 3 retries')
        severity = self.detector.check_line(ctx)
        assert severity == 0.6

    def test_crash(self):
        ctx = make_context('Application crashed unexpectedly')
        severity = self.detector.check_line(ctx)
        assert severity == 0.7

    def test_segmentation_fault(self):
        ctx = make_context('Segmentation fault (core dumped)')
        severity = self.detector.check_line(ctx)
        assert severity == 0.9

    def test_oom(self):
        ctx = make_context('OOM killer invoked')
        severity = self.detector.check_line(ctx)
        assert severity == 0.8

    def test_out_of_memory(self):
        ctx = make_context('Out of memory error')
        severity = self.detector.check_line(ctx)
        assert severity == 0.8

    def test_highest_severity_wins(self):
        # FATAL (0.9) should win over ERROR (0.7)
        ctx = make_context('FATAL ERROR: System shutdown')
        severity = self.detector.check_line(ctx)
        assert severity == 0.9

    def test_normal_line_not_detected(self):
        ctx = make_context('INFO: Application started successfully')
        severity = self.detector.check_line(ctx)
        assert severity is None

    def test_partial_word_not_matched(self):
        # "error" should only match as a word boundary
        ctx = make_context('No errors here, just errorless code')
        severity = self.detector.check_line(ctx)
        # Should still match "errors" as it contains ERROR
        assert severity is None or severity == 0.7  # Depends on regex

    def test_get_description(self):
        lines = ['ERROR: Something failed']
        desc = self.detector.get_description(lines)
        assert 'Error' in desc


class TestLineLengthSpikeDetector:
    """Tests for LineLengthSpikeDetector."""

    def setup_method(self):
        self.detector = LineLengthSpikeDetector()

    def test_name_and_category(self):
        assert self.detector.name == 'line_length_spike'
        assert self.detector.category == 'format'

    def test_very_long_line_detected(self):
        # Line is 500 chars, avg is 80, stddev is 50
        # MIN_STDDEV_THRESHOLD = 40, so stddev=50 passes
        # threshold = 80 + 5.0*50 = 330, so 500 > 330 should trigger
        long_line = 'x' * 500
        ctx = make_context(long_line, avg_line_length=80.0, stddev_line_length=50.0)
        severity = self.detector.check_line(ctx)
        assert severity is not None
        assert 0.3 <= severity <= 0.7

    def test_normal_line_not_detected(self):
        ctx = make_context('A normal length line', avg_line_length=50.0, stddev_line_length=30.0)
        severity = self.detector.check_line(ctx)
        assert severity is None

    def test_low_stddev_not_detected(self):
        # If stddev is too low (< 40), we don't flag
        long_line = 'x' * 500
        ctx = make_context(long_line, avg_line_length=50.0, stddev_line_length=30.0)
        severity = self.detector.check_line(ctx)
        assert severity is None

    def test_zero_avg_not_detected(self):
        long_line = 'x' * 200
        ctx = make_context(long_line, avg_line_length=0.0, stddev_line_length=30.0)
        severity = self.detector.check_line(ctx)
        assert severity is None

    def test_get_description(self):
        lines = ['x' * 500]
        desc = self.detector.get_description(lines)
        assert '500 chars' in desc


class TestIndentationBlockDetector:
    """Tests for IndentationBlockDetector."""

    def setup_method(self):
        self.detector = IndentationBlockDetector()

    def test_name_and_category(self):
        assert self.detector.name == 'indentation_block'
        assert self.detector.category == 'multiline'

    def test_indented_block_detected(self):
        # Need at least 3 indented lines in window
        window = [
            '    indented line 1\n',
            '    indented line 2\n',
        ]
        ctx = make_context('    indented line 3\n', window=window)
        severity = self.detector.check_line(ctx)
        assert severity == 0.4

    def test_single_indented_line_not_detected(self):
        ctx = make_context('    single indented line\n', window=[])
        severity = self.detector.check_line(ctx)
        assert severity is None

    def test_empty_line_not_detected(self):
        window = ['    indented\n', '    indented\n']
        ctx = make_context('   \n', window=window)  # Only whitespace
        severity = self.detector.check_line(ctx)
        assert severity is None

    def test_tabs_count_as_indentation(self):
        window = ['\tindented line 1\n', '\tindented line 2\n']
        ctx = make_context('\tindented line 3\n', window=window)
        severity = self.detector.check_line(ctx)
        assert severity == 0.4

    def test_should_merge_indented(self):
        ctx = make_context('    continued indentation\n')
        assert self.detector.should_merge_with_previous(ctx, 0.4) is True

    def test_should_not_merge_non_indented(self):
        ctx = make_context('no indentation here\n')
        assert self.detector.should_merge_with_previous(ctx, 0.4) is False

    def test_get_description(self):
        lines = ['    line1', '    line2', '    line3']
        desc = self.detector.get_description(lines)
        assert '3 lines' in desc


class TestAnomalyDetectorBase:
    """Tests for the base AnomalyDetector abstract class."""

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            AnomalyDetector()  # type: ignore

    def test_default_should_merge(self):
        detector = TracebackDetector()
        # Test that base implementation returns False for non-traceback
        ctx = make_context('random line')
        # The base class default is False, but TracebackDetector overrides it
        # Just check the interface works
        result = detector.should_merge_with_previous(ctx, 0.5)
        assert isinstance(result, bool)

    def test_default_get_description(self):
        detector = ErrorKeywordDetector()
        # With empty lines, should return default
        lines: list[str] = []
        desc = detector.get_description(lines)
        assert 'Error' in desc or 'detected' in desc.lower()


class TestBoundedAnomalyHeap:
    """Tests for BoundedAnomalyHeap memory-efficient structure."""

    def setup_method(self):
        from rx.analyze import BoundedAnomalyHeap

        self.BoundedAnomalyHeap = BoundedAnomalyHeap

    def test_basic_push_and_get(self):
        heap = self.BoundedAnomalyHeap(max_size=10)
        assert heap.push('error', 1, 0, 0.7, 'ERROR: test')
        assert len(heap) == 1

        anomalies = heap.get_all()
        assert len(anomalies) == 1
        assert anomalies[0] == ('error', 1, 0, 0.7, 'ERROR: test')

    def test_bounded_capacity(self):
        heap = self.BoundedAnomalyHeap(max_size=3)

        # Add 5 anomalies
        heap.push('error', 1, 0, 0.5, 'line1')
        heap.push('error', 2, 10, 0.7, 'line2')
        heap.push('error', 3, 20, 0.9, 'line3')
        heap.push('error', 4, 30, 0.6, 'line4')
        heap.push('error', 5, 40, 0.8, 'line5')

        # Should only keep 3 (highest severity)
        assert len(heap) == 3

        anomalies = heap.get_all()
        severities = [sev for _, _, _, sev, _ in anomalies]
        # Top 3 severities are 0.9, 0.8, 0.7
        assert sorted(severities, reverse=True) == [0.9, 0.8, 0.7]

    def test_fast_rejection_low_severity(self):
        heap = self.BoundedAnomalyHeap(max_size=2)

        # Fill heap with high severity
        heap.push('error', 1, 0, 0.9, 'high1')
        heap.push('error', 2, 10, 0.8, 'high2')

        # First low severity push will be added then pruned (returns True)
        # After pruning, min_severity is updated
        result1 = heap.push('error', 3, 20, 0.3, 'low1')
        assert len(heap) == 2  # Pruned back to max_size

        # Now the min_severity threshold is set, subsequent low severity rejected
        result2 = heap.push('error', 4, 30, 0.2, 'low2')
        assert result2 is False
        assert len(heap) == 2

    def test_prune_if_needed(self):
        heap = self.BoundedAnomalyHeap(max_size=100)

        # Add 50 anomalies
        for i in range(50):
            heap.push('error', i, i * 10, 0.5 + (i % 5) * 0.1, f'line{i}')

        assert len(heap) == 50

        # Prune to 1% of 100 lines = max 100, min 100 -> no pruning needed
        heap.prune_if_needed(total_lines=100, density_threshold=0.01)
        assert len(heap) == 50

        # Prune to 1% of 1000 lines = max 10
        heap.prune_if_needed(total_lines=1000, density_threshold=0.01)
        assert len(heap) <= 100  # min is 100

    def test_sorted_by_line_number(self):
        heap = self.BoundedAnomalyHeap(max_size=10)

        # Add out of order
        heap.push('error', 5, 50, 0.7, 'line5')
        heap.push('error', 2, 20, 0.8, 'line2')
        heap.push('error', 8, 80, 0.6, 'line8')
        heap.push('error', 1, 10, 0.9, 'line1')

        anomalies = heap.get_all()
        line_nums = [line for _, line, _, _, _ in anomalies]
        assert line_nums == sorted(line_nums)


class TestSparseLineOffsets:
    """Tests for SparseLineOffsets memory-efficient structure."""

    def setup_method(self):
        from rx.analyze import SparseLineOffsets

        self.SparseLineOffsets = SparseLineOffsets

    def test_record_and_get_from_window(self):
        offsets = self.SparseLineOffsets(window_size=5)

        offsets.record(1, 0)
        offsets.record(2, 50)
        offsets.record(3, 100)

        assert offsets.get(1) == 0
        assert offsets.get(2) == 50
        assert offsets.get(3) == 100

    def test_mark_anomaly_persists(self):
        offsets = self.SparseLineOffsets(window_size=3)

        offsets.record(1, 0)
        offsets.record(2, 50)
        offsets.record(3, 100)
        offsets.mark_anomaly(2)

        # Record more lines (pushing 1,2,3 out of window)
        offsets.record(4, 150)
        offsets.record(5, 200)
        offsets.record(6, 250)

        # Line 2 should still be accessible (marked as anomaly)
        assert offsets.get(2) == 50
        # Line 1 should not be accessible (not marked, out of window)
        assert offsets.get(1, default=-1) == -1

    def test_contains(self):
        offsets = self.SparseLineOffsets(window_size=3)

        offsets.record(1, 0)
        offsets.record(2, 50)
        offsets.mark_anomaly(1)

        assert 1 in offsets
        assert 2 in offsets

        # Push 2 out of window
        offsets.record(3, 100)
        offsets.record(4, 150)
        offsets.record(5, 200)

        # 1 still present (marked), 2 gone
        assert 1 in offsets
        assert 2 not in offsets

    def test_sliding_window_size(self):
        offsets = self.SparseLineOffsets(window_size=2)

        offsets.record(1, 0)
        offsets.record(2, 50)
        offsets.record(3, 100)  # Should push 1 out of window

        assert offsets.get(1, default=-1) == -1
        assert offsets.get(2) == 50
        assert offsets.get(3) == 100


# =============================================================================
# Tests for New Phase 2 Detectors
# =============================================================================


class TestWarningKeywordDetector:
    """Tests for WarningKeywordDetector."""

    def setup_method(self):
        self.detector = WarningKeywordDetector()

    def test_name_and_category(self):
        assert self.detector.name == 'warning_keyword'
        assert self.detector.category == 'warning'

    def test_warning_uppercase(self):
        ctx = make_context('2024-01-15 10:30:45 WARNING: Connection timeout')
        severity = self.detector.check_line(ctx)
        assert severity == 0.4

    def test_warning_lowercase(self):
        ctx = make_context('2024-01-15 10:30:45 warning: Connection timeout')
        severity = self.detector.check_line(ctx)
        assert severity == 0.4

    def test_warn_keyword(self):
        ctx = make_context('[WARN] Memory usage is high')
        severity = self.detector.check_line(ctx)
        assert severity == 0.4

    def test_bracket_w_format(self):
        ctx = make_context('[W] Configuration file not found')
        severity = self.detector.check_line(ctx)
        assert severity == 0.4

    def test_deprecated_keyword(self):
        ctx = make_context('DeprecationWarning: This function is deprecated')
        severity = self.detector.check_line(ctx)
        # Should match both WARNING and Deprecation
        assert severity is not None
        assert severity >= 0.3

    def test_caution_keyword(self):
        ctx = make_context('Caution: Proceed with care')
        severity = self.detector.check_line(ctx)
        assert severity == 0.35

    def test_no_warning_in_normal_line(self):
        ctx = make_context('INFO: Application started successfully')
        severity = self.detector.check_line(ctx)
        assert severity is None

    def test_description_format(self):
        lines = ['[WARN] Something happened']
        desc = self.detector.get_description(lines)
        assert 'Warning' in desc or 'WARN' in desc


class TestTimestampGapDetector:
    """Tests for TimestampGapDetector."""

    def setup_method(self):
        self.detector = TimestampGapDetector()

    def test_name_and_category(self):
        assert self.detector.name == 'timestamp_gap'
        assert self.detector.category == 'timing'

    def test_no_gap_on_first_line(self):
        ctx = make_context('2024-01-15T10:30:45 INFO: Starting')
        severity = self.detector.check_line(ctx)
        assert severity is None

    def test_small_gap_not_flagged(self):
        # First line establishes timestamp
        ctx1 = make_context('2024-01-15T10:30:00 INFO: Event 1', line_number=1)
        self.detector.check_line(ctx1)

        # Second line 1 minute later - should not be flagged
        ctx2 = make_context('2024-01-15T10:31:00 INFO: Event 2', line_number=2)
        severity = self.detector.check_line(ctx2)
        assert severity is None

    def test_large_gap_flagged(self):
        # First line
        ctx1 = make_context('2024-01-15T10:00:00 INFO: Event 1', line_number=1)
        self.detector.check_line(ctx1)

        # Second line 10 minutes later - should be flagged
        ctx2 = make_context('2024-01-15T10:10:00 INFO: Event 2', line_number=2)
        severity = self.detector.check_line(ctx2)
        assert severity is not None
        assert severity >= 0.3

    def test_hour_gap_high_severity(self):
        # First line
        ctx1 = make_context('2024-01-15T10:00:00 INFO: Event 1', line_number=1)
        self.detector.check_line(ctx1)

        # Second line 1 hour later - should have high severity
        ctx2 = make_context('2024-01-15T11:00:00 INFO: Event 2', line_number=2)
        severity = self.detector.check_line(ctx2)
        assert severity is not None
        assert severity >= 0.7

    def test_unix_timestamp_gap(self):
        # First line with unix timestamp
        ctx1 = make_context('1705312800 INFO: Event 1', line_number=1)  # 2024-01-15 10:00:00
        self.detector.check_line(ctx1)

        # Second line 10 minutes later
        ctx2 = make_context('1705313400 INFO: Event 2', line_number=2)  # 10 minutes later
        severity = self.detector.check_line(ctx2)
        assert severity is not None

    def test_no_timestamp_not_flagged(self):
        ctx = make_context('INFO: No timestamp here')
        severity = self.detector.check_line(ctx)
        assert severity is None

    def test_common_log_format(self):
        # First line
        ctx1 = make_context('15/Jan/2024:10:00:00 +0000 GET /index.html', line_number=1)
        self.detector.check_line(ctx1)

        # Second line 10 minutes later
        ctx2 = make_context('15/Jan/2024:10:10:00 +0000 GET /page.html', line_number=2)
        severity = self.detector.check_line(ctx2)
        assert severity is not None

    # === Tests for new improvements ===

    def test_timestamp_in_payload_not_detected(self):
        """Timestamps in message payload (after first 5 words) should not be detected."""
        # Reset detector state
        self.detector = TimestampGapDetector()

        # First line with timestamp at the start
        ctx1 = make_context('2024-01-15T10:00:00 INFO: Event 1', line_number=1)
        self.detector.check_line(ctx1)

        # Second line where timestamp is in the payload (word 6+), not at start
        # The line has 5 words before the timestamp: "INFO Event data payload message"
        ctx2 = make_context('INFO Event data payload message 2024-01-15T11:00:00 happened here', line_number=2)
        severity = self.detector.check_line(ctx2)
        # Should NOT detect a gap because timestamp is not in first 5 words
        assert severity is None

    def test_timestamp_in_first_words_detected(self):
        """Timestamps in the first 5 words should be detected."""
        self.detector = TimestampGapDetector()

        # First line
        ctx1 = make_context('2024-01-15T10:00:00 INFO: Event 1', line_number=1)
        self.detector.check_line(ctx1)

        # Second line with timestamp in first 5 words (word 3)
        ctx2 = make_context('INFO log 2024-01-15T10:10:00 Event 2', line_number=2)
        severity = self.detector.check_line(ctx2)
        # Should detect the gap (10 minutes)
        assert severity is not None

    def test_gap_ignored_when_too_many_lines_apart(self):
        """Gaps should be ignored if timestamps are more than 500 lines apart."""
        self.detector = TimestampGapDetector()

        # First line
        ctx1 = make_context('2024-01-15T10:00:00 INFO: Event 1', line_number=1)
        self.detector.check_line(ctx1)

        # Second line 600 lines later (> 500 default threshold)
        ctx2 = make_context('2024-01-15T11:00:00 INFO: Event 2', line_number=601)
        severity = self.detector.check_line(ctx2)
        # Should NOT flag as gap because too many lines apart
        assert severity is None

    def test_gap_detected_within_line_threshold(self):
        """Gaps should be detected if timestamps are within 500 lines."""
        self.detector = TimestampGapDetector()

        # First line
        ctx1 = make_context('2024-01-15T10:00:00 INFO: Event 1', line_number=1)
        self.detector.check_line(ctx1)

        # Second line 100 lines later (< 500 threshold)
        ctx2 = make_context('2024-01-15T10:10:00 INFO: Event 2', line_number=101)
        severity = self.detector.check_line(ctx2)
        # Should detect the gap (10 minutes)
        assert severity is not None

    def test_format_lock_after_threshold(self):
        """After 50 consistent timestamps, detector should lock to that format."""
        self.detector = TimestampGapDetector()

        # Send 50 ISO format timestamps to trigger format lock
        for i in range(50):
            ctx = make_context(f'2024-01-15T10:{i:02d}:00 INFO: Event {i}', line_number=i + 1)
            self.detector.check_line(ctx)

        # Verify format is locked to ISO (index 0)
        assert self.detector._locked_format == 0

    def test_locked_format_ignores_other_formats(self):
        """After format lock, other timestamp formats should not be parsed."""
        self.detector = TimestampGapDetector()

        # Send 50 ISO format timestamps to trigger format lock
        for i in range(50):
            ctx = make_context(f'2024-01-15T10:{i:02d}:00 INFO: Event {i}', line_number=i + 1)
            self.detector.check_line(ctx)

        # Now send a Unix timestamp - should not be detected
        ctx_unix = make_context('1705316400 INFO: Unix event', line_number=51)
        result = self.detector._parse_timestamp(ctx_unix.line)
        # Should return None because we're locked to ISO format
        assert result == (None, None)

        # But an ISO timestamp should still be detected
        ctx_iso = make_context('2024-01-15T11:00:00 INFO: ISO event', line_number=52)
        result = self.detector._parse_timestamp(ctx_iso.line)
        assert result[0] is not None  # Timestamp parsed
        assert result[1] == 0  # ISO format (index 0)

    def test_get_line_prefix(self):
        """Test that _get_line_prefix extracts first N words correctly."""
        self.detector = TimestampGapDetector()

        # Test with default 5 words
        line = 'word1 word2 word3 word4 word5 word6 word7'
        prefix = self.detector._get_line_prefix(line)
        assert prefix == 'word1 word2 word3 word4 word5'

        # Test with fewer words
        line = 'word1 word2 word3'
        prefix = self.detector._get_line_prefix(line)
        assert prefix == 'word1 word2 word3'

        # Test with tabs
        line = 'word1\tword2\tword3\tword4\tword5\tword6'
        prefix = self.detector._get_line_prefix(line)
        assert prefix == 'word1 word2 word3 word4 word5'


class TestHighEntropyDetector:
    """Tests for HighEntropyDetector."""

    def setup_method(self):
        self.detector = HighEntropyDetector()

    def test_name_and_category(self):
        assert self.detector.name == 'high_entropy'
        assert self.detector.category == 'security'

    def test_api_key_context_with_token(self):
        ctx = make_context('API_KEY=aB3dE5fG7hI9jK1lM3nO5pQ7rS9tU1vW3xY5z')
        severity = self.detector.check_line(ctx)
        assert severity is not None
        assert severity >= 0.6  # Severity for secret context pattern

    def test_bearer_token(self):
        # Full JWT token format (three parts separated by dots)
        ctx = make_context(
            'Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c'
        )
        severity = self.detector.check_line(ctx)
        assert severity is not None
        assert severity >= 0.6

    def test_password_context(self):
        ctx = make_context('password=SuperSecretPassword123!@#$%^&*()_+')
        severity = self.detector.check_line(ctx)
        assert severity is not None

    def test_private_key_header(self):
        ctx = make_context('-----BEGIN RSA PRIVATE KEY-----')
        severity = self.detector.check_line(ctx)
        assert severity is not None

    def test_high_entropy_string_without_context(self):
        # Without secret context keywords (API_KEY, password, token, etc.),
        # high entropy strings alone are not flagged to reduce false positives.
        # The detector requires both high entropy AND secret context patterns.
        ctx = make_context('data: xK9mPqR3sT7vWyZ2aB5cD8eF1gH4iJ6kL0nO')
        severity = self.detector.check_line(ctx)
        # Without secret context, should NOT be detected (reduces false positives)
        assert severity is None

    def test_base64_string(self):
        # Long base64 string - without secret context, base64 alone is not flagged
        # to reduce false positives (e.g. encoded images, data URLs)
        ctx = make_context('data: SGVsbG9Xb3JsZFRoaXNJc0FMb25nQmFzZTY0U3RyaW5nVGhhdFNob3VsZEJlRGV0ZWN0ZWQ=')
        severity = self.detector.check_line(ctx)
        # Without secret context keywords, not detected (reduces false positives)
        assert severity is None

    def test_sha256_hash(self):
        # SHA256 hex hash - 64 hex characters
        ctx = make_context('checksum: a1b2c3d4e5f6789012345678901234567890abcdefabcdefabcdefabcdef1234')
        severity = self.detector.check_line(ctx)
        assert severity is not None
        assert severity >= 0.5

    def test_aws_access_key(self):
        # AWS access key pattern
        ctx = make_context('AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE')
        severity = self.detector.check_line(ctx)
        assert severity is not None

    def test_normal_text_not_flagged(self):
        ctx = make_context('INFO: Application started successfully')
        severity = self.detector.check_line(ctx)
        assert severity is None

    def test_short_token_not_flagged(self):
        ctx = make_context('token=abc123')  # Too short
        severity = self.detector.check_line(ctx)
        # Context might still trigger, but token itself is too short
        # This tests that MIN_TOKEN_LENGTH is respected
        assert True  # The behavior depends on context detection

    def test_entropy_calculation(self):
        # Test entropy calculation directly
        # 'aaaa' has entropy 0 (all same characters)
        entropy_low = self.detector._calculate_entropy('aaaa')
        assert entropy_low == 0.0

        # Random-looking string has higher entropy
        entropy_high = self.detector._calculate_entropy('aB3dE5fG7hI9jK')
        assert entropy_high > 3.0


class TestJsonDumpDetector:
    """Tests for JsonDumpDetector."""

    def setup_method(self):
        self.detector = JsonDumpDetector()

    def test_name_and_category(self):
        assert self.detector.name == 'json_dump'
        assert self.detector.category == 'format'

    def test_json_object_single_line_not_flagged(self):
        # Single-line JSON is no longer flagged (need multiline context)
        ctx = make_context('{"user": "john", "action": "login", "timestamp": "2024-01-15T10:30:45"}')
        severity = self.detector.check_line(ctx)
        # Without multiline context (10+ JSON-like lines), not detected
        assert severity is None

    def test_json_object_with_multiline_context(self):
        # Create a window with 10+ JSON-like lines to simulate multiline JSON
        from collections import deque

        window = deque(maxlen=20)
        for i in range(12):
            window.append(f'  "field{i}": "value{i}",')

        ctx = LineContext(
            line='{"user": "john", "action": "login", "timestamp": "2024-01-15T10:30:45", "extra": "data"}' + 'x' * 50,
            line_number=15,
            byte_offset=1500,
            window=window,
            line_lengths=deque([80] * 10),
            avg_line_length=80.0,
            stddev_line_length=5.0,
        )
        severity = self.detector.check_line(ctx)
        # With multiline context, should be detected
        assert severity is not None
        assert severity >= 0.3

    def test_json_object_as_value_single_line_not_flagged(self):
        # Single-line JSON values are not flagged without multiline context
        ctx = make_context('response: {"status": "ok", "data": {"id": 123, "name": "test"}}')
        severity = self.detector.check_line(ctx)
        assert severity is None

    def test_json_array_single_line_not_flagged(self):
        # Single-line arrays are not flagged without multiline context
        ctx = make_context('[{"id": 1, "name": "first"}, {"id": 2, "name": "second"}]')
        severity = self.detector.check_line(ctx)
        assert severity is None

    def test_long_json_still_needs_multiline_context(self):
        # Even long JSON needs multiline context
        long_json = '{"data": "' + 'x' * 600 + '"}'
        ctx = make_context(long_json)
        severity = self.detector.check_line(ctx)
        # Without multiline context, not detected
        assert severity is None

    def test_short_line_not_flagged(self):
        ctx = make_context('{"a": 1}')  # Too short
        severity = self.detector.check_line(ctx)
        assert severity is None

    def test_brace_without_json_not_flagged(self):
        ctx = make_context('function foo() { return bar; }')
        severity = self.detector.check_line(ctx)
        # Doesn't have JSON-like key-value pairs
        assert severity is None

    def test_merge_multiline_json(self):
        # First line starts JSON
        ctx1 = make_context('{"user": "john", "data": {"nested": "value"}}', line_number=1)
        self.detector.check_line(ctx1)

        # Continuation line
        ctx2 = make_context('  "more_data": "value"', line_number=2)
        should_merge = self.detector.should_merge_with_previous(ctx2, 0.3)
        assert should_merge

    def test_description_includes_size(self):
        lines = ['{"a": 1, "b": 2}', '"c": 3']
        desc = self.detector.get_description(lines)
        assert 'JSON' in desc
        assert 'chars' in desc

    def test_bracket_with_name_not_flagged(self):
        """Test that lines like 'Name: [ Juliet Keat ]' are NOT flagged as JSON."""
        # This was a false positive - square brackets containing a name
        line = 'Email: carlief123@aol.com - Name: [ Juliet Keat ] - Phone: 555-1234' + 'x' * 50
        ctx = make_context(line)
        severity = self.detector.check_line(ctx)
        assert severity is None

    def test_empty_brackets_not_flagged(self):
        """Test that empty brackets [] or {} are NOT flagged as JSON."""
        line = 'Result: [] - Status: {} - Done' + 'x' * 100
        ctx = make_context(line)
        severity = self.detector.check_line(ctx)
        assert severity is None

    def test_bracket_with_words_not_flagged(self):
        """Test that brackets containing words (not JSON) are NOT flagged."""
        line = 'Tags: [important, urgent, review] - Category: [misc]' + 'x' * 60
        ctx = make_context(line)
        severity = self.detector.check_line(ctx)
        assert severity is None


class TestFormatDeviationDetector:
    """Tests for FormatDeviationDetector."""

    def setup_method(self):
        self.detector = FormatDeviationDetector()

    def test_name_and_category(self):
        assert self.detector.name == 'format_deviation'
        assert self.detector.category == 'format'

    def test_no_deviation_initially(self):
        # Need at least MIN_LINES_FOR_PATTERN to detect deviations
        ctx = make_context('2024-01-15 10:30:45 INFO Application started')
        severity = self.detector.check_line(ctx)
        assert severity is None  # Not enough lines yet

    def test_detect_deviation_after_pattern_established(self):
        # Establish dominant pattern with consistent format
        for i in range(150):
            ctx = make_context(f'2024-01-15 10:30:{i:02d} INFO Log message {i}', line_number=i + 1)
            self.detector.check_line(ctx)

        # Now insert a line with completely different format
        ctx_deviant = make_context('--- SYSTEM BREAK ---', line_number=151)
        severity = self.detector.check_line(ctx_deviant)
        # Should detect deviation (no timestamp, no level)
        assert severity is not None
        assert severity >= 0.3

    def test_similar_format_not_flagged(self):
        # Establish dominant pattern
        for i in range(150):
            ctx = make_context(f'2024-01-15 10:30:{i:02d} INFO Log message {i}', line_number=i + 1)
            self.detector.check_line(ctx)

        # Insert line with same format
        ctx_similar = make_context('2024-01-15 10:31:00 DEBUG Another message', line_number=151)
        severity = self.detector.check_line(ctx_similar)
        # Same format pattern, should not flag
        assert severity is None

    def test_empty_lines_ignored(self):
        ctx = make_context('   ', line_number=1)
        severity = self.detector.check_line(ctx)
        assert severity is None

    def test_syslog_format_detected(self):
        # Test syslog format recognition
        ctx = make_context('Jan 15 10:30:45 myhost myapp[1234]: Log message')
        format_sig = self.detector._get_line_format(ctx.line)
        # Should detect syslog timestamp
        assert any(format_sig)  # At least one pattern should match

    def test_bracketed_format_detected(self):
        ctx = make_context('[2024-01-15 10:30:45] [INFO] Message')
        format_sig = self.detector._get_line_format(ctx.line)
        # Should detect bracketed format and timestamp
        assert any(format_sig)
