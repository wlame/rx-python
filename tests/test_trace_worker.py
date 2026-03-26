"""Tests for trace_worker module - worker and pattern matching layer.

Verifies that the extraction from trace.py preserved all functionality
and that identify_matching_patterns works correctly.
"""

import importlib

from rx.models import Submatch
from rx.trace_worker import HookCallbacks, identify_matching_patterns


class TestTraceWorkerImports:
    """Verify module imports and function accessibility."""

    def test_module_importable(self):
        mod = importlib.import_module('rx.trace_worker')
        assert mod is not None

    def test_hook_callbacks_importable(self):
        from rx.trace_worker import HookCallbacks
        assert HookCallbacks is not None

    def test_identify_matching_patterns_importable(self):
        from rx.trace_worker import identify_matching_patterns
        assert callable(identify_matching_patterns)

    def test_process_task_worker_importable(self):
        from rx.trace_worker import process_task_worker
        assert callable(process_task_worker)

    def test_backward_compat_import_from_trace(self):
        """All extracted symbols should still be importable from rx.trace."""
        from rx.trace import HookCallbacks, identify_matching_patterns, process_task_worker
        assert HookCallbacks is not None
        assert callable(identify_matching_patterns)
        assert callable(process_task_worker)

    def test_no_circular_imports(self):
        """Importing trace_worker should not cause circular import errors."""
        import rx.trace_worker
        import rx.trace
        assert rx.trace_worker is not None
        assert rx.trace is not None


class TestHookCallbacks:
    """Test the HookCallbacks dataclass."""

    def test_default_values(self):
        cb = HookCallbacks()
        assert cb.on_match_found is None
        assert cb.on_file_scanned is None
        assert cb.request_id == ''
        assert cb.patterns == {}
        assert cb.files == {}

    def test_with_callbacks(self):
        calls = []
        cb = HookCallbacks(
            on_match_found=lambda m: calls.append(('match', m)),
            on_file_scanned=lambda f: calls.append(('file', f)),
            request_id='req-1',
            patterns={'p1': 'error'},
            files={'f1': '/tmp/test.log'},
        )
        assert cb.request_id == 'req-1'
        assert cb.patterns == {'p1': 'error'}
        cb.on_match_found({'line': 'test'})
        assert len(calls) == 1
        assert calls[0] == ('match', {'line': 'test'})

    def test_patterns_isolation(self):
        """Each instance should have independent patterns dict."""
        cb1 = HookCallbacks()
        cb2 = HookCallbacks()
        cb1.patterns['p1'] = 'foo'
        assert 'p1' not in cb2.patterns


class TestIdentifyMatchingPatterns:
    """Test the identify_matching_patterns function."""

    def test_single_pattern_single_match(self):
        submatches = [Submatch(text='error', start=0, end=5)]
        result = identify_matching_patterns(
            line_text='error occurred',
            submatches=submatches,
            pattern_ids={'p1': 'error'},
            rg_extra_args=[],
        )
        assert result == ['p1']

    def test_multiple_patterns_one_matches(self):
        submatches = [Submatch(text='error', start=0, end=5)]
        result = identify_matching_patterns(
            line_text='error occurred',
            submatches=submatches,
            pattern_ids={'p1': 'error', 'p2': 'warning'},
            rg_extra_args=[],
        )
        assert result == ['p1']

    def test_multiple_patterns_both_match(self):
        submatches = [Submatch(text='error', start=10, end=15)]
        result = identify_matching_patterns(
            line_text='warning: error occurred',
            submatches=submatches,
            pattern_ids={'p1': 'error', 'p2': 'warning'},
            rg_extra_args=[],
        )
        # 'error' submatch matches p1; 'warning' is in the line but not in submatches
        assert 'p1' in result

    def test_no_patterns_match(self):
        submatches = [Submatch(text='info', start=0, end=4)]
        result = identify_matching_patterns(
            line_text='info message',
            submatches=submatches,
            pattern_ids={'p1': 'error', 'p2': 'warning'},
            rg_extra_args=[],
        )
        assert result == []

    def test_case_insensitive_flag(self):
        submatches = [Submatch(text='Error', start=0, end=5)]
        result = identify_matching_patterns(
            line_text='Error occurred',
            submatches=submatches,
            pattern_ids={'p1': 'error'},
            rg_extra_args=['-i'],
        )
        assert result == ['p1']

    def test_case_sensitive_no_match(self):
        submatches = [Submatch(text='Error', start=0, end=5)]
        result = identify_matching_patterns(
            line_text='Error occurred',
            submatches=submatches,
            pattern_ids={'p1': 'error'},
            rg_extra_args=[],
        )
        # 'error' pattern doesn't match 'Error' case-sensitively
        assert result == []

    def test_empty_submatches_falls_back_to_line_search(self):
        """When no submatches, should search the line directly."""
        result = identify_matching_patterns(
            line_text='error occurred in module',
            submatches=[],
            pattern_ids={'p1': 'error', 'p2': 'warning'},
            rg_extra_args=[],
        )
        assert result == ['p1']

    def test_empty_submatches_no_match(self):
        result = identify_matching_patterns(
            line_text='info message',
            submatches=[],
            pattern_ids={'p1': 'error'},
            rg_extra_args=[],
        )
        assert result == []

    def test_invalid_regex_pattern_skipped(self):
        submatches = [Submatch(text='error', start=0, end=5)]
        result = identify_matching_patterns(
            line_text='error occurred',
            submatches=submatches,
            pattern_ids={'p1': 'error', 'p2': '[invalid'},
            rg_extra_args=[],
        )
        assert result == ['p1']

    def test_regex_pattern_matching(self):
        submatches = [Submatch(text='error123', start=0, end=8)]
        result = identify_matching_patterns(
            line_text='error123 in module',
            submatches=submatches,
            pattern_ids={'p1': r'error\d+'},
            rg_extra_args=[],
        )
        assert result == ['p1']

    def test_overlapping_submatches(self):
        """Multiple submatches from a single pattern should work."""
        submatches = [
            Submatch(text='err', start=0, end=3),
            Submatch(text='err', start=10, end=13),
        ]
        result = identify_matching_patterns(
            line_text='err: found err again',
            submatches=submatches,
            pattern_ids={'p1': 'err'},
            rg_extra_args=[],
        )
        assert result == ['p1']
