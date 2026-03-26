"""Tests for the trace module split into trace.py, trace_compressed.py, trace_worker.py.

Verifies that:
- All public API re-exports work from rx.trace
- No circular imports exist between the three modules
- trace.py stays under 1000 lines
"""

import importlib
import pathlib


class TestReExports:
    """All public symbols should be importable from rx.trace for backward compat."""

    def test_parse_paths(self):
        from rx.trace import parse_paths
        assert callable(parse_paths)

    def test_hook_callbacks(self):
        from rx.trace import HookCallbacks
        assert HookCallbacks is not None

    def test_identify_matching_patterns(self):
        from rx.trace import identify_matching_patterns
        assert callable(identify_matching_patterns)

    def test_debug_mode(self):
        from rx.trace import DEBUG_MODE
        assert isinstance(DEBUG_MODE, bool)

    def test_process_task_worker(self):
        from rx.trace import process_task_worker
        assert callable(process_task_worker)

    def test_process_compressed_file(self):
        from rx.trace import process_compressed_file
        assert callable(process_compressed_file)

    def test_process_seekable_zstd_file(self):
        from rx.trace import process_seekable_zstd_file
        assert callable(process_seekable_zstd_file)


class TestDirectImports:
    """Symbols should also be importable from their new canonical locations."""

    def test_trace_compressed_imports(self):
        from rx.trace_compressed import (
            process_compressed_file,
            process_seekable_zstd_file,
            process_seekable_zstd_frame_batch,
        )
        assert callable(process_compressed_file)
        assert callable(process_seekable_zstd_file)
        assert callable(process_seekable_zstd_frame_batch)

    def test_trace_worker_imports(self):
        from rx.trace_worker import (
            HookCallbacks,
            identify_matching_patterns,
            process_task_worker,
        )
        assert HookCallbacks is not None
        assert callable(identify_matching_patterns)
        assert callable(process_task_worker)


class TestNoCircularImports:
    """Import each module in isolation to verify no circular dependencies."""

    def test_trace_compressed_standalone(self):
        mod = importlib.import_module('rx.trace_compressed')
        assert mod is not None

    def test_trace_worker_standalone(self):
        mod = importlib.import_module('rx.trace_worker')
        assert mod is not None

    def test_trace_standalone(self):
        mod = importlib.import_module('rx.trace')
        assert mod is not None

    def test_import_order_compressed_then_trace(self):
        import rx.trace_compressed
        import rx.trace
        assert rx.trace_compressed is not None
        assert rx.trace is not None

    def test_import_order_worker_then_trace(self):
        import rx.trace_worker
        import rx.trace
        assert rx.trace_worker is not None
        assert rx.trace is not None

    def test_import_order_trace_then_children(self):
        import rx.trace
        import rx.trace_compressed
        import rx.trace_worker
        assert rx.trace is not None


class TestModuleSize:
    """Verify trace.py stays manageable after split."""

    def test_trace_under_1000_lines(self):
        trace_path = pathlib.Path('src/rx/trace.py')
        line_count = len(trace_path.read_text().splitlines())
        assert line_count < 1000, f'trace.py has {line_count} lines, expected < 1000'

    def test_trace_compressed_exists(self):
        assert pathlib.Path('src/rx/trace_compressed.py').exists()

    def test_trace_worker_exists(self):
        assert pathlib.Path('src/rx/trace_worker.py').exists()


class TestIdentityConsistency:
    """Verify re-exported symbols are the same objects."""

    def test_hook_callbacks_identity(self):
        from rx.trace import HookCallbacks as HC1
        from rx.trace_worker import HookCallbacks as HC2
        assert HC1 is HC2

    def test_identify_matching_patterns_identity(self):
        from rx.trace import identify_matching_patterns as IMP1
        from rx.trace_worker import identify_matching_patterns as IMP2
        assert IMP1 is IMP2

    def test_process_compressed_file_identity(self):
        from rx.trace import process_compressed_file as PCF1
        from rx.trace_compressed import process_compressed_file as PCF2
        assert PCF1 is PCF2
