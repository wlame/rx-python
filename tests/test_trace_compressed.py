"""Tests for trace_compressed module - compressed file processing layer.

Verifies that the extraction from trace.py preserved all functionality
and that the module's public API is accessible.
"""

import importlib

from rx.models import ContextLine, Submatch


class TestTraceCompressedImports:
    """Verify module imports and function accessibility."""

    def test_module_importable(self):
        mod = importlib.import_module('rx.trace_compressed')
        assert mod is not None

    def test_process_compressed_file_importable(self):
        from rx.trace_compressed import process_compressed_file
        assert callable(process_compressed_file)

    def test_process_seekable_zstd_frame_batch_importable(self):
        from rx.trace_compressed import process_seekable_zstd_frame_batch
        assert callable(process_seekable_zstd_frame_batch)

    def test_process_seekable_zstd_file_importable(self):
        from rx.trace_compressed import process_seekable_zstd_file
        assert callable(process_seekable_zstd_file)

    def test_backward_compat_import_from_trace(self):
        """Functions should still be accessible via rx.trace imports."""
        from rx.trace import process_compressed_file, process_seekable_zstd_file
        assert callable(process_compressed_file)
        assert callable(process_seekable_zstd_file)

    def test_no_circular_imports(self):
        """Importing trace_compressed should not cause circular import errors."""
        import rx.trace_compressed
        import rx.trace
        assert rx.trace_compressed is not None
        assert rx.trace is not None


class TestProcessCompressedFileValidation:
    """Test input validation for process_compressed_file."""

    def test_rejects_non_compressed_file(self, tmp_path):
        """Should raise ValueError for non-compressed files."""
        plain_file = tmp_path / 'plain.txt'
        plain_file.write_text('hello world\n')

        from rx.trace_compressed import process_compressed_file
        import pytest

        with pytest.raises(ValueError, match='not compressed'):
            process_compressed_file(
                filepath=str(plain_file),
                pattern_ids={'p1': 'hello'},
            )


class TestProcessCompressedFileSignature:
    """Test that function signatures match expected parameters."""

    def test_process_compressed_file_params(self):
        import inspect
        from rx.trace_compressed import process_compressed_file

        sig = inspect.signature(process_compressed_file)
        params = list(sig.parameters.keys())
        assert 'filepath' in params
        assert 'pattern_ids' in params
        assert 'rg_extra_args' in params
        assert 'context_before' in params
        assert 'context_after' in params
        assert 'max_results' in params

    def test_process_seekable_zstd_frame_batch_params(self):
        import inspect
        from rx.trace_compressed import process_seekable_zstd_frame_batch

        sig = inspect.signature(process_seekable_zstd_frame_batch)
        params = list(sig.parameters.keys())
        assert 'filepath' in params
        assert 'frame_indices' in params
        assert 'frame_infos' in params
        assert 'pattern_ids' in params

    def test_process_seekable_zstd_file_params(self):
        import inspect
        from rx.trace_compressed import process_seekable_zstd_file

        sig = inspect.signature(process_seekable_zstd_file)
        params = list(sig.parameters.keys())
        assert 'filepath' in params
        assert 'pattern_ids' in params
        assert 'max_results' in params
