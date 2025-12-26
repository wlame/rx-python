"""Tests for RX_CACHE_DIR environment variable configuration."""

import os
import tempfile
from pathlib import Path


class TestRxCacheDirConfig:
    """Test that RX_CACHE_DIR env var is respected by all cache modules."""

    def test_get_rx_cache_base_default(self, monkeypatch):
        """Test default cache directory when no env vars are set."""
        monkeypatch.delenv('RX_CACHE_DIR', raising=False)
        monkeypatch.delenv('XDG_CACHE_HOME', raising=False)

        from rx.utils import get_rx_cache_base

        result = get_rx_cache_base()
        assert result == Path.home() / '.cache' / 'rx'

    def test_get_rx_cache_base_with_rx_cache_dir(self, monkeypatch):
        """Test that RX_CACHE_DIR takes priority."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv('RX_CACHE_DIR', tmpdir)
            monkeypatch.setenv('XDG_CACHE_HOME', '/should/be/ignored')

            from rx.utils import get_rx_cache_base

            result = get_rx_cache_base()
            assert result == Path(tmpdir) / 'rx'

    def test_get_rx_cache_base_with_xdg_cache_home(self, monkeypatch):
        """Test that XDG_CACHE_HOME is used when RX_CACHE_DIR is not set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.delenv('RX_CACHE_DIR', raising=False)
            monkeypatch.setenv('XDG_CACHE_HOME', tmpdir)

            from rx.utils import get_rx_cache_base

            result = get_rx_cache_base()
            assert result == Path(tmpdir) / 'rx'

    def test_get_rx_cache_dir_creates_subdirectory(self, monkeypatch):
        """Test that get_rx_cache_dir creates the subdirectory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv('RX_CACHE_DIR', tmpdir)

            from rx.utils import get_rx_cache_dir

            result = get_rx_cache_dir('test_subdir')
            assert result == Path(tmpdir) / 'rx' / 'test_subdir'
            assert result.exists()


class TestIndexCacheDir:
    """Test that index module respects RX_CACHE_DIR."""

    def test_index_cache_dir_uses_rx_cache_dir(self, monkeypatch):
        """Test that index.get_cache_dir uses RX_CACHE_DIR."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv('RX_CACHE_DIR', tmpdir)

            from rx.index import get_cache_dir

            result = get_cache_dir()
            assert str(result).startswith(tmpdir)
            assert 'indexes' in str(result)


class TestTraceCacheDir:
    """Test that trace_cache module respects RX_CACHE_DIR."""

    def test_trace_cache_dir_uses_rx_cache_dir(self, monkeypatch):
        """Test that trace_cache.get_trace_cache_dir uses RX_CACHE_DIR."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv('RX_CACHE_DIR', tmpdir)

            from rx.trace_cache import get_trace_cache_dir

            result = get_trace_cache_dir()
            assert str(result).startswith(tmpdir)
            assert 'trace_cache' in str(result)


class TestUnifiedIndexCacheDir:
    """Test that unified_index module respects RX_CACHE_DIR."""

    def test_unified_index_cache_dir_uses_rx_cache_dir(self, monkeypatch):
        """Test that unified_index.get_index_cache_dir uses RX_CACHE_DIR."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv('RX_CACHE_DIR', tmpdir)

            from rx.unified_index import get_index_cache_dir

            result = get_index_cache_dir()
            assert str(result).startswith(tmpdir)
            assert 'indexes' in str(result)


class TestSeekableIndexCacheDir:
    """Test that seekable_index module respects RX_CACHE_DIR."""

    def test_seekable_index_dir_uses_rx_cache_dir(self, monkeypatch):
        """Test that seekable_index.get_index_dir uses RX_CACHE_DIR."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv('RX_CACHE_DIR', tmpdir)

            from rx.seekable_index import get_index_dir

            result = get_index_dir()
            assert str(result).startswith(tmpdir)
            assert 'indexes' in str(result)


class TestCompressedIndexCacheDir:
    """Test that compressed_index module respects RX_CACHE_DIR."""

    def test_compressed_index_dir_uses_rx_cache_dir(self, monkeypatch):
        """Test that compressed_index.get_compressed_index_dir uses RX_CACHE_DIR (unified indexes dir)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv('RX_CACHE_DIR', tmpdir)

            from rx.compressed_index import get_compressed_index_dir

            result = get_compressed_index_dir()
            assert str(result).startswith(tmpdir)
            assert 'indexes' in str(result)


class TestCacheIntegration:
    """Integration tests for cache directory configuration."""

    def test_unified_index_writes_to_custom_cache_dir(self, monkeypatch):
        """Test that unified index actually writes cache to custom directory."""
        from datetime import datetime

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv('RX_CACHE_DIR', tmpdir)

            # Create a test file
            test_file = os.path.join(tmpdir, 'test.txt')
            with open(test_file, 'w') as f:
                f.write('test content\n' * 100)

            from rx.models import FileType, UnifiedFileIndex
            from rx.unified_index import (
                UNIFIED_INDEX_VERSION,
                get_index_cache_dir,
                load_index,
                save_index,
            )

            # Create and save an index
            stat = os.stat(test_file)
            index = UnifiedFileIndex(
                version=UNIFIED_INDEX_VERSION,
                source_path=test_file,
                source_modified_at=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                source_size_bytes=stat.st_size,
                created_at=datetime.utcnow().isoformat(),
                build_time_seconds=0.1,
                file_type=FileType.TEXT,
                is_text=True,
                line_index=[[1, 0]],
                line_count=100,
            )
            save_index(index)

            # Verify it was written to the custom directory
            cache_dir = get_index_cache_dir()
            assert str(cache_dir).startswith(tmpdir)
            assert any(cache_dir.iterdir())  # Should have at least one file

            # Verify we can load it back
            loaded = load_index(test_file)
            assert loaded is not None
            assert loaded.line_count == 100

    def test_trace_cache_writes_to_custom_cache_dir(self, monkeypatch):
        """Test that trace cache writes to custom directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv('RX_CACHE_DIR', tmpdir)

            # Create a test file
            test_file = os.path.join(tmpdir, 'test.txt')
            with open(test_file, 'w') as f:
                f.write('error line\n' * 100)

            from rx.trace_cache import (
                build_cache_from_matches,
                get_trace_cache_dir,
                get_trace_cache_path,
                save_trace_cache,
            )

            patterns = ['error']
            rg_flags = []

            # Build cache data
            matches = [{'pattern_index': 0, 'offset': 0, 'line_number': 1}]
            cache_data = build_cache_from_matches(test_file, patterns, rg_flags, matches)

            # Save cache
            cache_path = get_trace_cache_path(test_file, patterns, rg_flags)
            save_trace_cache(cache_data, cache_path)

            # Verify it was written to the custom directory
            cache_dir = get_trace_cache_dir()
            assert str(cache_dir).startswith(tmpdir)
            assert cache_path.exists()
            assert str(cache_path).startswith(tmpdir)
