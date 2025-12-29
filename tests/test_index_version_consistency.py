"""Tests for index version consistency across all index types.

This module verifies that all index creation methods use the same version (UNIFIED_INDEX_VERSION = 2)
and that indexes created by different modules are compatible.
"""

import gzip
import os
import tempfile

import pytest

from rx.compressed_index import (
    build_compressed_index,
    get_compressed_index_path,
)
from rx.indexer import FileIndexer
from rx.models import UnifiedFileIndex
from rx.unified_index import UNIFIED_INDEX_VERSION, get_index_path
from rx.unified_index import build_index as build_unified_index


# Check if zstandard module is available
try:
    import zstandard

    ZSTANDARD_AVAILABLE = True
except ImportError:
    ZSTANDARD_AVAILABLE = False

# Check if t2sz is available
from rx.seekable_zstd import check_t2sz_available, create_seekable_zstd


CAN_CREATE_SEEKABLE = check_t2sz_available() or ZSTANDARD_AVAILABLE

if CAN_CREATE_SEEKABLE:
    from rx.seekable_index import (
        build_index as build_seekable_index,
    )
    from rx.seekable_index import (
        get_index_path as get_seekable_index_path,
    )


class TestUnifiedVersionConstant:
    """Verify UNIFIED_INDEX_VERSION is set correctly."""

    def test_unified_index_version_is_2(self):
        """Test that UNIFIED_INDEX_VERSION equals 2."""
        assert UNIFIED_INDEX_VERSION == 2, f'Expected version 2, got {UNIFIED_INDEX_VERSION}'


class TestTextFileIndexVersion:
    """Test that plain text file indexes use correct version."""

    @pytest.fixture
    def temp_text_file(self):
        """Create a temporary text file."""
        content = '\n'.join([f'Line {i}: Some content here' for i in range(100)])
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write(content)
            temp_path = f.name
        yield temp_path
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        # Cleanup index
        index_path = get_index_path(temp_path)
        if index_path.exists():
            index_path.unlink()

    def test_build_unified_index_uses_version_2(self, temp_text_file):
        """Test that build_index from unified_index.py creates version 2."""
        result = build_unified_index(temp_text_file)
        # The result is an IndexBuildResult, not a dict - version is used when saving
        # We need to use FileIndexer to get a proper UnifiedFileIndex
        indexer = FileIndexer(analyze=True)
        index = indexer.index_file(temp_text_file)
        assert index.version == UNIFIED_INDEX_VERSION
        assert index.version == 2

    def test_file_indexer_creates_version_2(self, temp_text_file):
        """Test that FileIndexer creates indexes with version 2."""
        indexer = FileIndexer(analyze=True)
        index = indexer.index_file(temp_text_file)

        assert isinstance(index, UnifiedFileIndex)
        assert index.version == 2
        assert index.version == UNIFIED_INDEX_VERSION


class TestCompressedFileIndexVersion:
    """Test that compressed file indexes use correct version."""

    @pytest.fixture
    def temp_gzip_file(self, tmp_path, monkeypatch):
        """Create a temporary gzip file."""
        monkeypatch.setenv('RX_CACHE_DIR', str(tmp_path))

        lines = [f'Line {i:04d}: Content for line {i}\n' for i in range(1, 101)]
        content = ''.join(lines).encode('utf-8')

        gzip_path = tmp_path / 'test.gz'
        with gzip.open(gzip_path, 'wb') as gz:
            gz.write(content)

        yield str(gzip_path)

    def test_build_compressed_index_uses_version_2(self, temp_gzip_file):
        """Test that build_compressed_index creates version 2."""
        index_data = build_compressed_index(temp_gzip_file)

        assert index_data['version'] == UNIFIED_INDEX_VERSION
        assert index_data['version'] == 2


@pytest.mark.skipif(not CAN_CREATE_SEEKABLE, reason='Neither t2sz nor zstandard available')
class TestSeekableZstdIndexVersion:
    """Test that seekable zstd file indexes use correct version."""

    @pytest.fixture
    def temp_seekable_zstd_file(self, tmp_path, monkeypatch):
        """Create a temporary seekable zstd file."""
        monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'cache'))

        input_file = tmp_path / 'input.txt'
        lines = [f'Line number {i}\n' for i in range(500)]
        content = ''.join(lines)
        input_file.write_text(content)

        output_file = tmp_path / 'output.zst'
        create_seekable_zstd(
            input_file,
            output_file,
            frame_size_bytes=2 * 1024,
            compression_level=1,
        )

        yield output_file

    def test_build_seekable_index_uses_version_2(self, temp_seekable_zstd_file):
        """Test that build_index from seekable_index.py creates version 2."""
        index = build_seekable_index(temp_seekable_zstd_file)

        assert index.version == UNIFIED_INDEX_VERSION
        assert index.version == 2

    def test_seekable_index_to_dict_has_version_2(self, temp_seekable_zstd_file):
        """Test that SeekableIndex.to_dict() includes version 2."""
        index = build_seekable_index(temp_seekable_zstd_file)
        index_dict = index.to_dict()

        assert index_dict['version'] == UNIFIED_INDEX_VERSION
        assert index_dict['version'] == 2


class TestIndexPathConsistency:
    """Test that all index types use the same path format."""

    def test_unified_and_compressed_use_same_path_format(self, tmp_path):
        """Test that unified_index and compressed_index produce same path format."""
        test_file = tmp_path / 'test.gz'
        test_file.touch()

        unified_path = get_index_path(str(test_file))
        compressed_path = get_compressed_index_path(str(test_file))

        # Both should use {filename}_{hash}.json format
        assert unified_path == compressed_path, (
            f'Path mismatch:\n  unified: {unified_path}\n  compressed: {compressed_path}'
        )

    @pytest.mark.skipif(not CAN_CREATE_SEEKABLE, reason='Neither t2sz nor zstandard available')
    def test_unified_and_seekable_use_same_path_format(self, tmp_path):
        """Test that unified_index and seekable_index produce same path format."""
        test_file = tmp_path / 'test.zst'
        test_file.touch()

        unified_path = get_index_path(str(test_file))
        seekable_path = get_seekable_index_path(str(test_file))

        # Both should use {filename}_{hash}.json format
        assert unified_path == seekable_path, f'Path mismatch:\n  unified: {unified_path}\n  seekable: {seekable_path}'


class TestAllIndexTypesUseVersion2:
    """Integration test ensuring all index creation paths use version 2."""

    @pytest.fixture
    def temp_files(self, tmp_path, monkeypatch):
        """Create temporary files for all index types."""
        monkeypatch.setenv('RX_CACHE_DIR', str(tmp_path / 'cache'))
        monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'cache'))

        files = {}

        # Plain text file
        text_file = tmp_path / 'test.txt'
        text_file.write_text('\n'.join([f'Line {i}' for i in range(100)]))
        files['text'] = text_file

        # Gzip file
        gzip_file = tmp_path / 'test.gz'
        with gzip.open(gzip_file, 'wt') as f:
            f.write('\n'.join([f'Line {i}' for i in range(100)]))
        files['gzip'] = gzip_file

        # Seekable zstd file (if available)
        if CAN_CREATE_SEEKABLE:
            input_file = tmp_path / 'input.txt'
            input_file.write_text('\n'.join([f'Line {i}' for i in range(100)]))
            zst_file = tmp_path / 'test.zst'
            create_seekable_zstd(input_file, zst_file, frame_size_bytes=1024)
            files['seekable_zstd'] = zst_file

        yield files

    def test_all_index_types_use_version_2(self, temp_files):
        """Comprehensive test that all index types use version 2."""
        versions_found = {}

        # Test unified/text file index
        indexer = FileIndexer(analyze=True)
        text_index = indexer.index_file(str(temp_files['text']))
        versions_found['unified_text'] = text_index.version

        # Test compressed index
        compressed_index = build_compressed_index(str(temp_files['gzip']))
        versions_found['compressed_gzip'] = compressed_index['version']

        # Test seekable zstd index (if available)
        if 'seekable_zstd' in temp_files:
            seekable_index = build_seekable_index(temp_files['seekable_zstd'])
            versions_found['seekable_zstd'] = seekable_index.version

        # All versions should be 2
        for index_type, version in versions_found.items():
            assert version == 2, f'{index_type} index has version {version}, expected 2'

        # All versions should match UNIFIED_INDEX_VERSION
        for index_type, version in versions_found.items():
            assert version == UNIFIED_INDEX_VERSION, (
                f'{index_type} index version {version} != UNIFIED_INDEX_VERSION ({UNIFIED_INDEX_VERSION})'
            )


class TestCacheFormatConsistency:
    """Test that cached indexes can be loaded back correctly."""

    @pytest.fixture
    def temp_gzip_with_cache(self, tmp_path, monkeypatch):
        """Create a gzip file with isolated cache directory."""
        cache_dir = tmp_path / 'cache'
        monkeypatch.setenv('RX_CACHE_DIR', str(cache_dir))

        gzip_file = tmp_path / 'test.gz'
        content = '\n'.join([f'Line {i}: Some content here for testing' for i in range(100)])
        with gzip.open(gzip_file, 'wt') as f:
            f.write(content)

        yield gzip_file, cache_dir

    def test_compressed_file_index_can_be_reloaded(self, temp_gzip_with_cache):
        """Test that indexing a compressed file creates a cache that can be reloaded.

        This is a regression test for the bug where compressed_index saved a dict format
        but unified_index expected UnifiedFileIndex format, causing validation errors.
        """
        from rx.unified_index import load_index

        gzip_file, cache_dir = temp_gzip_with_cache

        # Index the compressed file with analyze=True
        indexer = FileIndexer(analyze=True)
        first_index = indexer.index_file(str(gzip_file))

        # Verify index was created
        assert first_index is not None
        assert isinstance(first_index, UnifiedFileIndex)
        assert first_index.line_count == 100
        assert first_index.file_type.value == 'compressed'

        # Now try to load it back from cache - this should NOT raise validation errors
        reloaded = load_index(str(gzip_file))

        assert reloaded is not None, 'Failed to reload cached index - cache format mismatch?'
        assert isinstance(reloaded, UnifiedFileIndex)
        assert reloaded.version == UNIFIED_INDEX_VERSION
        assert reloaded.line_count == first_index.line_count
        assert reloaded.file_type == first_index.file_type

    def test_compressed_file_index_second_run_uses_cache(self, temp_gzip_with_cache):
        """Test that second indexing run uses cached index without rebuilding."""
        gzip_file, cache_dir = temp_gzip_with_cache

        # First run - builds index
        indexer1 = FileIndexer(analyze=True)
        first_index = indexer1.index_file(str(gzip_file))
        first_build_time = first_index.build_time_seconds

        # Second run - should use cache (much faster)
        indexer2 = FileIndexer(analyze=True)
        second_index = indexer2.index_file(str(gzip_file))

        # Both should return valid indexes
        assert first_index is not None
        assert second_index is not None

        # Data should match
        assert second_index.line_count == first_index.line_count
        assert second_index.source_path == first_index.source_path

    def test_no_validation_errors_on_compressed_index_load(self, temp_gzip_with_cache, caplog):
        """Test that loading compressed file index doesn't produce validation warnings."""
        import logging

        gzip_file, cache_dir = temp_gzip_with_cache

        # Index the file
        indexer = FileIndexer(analyze=True)
        indexer.index_file(str(gzip_file))

        # Clear log and try to load
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            from rx.unified_index import load_index

            loaded = load_index(str(gzip_file))

        # Should load successfully
        assert loaded is not None

        # Should NOT have validation error warnings
        validation_errors = [
            r for r in caplog.records if 'validation error' in r.message.lower() or 'file_type' in r.message.lower()
        ]
        assert len(validation_errors) == 0, f'Validation errors found in logs: {[r.message for r in validation_errors]}'
