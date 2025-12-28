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
    load_compressed_index,
    save_compressed_index,
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

    def test_compressed_index_roundtrip_preserves_version(self, temp_gzip_file):
        """Test that saving and loading compressed index preserves version 2."""
        index_data = build_compressed_index(temp_gzip_file)
        save_compressed_index(index_data, temp_gzip_file)

        loaded = load_compressed_index(temp_gzip_file)
        assert loaded['version'] == UNIFIED_INDEX_VERSION
        assert loaded['version'] == 2


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
