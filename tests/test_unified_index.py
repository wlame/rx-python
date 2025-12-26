"""Tests for unified index cache module."""

import json
import os
import tempfile
import time
from datetime import datetime

import pytest

from rx.models import FileType, UnifiedFileIndex
from rx.unified_index import (
    UNIFIED_INDEX_VERSION,
    clear_all_indexes,
    delete_index,
    get_cache_key,
    get_cached_line_count,
    get_index_cache_dir,
    get_index_path,
    get_large_file_threshold_bytes,
    is_index_valid,
    load_index,
    save_index,
    should_create_index,
)


class TestCacheKeyGeneration:
    """Test cache key generation."""

    def test_cache_key_consistent(self):
        """Test that cache key is consistent for same path."""
        path = '/path/to/test/file.txt'
        key1 = get_cache_key(path)
        key2 = get_cache_key(path)
        assert key1 == key2

    def test_cache_key_different_for_different_paths(self):
        """Test that different paths get different keys."""
        key1 = get_cache_key('/path/to/file1.txt')
        key2 = get_cache_key('/path/to/file2.txt')
        assert key1 != key2

    def test_cache_key_includes_basename(self):
        """Test that cache key includes sanitized basename."""
        path = '/some/path/myfile.log'
        key = get_cache_key(path)
        assert 'myfile.log' in key

    def test_cache_key_sanitizes_special_chars(self):
        """Test that special characters in filename are sanitized."""
        path = '/path/file with spaces & symbols!.txt'
        key = get_cache_key(path)
        # Should not contain spaces or special chars
        assert ' ' not in key
        assert '&' not in key
        assert '!' not in key


class TestCacheDirectory:
    """Test cache directory management."""

    def test_cache_dir_created(self):
        """Test that cache directory is created."""
        cache_dir = get_index_cache_dir()
        assert cache_dir.exists()
        assert cache_dir.is_dir()

    def test_cache_dir_location(self):
        """Test cache directory is in expected location (unified indexes dir)."""
        cache_dir = get_index_cache_dir()
        # Should use the 'indexes' subdirectory
        assert 'indexes' in str(cache_dir)
        # Should be within the RX_CACHE_DIR (set by conftest.py fixture)
        rx_cache_dir = os.environ.get('RX_CACHE_DIR')
        if rx_cache_dir:
            assert str(cache_dir).startswith(rx_cache_dir)


class TestIndexSaveAndLoad:
    """Test saving and loading index."""

    def setup_method(self):
        """Create test file before each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, 'test.txt')
        with open(self.test_file, 'w') as f:
            f.write('Line 1\nLine 2\nLine 3\n')

    def teardown_method(self):
        """Clean up after each test."""
        # Clean up test file
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        os.rmdir(self.temp_dir)

        # Clean up cache
        delete_index(self.test_file)

    def _create_test_index(self) -> UnifiedFileIndex:
        """Create a test index for the test file."""
        stat = os.stat(self.test_file)
        return UnifiedFileIndex(
            version=UNIFIED_INDEX_VERSION,
            source_path=self.test_file,
            source_modified_at=datetime.fromtimestamp(stat.st_mtime).isoformat(),
            source_size_bytes=stat.st_size,
            created_at=datetime.utcnow().isoformat(),
            build_time_seconds=0.1,
            file_type=FileType.TEXT,
            is_text=True,
            line_index=[[1, 0]],
            line_count=3,
            empty_line_count=0,
            analysis_performed=True,
        )

    def test_save_and_load_index(self):
        """Test saving and loading index works."""
        index = self._create_test_index()

        # Save index
        saved = save_index(index)
        assert saved is True

        # Load index
        loaded = load_index(self.test_file)
        assert loaded is not None
        assert loaded.source_path == self.test_file
        assert loaded.line_count == 3
        assert loaded.analysis_performed is True

    def test_index_file_exists_on_disk(self):
        """Test that index file is actually created on disk."""
        index = self._create_test_index()

        save_index(index)

        cache_path = get_index_path(self.test_file)
        assert cache_path.exists()

        # Verify it's valid JSON
        with open(cache_path) as f:
            data = json.load(f)
        assert 'version' in data
        assert data['source_path'] == self.test_file

    def test_load_nonexistent_index(self):
        """Test loading index that doesn't exist returns None."""
        loaded = load_index(self.test_file)
        assert loaded is None

    def test_index_invalidated_on_file_change(self):
        """Test that index is invalidated when file changes."""
        index = self._create_test_index()

        # Save index
        save_index(index)

        # Verify index loads
        loaded = load_index(self.test_file)
        assert loaded is not None

        # Wait a bit to ensure mtime changes
        time.sleep(0.1)

        # Modify file
        with open(self.test_file, 'a') as f:
            f.write('Line 4\n')

        # Index should now be invalid
        loaded = load_index(self.test_file)
        assert loaded is None

    def test_index_valid_when_file_unchanged(self):
        """Test that index is valid if file hasn't changed."""
        index = self._create_test_index()

        # Save index
        save_index(index)

        # Load multiple times - should keep returning cached data
        loaded1 = load_index(self.test_file)
        loaded2 = load_index(self.test_file)
        loaded3 = load_index(self.test_file)

        assert loaded1 is not None
        assert loaded2 is not None
        assert loaded3 is not None
        assert loaded1.source_path == loaded2.source_path == loaded3.source_path


class TestIndexDeletion:
    """Test index deletion."""

    def setup_method(self):
        """Create test file and index."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, 'test.txt')
        with open(self.test_file, 'w') as f:
            f.write('test content\n')

        # Create index
        stat = os.stat(self.test_file)
        index = UnifiedFileIndex(
            version=UNIFIED_INDEX_VERSION,
            source_path=self.test_file,
            source_modified_at=datetime.fromtimestamp(stat.st_mtime).isoformat(),
            source_size_bytes=stat.st_size,
            created_at=datetime.utcnow().isoformat(),
            build_time_seconds=0.1,
            file_type=FileType.TEXT,
            is_text=True,
            line_index=[[1, 0]],
        )
        save_index(index)

    def teardown_method(self):
        """Clean up."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        os.rmdir(self.temp_dir)
        delete_index(self.test_file)

    def test_delete_existing_index(self):
        """Test deleting existing index."""
        # Verify index exists
        assert load_index(self.test_file) is not None

        # Delete index
        deleted = delete_index(self.test_file)
        assert deleted is True

        # Verify index is gone
        assert load_index(self.test_file) is None

    def test_delete_nonexistent_index(self):
        """Test deleting index that doesn't exist."""
        # Delete first time
        delete_index(self.test_file)

        # Try to delete again
        deleted = delete_index(self.test_file)
        assert deleted is False


class TestClearAllIndexes:
    """Test clearing all indexes."""

    def setup_method(self):
        """Create multiple test files and indexes."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_files = []

        for i in range(3):
            filepath = os.path.join(self.temp_dir, f'file{i}.txt')
            with open(filepath, 'w') as f:
                f.write(f'File {i} content\n')
            self.test_files.append(filepath)

            # Create index for each
            stat = os.stat(filepath)
            index = UnifiedFileIndex(
                version=UNIFIED_INDEX_VERSION,
                source_path=filepath,
                source_modified_at=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                source_size_bytes=stat.st_size,
                created_at=datetime.utcnow().isoformat(),
                build_time_seconds=0.1,
                file_type=FileType.TEXT,
                is_text=True,
                line_index=[[1, 0]],
                line_count=1,
            )
            save_index(index)

    def teardown_method(self):
        """Clean up."""
        for filepath in self.test_files:
            if os.path.exists(filepath):
                os.remove(filepath)
            delete_index(filepath)
        os.rmdir(self.temp_dir)

    def test_clear_all_indexes(self):
        """Test clearing all indexes."""
        # Verify indexes exist
        for filepath in self.test_files:
            assert load_index(filepath) is not None

        # Clear all
        count = clear_all_indexes()
        assert count >= 3  # At least our 3 indexes

        # Verify our indexes are gone
        for filepath in self.test_files:
            assert load_index(filepath) is None


class TestIndexValidation:
    """Test index validation logic."""

    def setup_method(self):
        """Create test file."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, 'test.txt')
        with open(self.test_file, 'w') as f:
            f.write('original content\n')

    def teardown_method(self):
        """Clean up."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        os.rmdir(self.temp_dir)
        delete_index(self.test_file)

    def test_index_valid_when_file_unchanged(self):
        """Test index is valid when file hasn't changed."""
        stat = os.stat(self.test_file)

        index = UnifiedFileIndex(
            version=UNIFIED_INDEX_VERSION,
            source_path=self.test_file,
            source_modified_at=datetime.fromtimestamp(stat.st_mtime).isoformat(),
            source_size_bytes=stat.st_size,
            created_at=datetime.utcnow().isoformat(),
            build_time_seconds=0.1,
            file_type=FileType.TEXT,
            is_text=True,
            line_index=[[1, 0]],
        )

        assert is_index_valid(self.test_file, index) is True

    def test_index_invalid_when_size_changes(self):
        """Test index is invalid when file size changes."""
        stat = os.stat(self.test_file)

        index = UnifiedFileIndex(
            version=UNIFIED_INDEX_VERSION,
            source_path=self.test_file,
            source_modified_at=datetime.fromtimestamp(stat.st_mtime).isoformat(),
            source_size_bytes=stat.st_size,
            created_at=datetime.utcnow().isoformat(),
            build_time_seconds=0.1,
            file_type=FileType.TEXT,
            is_text=True,
            line_index=[[1, 0]],
        )

        # Modify file
        with open(self.test_file, 'a') as f:
            f.write('more content\n')

        assert is_index_valid(self.test_file, index) is False

    def test_index_invalid_when_file_deleted(self):
        """Test index is invalid when file is deleted."""
        stat = os.stat(self.test_file)

        index = UnifiedFileIndex(
            version=UNIFIED_INDEX_VERSION,
            source_path=self.test_file,
            source_modified_at=datetime.fromtimestamp(stat.st_mtime).isoformat(),
            source_size_bytes=stat.st_size,
            created_at=datetime.utcnow().isoformat(),
            build_time_seconds=0.1,
            file_type=FileType.TEXT,
            is_text=True,
            line_index=[[1, 0]],
        )

        # Delete file
        os.remove(self.test_file)

        assert is_index_valid(self.test_file, index) is False


class TestCachedLineCount:
    """Test cached line count helper."""

    def setup_method(self):
        """Create test file."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, 'test.txt')
        with open(self.test_file, 'w') as f:
            f.write('Line 1\nLine 2\nLine 3\n')

    def teardown_method(self):
        """Clean up."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        os.rmdir(self.temp_dir)
        delete_index(self.test_file)

    def test_get_cached_line_count_returns_none_without_cache(self):
        """Test that None is returned when no cache exists."""
        count = get_cached_line_count(self.test_file)
        assert count is None

    def test_get_cached_line_count_returns_count(self):
        """Test that line count is returned when cached."""
        stat = os.stat(self.test_file)
        index = UnifiedFileIndex(
            version=UNIFIED_INDEX_VERSION,
            source_path=self.test_file,
            source_modified_at=datetime.fromtimestamp(stat.st_mtime).isoformat(),
            source_size_bytes=stat.st_size,
            created_at=datetime.utcnow().isoformat(),
            build_time_seconds=0.1,
            file_type=FileType.TEXT,
            is_text=True,
            line_index=[[1, 0]],
            line_count=42,
            analysis_performed=True,
        )
        save_index(index)

        count = get_cached_line_count(self.test_file)
        assert count == 42


class TestShouldCreateIndex:
    """Test should_create_index logic."""

    def test_should_create_with_analyze_flag(self):
        """Test that index is always created with --analyze flag."""
        # Small file with analyze=True should create index
        assert should_create_index(100, analyze=True) is True
        # Large file with analyze=True should create index
        assert should_create_index(100 * 1024 * 1024, analyze=True) is True

    def test_should_not_create_small_file_without_analyze(self):
        """Test that small files don't get indexed without --analyze."""
        # 1KB file without analyze should not create index
        assert should_create_index(1024, analyze=False) is False
        # 10MB file without analyze should not create index
        assert should_create_index(10 * 1024 * 1024, analyze=False) is False

    def test_should_create_large_file_without_analyze(self):
        """Test that large files get indexed even without --analyze."""
        threshold = get_large_file_threshold_bytes()
        # File at threshold should create index
        assert should_create_index(threshold, analyze=False) is True
        # File above threshold should create index
        assert should_create_index(threshold + 1, analyze=False) is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
