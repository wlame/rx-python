"""Tests for analyse cache module."""

import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path

import pytest

from rx.analyse_cache import (
    AnalyseCacheData,
    clear_all_caches,
    delete_cache,
    get_analyse_cache_dir,
    get_cache_key,
    get_cache_path,
    is_cache_valid,
    load_cache,
    save_cache,
)


class TestCacheKeyGeneration:
    """Test cache key generation."""

    def test_cache_key_consistent(self):
        """Test that cache key is consistent for same path."""
        path = "/path/to/test/file.txt"
        key1 = get_cache_key(path)
        key2 = get_cache_key(path)
        assert key1 == key2

    def test_cache_key_different_for_different_paths(self):
        """Test that different paths get different keys."""
        key1 = get_cache_key("/path/to/file1.txt")
        key2 = get_cache_key("/path/to/file2.txt")
        assert key1 != key2

    def test_cache_key_includes_basename(self):
        """Test that cache key includes sanitized basename."""
        path = "/some/path/myfile.log"
        key = get_cache_key(path)
        assert "myfile.log" in key

    def test_cache_key_sanitizes_special_chars(self):
        """Test that special characters in filename are sanitized."""
        path = "/path/file with spaces & symbols!.txt"
        key = get_cache_key(path)
        # Should not contain spaces or special chars
        assert " " not in key
        assert "&" not in key
        assert "!" not in key


class TestCacheDirectory:
    """Test cache directory management."""

    def test_cache_dir_created(self):
        """Test that cache directory is created."""
        cache_dir = get_analyse_cache_dir()
        assert cache_dir.exists()
        assert cache_dir.is_dir()

    def test_cache_dir_location(self):
        """Test cache directory is in expected location."""
        cache_dir = get_analyse_cache_dir()
        assert "analyse_cache" in str(cache_dir)
        assert ".cache/rx" in str(cache_dir) or "cache/rx" in str(cache_dir)


class TestCacheSaveAndLoad:
    """Test saving and loading cache."""

    def setup_method(self):
        """Create test file before each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test.txt")
        with open(self.test_file, "w") as f:
            f.write("Line 1\nLine 2\nLine 3\n")

    def teardown_method(self):
        """Clean up after each test."""
        # Clean up test file
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        os.rmdir(self.temp_dir)

        # Clean up cache
        delete_cache(self.test_file)

    def test_save_and_load_cache(self):
        """Test saving and loading cache works."""
        analysis_result = {
            "file": "f1",
            "size_bytes": 24,
            "size_human": "24 B",
            "is_text": True,
            "line_count": 3,
            "empty_line_count": 0,
        }

        # Save cache
        saved = save_cache(self.test_file, analysis_result)
        assert saved is True

        # Load cache
        loaded = load_cache(self.test_file)
        assert loaded is not None
        assert loaded == analysis_result

    def test_cache_file_exists_on_disk(self):
        """Test that cache file is actually created on disk."""
        analysis_result = {"file": "f1", "is_text": True}

        save_cache(self.test_file, analysis_result)

        cache_path = get_cache_path(self.test_file)
        assert cache_path.exists()

        # Verify it's valid JSON
        with open(cache_path, 'r') as f:
            data = json.load(f)
        assert "analysis_result" in data
        assert data["analysis_result"] == analysis_result

    def test_load_nonexistent_cache(self):
        """Test loading cache that doesn't exist returns None."""
        loaded = load_cache(self.test_file)
        assert loaded is None

    def test_cache_invalidated_on_file_change(self):
        """Test that cache is invalidated when file changes."""
        analysis_result = {"file": "f1", "is_text": True}

        # Save cache
        save_cache(self.test_file, analysis_result)

        # Verify cache loads
        loaded = load_cache(self.test_file)
        assert loaded is not None

        # Wait a bit to ensure mtime changes
        time.sleep(0.1)

        # Modify file
        with open(self.test_file, "a") as f:
            f.write("Line 4\n")

        # Cache should now be invalid
        loaded = load_cache(self.test_file)
        assert loaded is None

    def test_cache_validation_with_same_mtime(self):
        """Test that cache is valid if mtime hasn't changed."""
        analysis_result = {"file": "f1", "is_text": True}

        # Save cache
        save_cache(self.test_file, analysis_result)

        # Load multiple times - should keep returning cached data
        loaded1 = load_cache(self.test_file)
        loaded2 = load_cache(self.test_file)
        loaded3 = load_cache(self.test_file)

        assert loaded1 is not None
        assert loaded2 is not None
        assert loaded3 is not None
        assert loaded1 == loaded2 == loaded3


class TestCacheDeletion:
    """Test cache deletion."""

    def setup_method(self):
        """Create test file and cache."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test.txt")
        with open(self.test_file, "w") as f:
            f.write("test content\n")

        # Create cache
        save_cache(self.test_file, {"file": "f1"})

    def teardown_method(self):
        """Clean up."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        os.rmdir(self.temp_dir)
        delete_cache(self.test_file)

    def test_delete_existing_cache(self):
        """Test deleting existing cache."""
        # Verify cache exists
        assert load_cache(self.test_file) is not None

        # Delete cache
        deleted = delete_cache(self.test_file)
        assert deleted is True

        # Verify cache is gone
        assert load_cache(self.test_file) is None

    def test_delete_nonexistent_cache(self):
        """Test deleting cache that doesn't exist."""
        # Delete first time
        delete_cache(self.test_file)

        # Try to delete again
        deleted = delete_cache(self.test_file)
        assert deleted is False


class TestClearAllCaches:
    """Test clearing all caches."""

    def setup_method(self):
        """Create multiple test files and caches."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_files = []

        for i in range(3):
            filepath = os.path.join(self.temp_dir, f"file{i}.txt")
            with open(filepath, "w") as f:
                f.write(f"File {i} content\n")
            self.test_files.append(filepath)

            # Create cache for each
            save_cache(filepath, {"file": f"f{i}", "line_count": 1})

    def teardown_method(self):
        """Clean up."""
        for filepath in self.test_files:
            if os.path.exists(filepath):
                os.remove(filepath)
            delete_cache(filepath)
        os.rmdir(self.temp_dir)

    def test_clear_all_caches(self):
        """Test clearing all caches."""
        # Verify caches exist
        for filepath in self.test_files:
            assert load_cache(filepath) is not None

        # Clear all
        count = clear_all_caches()
        assert count >= 3  # At least our 3 caches

        # Verify our caches are gone
        for filepath in self.test_files:
            assert load_cache(filepath) is None


class TestCacheDataModel:
    """Test AnalyseCacheData model."""

    def test_cache_data_creation(self):
        """Test creating cache data model."""
        cache_data = AnalyseCacheData(
            file_path="/path/to/file.txt",
            file_size=1234,
            file_mtime="2024-01-01T12:00:00",
            cached_at="2024-01-01T12:01:00",
            analysis_result={"file": "f1", "is_text": True},
        )

        assert cache_data.file_path == "/path/to/file.txt"
        assert cache_data.file_size == 1234
        assert cache_data.version == 1
        assert cache_data.analysis_result["is_text"] is True

    def test_cache_data_serialization(self):
        """Test serializing cache data to JSON."""
        cache_data = AnalyseCacheData(
            file_path="/path/to/file.txt",
            file_size=1234,
            file_mtime="2024-01-01T12:00:00",
            cached_at="2024-01-01T12:01:00",
            analysis_result={"file": "f1"},
        )

        # Convert to dict
        data_dict = cache_data.model_dump()

        # Serialize to JSON
        json_str = json.dumps(data_dict)

        # Deserialize
        loaded_dict = json.loads(json_str)
        loaded_cache = AnalyseCacheData(**loaded_dict)

        assert loaded_cache.file_path == cache_data.file_path
        assert loaded_cache.file_size == cache_data.file_size
        assert loaded_cache.analysis_result == cache_data.analysis_result


class TestCacheValidation:
    """Test cache validation logic."""

    def setup_method(self):
        """Create test file."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test.txt")
        with open(self.test_file, "w") as f:
            f.write("original content\n")

    def teardown_method(self):
        """Clean up."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        os.rmdir(self.temp_dir)
        delete_cache(self.test_file)

    def test_cache_valid_when_file_unchanged(self):
        """Test cache is valid when file hasn't changed."""
        stat = os.stat(self.test_file)

        cache_data = AnalyseCacheData(
            file_path=self.test_file,
            file_size=stat.st_size,
            file_mtime=datetime.fromtimestamp(stat.st_mtime).isoformat(),
            cached_at=datetime.utcnow().isoformat(),
            analysis_result={"file": "f1"},
        )

        assert is_cache_valid(self.test_file, cache_data) is True

    def test_cache_invalid_when_size_changes(self):
        """Test cache is invalid when file size changes."""
        stat = os.stat(self.test_file)

        cache_data = AnalyseCacheData(
            file_path=self.test_file,
            file_size=stat.st_size,
            file_mtime=datetime.fromtimestamp(stat.st_mtime).isoformat(),
            cached_at=datetime.utcnow().isoformat(),
            analysis_result={"file": "f1"},
        )

        # Modify file
        with open(self.test_file, "a") as f:
            f.write("more content\n")

        assert is_cache_valid(self.test_file, cache_data) is False

    def test_cache_invalid_when_file_deleted(self):
        """Test cache is invalid when file is deleted."""
        stat = os.stat(self.test_file)

        cache_data = AnalyseCacheData(
            file_path=self.test_file,
            file_size=stat.st_size,
            file_mtime=datetime.fromtimestamp(stat.st_mtime).isoformat(),
            cached_at=datetime.utcnow().isoformat(),
            analysis_result={"file": "f1"},
        )

        # Delete file
        os.remove(self.test_file)

        assert is_cache_valid(self.test_file, cache_data) is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
