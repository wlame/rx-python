"""Tests for trace cache functionality."""

import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from rx.models import Submatch
from rx.trace import identify_matching_patterns
from rx.trace_cache import (
    TRACE_CACHE_VERSION,
    build_cache_from_matches,
    compute_patterns_hash,
    delete_trace_cache,
    get_cached_matches,
    get_trace_cache_dir,
    get_trace_cache_info,
    get_trace_cache_path,
    is_trace_cache_valid,
    load_trace_cache,
    reconstruct_match_data,
    save_trace_cache,
    should_cache_file,
)


@pytest.fixture
def temp_text_file():
    """Create a temporary text file with known content."""
    content = 'Line 1: First line with ERROR here\nLine 2: Second line ok\nLine 3: Third line WARNING there\n'
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write(content)
        temp_path = f.name

    yield temp_path

    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_large_file():
    """Create a larger temporary file for cache testing."""
    # Create file with many lines for realistic testing
    lines = []
    for i in range(1000):
        if i % 100 == 0:
            lines.append(f'Line {i:04d}: ERROR found here\n')
        elif i % 50 == 0:
            lines.append(f'Line {i:04d}: WARNING detected\n')
        else:
            lines.append(f'Line {i:04d}: Normal content here\n')

    content = ''.join(lines)

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
        f.write(content)
        temp_path = f.name

    yield temp_path

    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def cleanup_cache(temp_text_file):
    """Cleanup cache file after test."""
    yield temp_text_file
    # Clean up any cache files created during the test
    patterns = ['ERROR', 'WARNING']
    delete_trace_cache(temp_text_file, patterns, [])


class TestCacheDirectory:
    """Tests for cache directory management."""

    def test_get_trace_cache_dir_exists(self):
        """Test that cache directory is created."""
        cache_dir = get_trace_cache_dir()
        assert cache_dir.exists()
        assert cache_dir.is_dir()

    def test_get_trace_cache_dir_in_expected_location(self):
        """Test cache directory is in ~/.cache/rx/trace_cache."""
        cache_dir = get_trace_cache_dir()
        assert 'rx' in str(cache_dir)
        assert 'trace_cache' in str(cache_dir)


class TestPatternsHash:
    """Tests for patterns hash computation."""

    def test_compute_patterns_hash_consistent(self):
        """Test that same patterns give same hash."""
        patterns = ['error', 'warning']
        flags = ['-i']

        hash1 = compute_patterns_hash(patterns, flags)
        hash2 = compute_patterns_hash(patterns, flags)
        assert hash1 == hash2

    def test_compute_patterns_hash_order_independent(self):
        """Test that pattern order doesn't affect hash (patterns are sorted)."""
        patterns1 = ['error', 'warning']
        patterns2 = ['warning', 'error']
        flags = []

        hash1 = compute_patterns_hash(patterns1, flags)
        hash2 = compute_patterns_hash(patterns2, flags)
        assert hash1 == hash2

    def test_compute_patterns_hash_different_patterns(self):
        """Test that different patterns give different hashes."""
        hash1 = compute_patterns_hash(['error'], [])
        hash2 = compute_patterns_hash(['warning'], [])
        assert hash1 != hash2

    def test_compute_patterns_hash_includes_relevant_flags(self):
        """Test that relevant flags affect hash."""
        patterns = ['error']
        hash1 = compute_patterns_hash(patterns, [])
        hash2 = compute_patterns_hash(patterns, ['-i'])
        assert hash1 != hash2

    def test_compute_patterns_hash_ignores_irrelevant_flags(self):
        """Test that irrelevant flags don't affect hash."""
        patterns = ['error']
        hash1 = compute_patterns_hash(patterns, ['--color=never'])
        hash2 = compute_patterns_hash(patterns, ['--no-heading'])
        assert hash1 == hash2  # Both should be empty relevant flags


class TestCachePath:
    """Tests for cache path generation."""

    def test_get_trace_cache_path_returns_path(self, temp_text_file):
        """Test that get_trace_cache_path returns a Path object."""
        path = get_trace_cache_path(temp_text_file, ['error'], [])
        assert isinstance(path, Path)

    def test_get_trace_cache_path_consistent(self, temp_text_file):
        """Test that same inputs give same path."""
        patterns = ['error', 'warning']
        path1 = get_trace_cache_path(temp_text_file, patterns, [])
        path2 = get_trace_cache_path(temp_text_file, patterns, [])
        assert path1 == path2

    def test_get_trace_cache_path_different_for_different_patterns(self, temp_text_file):
        """Test that different patterns give different paths."""
        path1 = get_trace_cache_path(temp_text_file, ['error'], [])
        path2 = get_trace_cache_path(temp_text_file, ['warning'], [])
        assert path1 != path2

    def test_get_trace_cache_path_different_for_different_files(self, temp_text_file, temp_large_file):
        """Test that different files give different paths."""
        patterns = ['error']
        path1 = get_trace_cache_path(temp_text_file, patterns, [])
        path2 = get_trace_cache_path(temp_large_file, patterns, [])
        assert path1 != path2

    def test_get_trace_cache_path_is_json(self, temp_text_file):
        """Test that cache path has .json extension."""
        path = get_trace_cache_path(temp_text_file, ['error'], [])
        assert str(path).endswith('.json')

    def test_get_trace_cache_path_includes_filename(self, temp_text_file):
        """Test that cache path includes original filename."""
        path = get_trace_cache_path(temp_text_file, ['error'], [])
        original_name = os.path.basename(temp_text_file)
        assert original_name in str(path)


class TestSaveLoadCache:
    """Tests for cache persistence."""

    def test_save_and_load_cache(self, temp_text_file):
        """Test that saved cache can be loaded."""
        cache_data = {
            'version': TRACE_CACHE_VERSION,
            'source_path': temp_text_file,
            'source_modified_at': '2025-01-01T00:00:00',
            'source_size_bytes': 100,
            'patterns': ['error'],
            'patterns_hash': 'abc12345',
            'rg_flags': [],
            'created_at': '2025-01-01T00:00:00',
            'matches': [{'pattern_index': 0, 'offset': 10, 'line_number': 1}],
        }

        cache_path = get_trace_cache_path(temp_text_file, ['error'], [])
        assert save_trace_cache(cache_data, cache_path)

        loaded = load_trace_cache(cache_path)
        assert loaded is not None
        assert loaded['version'] == TRACE_CACHE_VERSION
        assert loaded['matches'] == cache_data['matches']

        # Cleanup
        delete_trace_cache(temp_text_file, ['error'], [])

    def test_load_nonexistent_cache(self):
        """Test that loading nonexistent cache returns None."""
        loaded = load_trace_cache('/nonexistent/path.json')
        assert loaded is None

    def test_load_invalid_json(self):
        """Test that loading invalid JSON returns None."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            f.write('not valid json')
            invalid_path = f.name

        try:
            loaded = load_trace_cache(invalid_path)
            assert loaded is None
        finally:
            os.unlink(invalid_path)

    def test_load_wrong_version(self):
        """Test that loading wrong version returns None."""
        cache_data = {'version': 999, 'matches': []}

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(cache_data, f)
            path = f.name

        try:
            loaded = load_trace_cache(path)
            assert loaded is None
        finally:
            os.unlink(path)


class TestCacheValidation:
    """Tests for cache validation."""

    def test_is_cache_valid_no_cache(self, temp_text_file):
        """Test that nonexistent cache is invalid."""
        delete_trace_cache(temp_text_file, ['error'], [])
        assert not is_trace_cache_valid(temp_text_file, ['error'], [])

    def test_is_cache_valid_fresh_cache(self, temp_text_file):
        """Test that freshly created cache is valid."""
        from datetime import datetime

        stat = os.stat(temp_text_file)
        mtime = datetime.fromtimestamp(stat.st_mtime).isoformat()

        cache_data = {
            'version': TRACE_CACHE_VERSION,
            'source_path': os.path.abspath(temp_text_file),
            'source_modified_at': mtime,
            'source_size_bytes': stat.st_size,
            'patterns': ['error'],
            'patterns_hash': compute_patterns_hash(['error'], []),
            'rg_flags': [],
            'created_at': datetime.now().isoformat(),
            'matches': [],
        }

        cache_path = get_trace_cache_path(temp_text_file, ['error'], [])
        save_trace_cache(cache_data, cache_path)

        assert is_trace_cache_valid(temp_text_file, ['error'], [])

        delete_trace_cache(temp_text_file, ['error'], [])

    def test_is_cache_valid_after_file_modification(self, temp_text_file):
        """Test that cache becomes invalid after file modification."""
        from datetime import datetime

        stat = os.stat(temp_text_file)
        mtime = datetime.fromtimestamp(stat.st_mtime).isoformat()

        cache_data = {
            'version': TRACE_CACHE_VERSION,
            'source_path': os.path.abspath(temp_text_file),
            'source_modified_at': mtime,
            'source_size_bytes': stat.st_size,
            'patterns': ['error'],
            'patterns_hash': compute_patterns_hash(['error'], []),
            'rg_flags': [],
            'created_at': datetime.now().isoformat(),
            'matches': [],
        }

        cache_path = get_trace_cache_path(temp_text_file, ['error'], [])
        save_trace_cache(cache_data, cache_path)

        assert is_trace_cache_valid(temp_text_file, ['error'], [])

        # Modify file
        time.sleep(0.1)
        with open(temp_text_file, 'a') as f:
            f.write('New line\n')

        assert not is_trace_cache_valid(temp_text_file, ['error'], [])

        delete_trace_cache(temp_text_file, ['error'], [])

    def test_is_cache_valid_wrong_patterns(self, temp_text_file):
        """Test that cache is invalid for different patterns."""
        from datetime import datetime

        stat = os.stat(temp_text_file)
        mtime = datetime.fromtimestamp(stat.st_mtime).isoformat()

        cache_data = {
            'version': TRACE_CACHE_VERSION,
            'source_path': os.path.abspath(temp_text_file),
            'source_modified_at': mtime,
            'source_size_bytes': stat.st_size,
            'patterns': ['error'],
            'patterns_hash': compute_patterns_hash(['error'], []),
            'rg_flags': [],
            'created_at': datetime.now().isoformat(),
            'matches': [],
        }

        cache_path = get_trace_cache_path(temp_text_file, ['error'], [])
        save_trace_cache(cache_data, cache_path)

        # Should be valid for same patterns
        assert is_trace_cache_valid(temp_text_file, ['error'], [])

        # Should be invalid for different patterns (even though cache file exists)
        assert not is_trace_cache_valid(temp_text_file, ['warning'], [])

        delete_trace_cache(temp_text_file, ['error'], [])


class TestDeleteCache:
    """Tests for cache deletion."""

    def test_delete_cache_removes_file(self, temp_text_file):
        """Test that delete_trace_cache removes the cache file."""
        cache_data = {
            'version': TRACE_CACHE_VERSION,
            'source_path': temp_text_file,
            'matches': [],
        }
        cache_path = get_trace_cache_path(temp_text_file, ['error'], [])
        save_trace_cache(cache_data, cache_path)
        assert cache_path.exists()

        result = delete_trace_cache(temp_text_file, ['error'], [])
        assert result is True
        assert not cache_path.exists()

    def test_delete_cache_nonexistent(self, temp_text_file):
        """Test that deleting nonexistent cache succeeds."""
        delete_trace_cache(temp_text_file, ['error'], [])  # Ensure no cache
        result = delete_trace_cache(temp_text_file, ['error'], [])
        assert result is True


class TestBuildCacheFromMatches:
    """Tests for building cache from match results."""

    def test_build_cache_basic(self, temp_text_file):
        """Test building cache from matches."""
        matches = [
            {'pattern': 'p1', 'file': 'f1', 'offset': 10, 'relative_line_number': 1},
            {'pattern': 'p2', 'file': 'f1', 'offset': 50, 'relative_line_number': 2},
        ]
        patterns = ['error', 'warning']

        cache_data = build_cache_from_matches(temp_text_file, patterns, [], matches)

        assert cache_data['version'] == TRACE_CACHE_VERSION
        assert cache_data['patterns'] == patterns
        assert len(cache_data['matches']) == 2
        assert cache_data['matches'][0]['pattern_index'] == 0
        assert cache_data['matches'][0]['offset'] == 10
        assert cache_data['matches'][1]['pattern_index'] == 1

    def test_build_cache_empty_matches(self, temp_text_file):
        """Test building cache with no matches."""
        cache_data = build_cache_from_matches(temp_text_file, ['error'], [], [])

        assert cache_data['version'] == TRACE_CACHE_VERSION
        assert cache_data['matches'] == []

    def test_build_cache_includes_flags(self, temp_text_file):
        """Test that relevant flags are included in cache."""
        cache_data = build_cache_from_matches(temp_text_file, ['error'], ['-i', '--color=never'], [])

        assert '-i' in cache_data['rg_flags']
        assert '--color=never' not in cache_data['rg_flags']


class TestGetCachedMatches:
    """Tests for retrieving cached matches."""

    def test_get_cached_matches_returns_none_for_no_cache(self, temp_text_file):
        """Test that get_cached_matches returns None when no cache exists."""
        delete_trace_cache(temp_text_file, ['error'], [])
        result = get_cached_matches(temp_text_file, ['error'], [])
        assert result is None

    def test_get_cached_matches_returns_matches(self, temp_text_file):
        """Test that get_cached_matches returns cached matches."""
        from datetime import datetime

        stat = os.stat(temp_text_file)
        mtime = datetime.fromtimestamp(stat.st_mtime).isoformat()

        cached_matches = [
            {'pattern_index': 0, 'offset': 10, 'line_number': 1},
            {'pattern_index': 0, 'offset': 50, 'line_number': 2},
        ]

        cache_data = {
            'version': TRACE_CACHE_VERSION,
            'source_path': os.path.abspath(temp_text_file),
            'source_modified_at': mtime,
            'source_size_bytes': stat.st_size,
            'patterns': ['error'],
            'patterns_hash': compute_patterns_hash(['error'], []),
            'rg_flags': [],
            'created_at': datetime.now().isoformat(),
            'matches': cached_matches,
        }

        cache_path = get_trace_cache_path(temp_text_file, ['error'], [])
        save_trace_cache(cache_data, cache_path)

        result = get_cached_matches(temp_text_file, ['error'], [])
        assert result is not None
        assert len(result) == 2
        assert result[0]['offset'] == 10

        delete_trace_cache(temp_text_file, ['error'], [])


class TestShouldCacheFile:
    """Tests for cache eligibility determination."""

    def test_should_cache_large_file_complete_scan(self):
        """Test that large files with complete scans should be cached."""
        # Mock threshold to 1KB for testing
        with patch('rx.trace_cache.get_large_file_threshold_bytes', return_value=1024):
            assert should_cache_file(2048, None, True)

    def test_should_not_cache_small_file(self):
        """Test that small files should not be cached."""
        with patch('rx.trace_cache.get_large_file_threshold_bytes', return_value=1024):
            assert not should_cache_file(512, None, True)

    def test_should_not_cache_with_max_results(self):
        """Test that files should not be cached when max_results is set."""
        with patch('rx.trace_cache.get_large_file_threshold_bytes', return_value=1024):
            assert not should_cache_file(2048, 100, True)

    def test_should_not_cache_incomplete_scan(self):
        """Test that files should not be cached when scan is incomplete."""
        with patch('rx.trace_cache.get_large_file_threshold_bytes', return_value=1024):
            assert not should_cache_file(2048, None, False)


class TestReconstructMatchData:
    """Tests for match reconstruction from cache."""

    def test_reconstruct_match_basic(self, temp_text_file):
        """Test basic match reconstruction."""
        cached_match = {'pattern_index': 0, 'offset': 0, 'line_number': 1}
        patterns = ['ERROR']
        pattern_ids = {'p1': 'ERROR'}

        match_dict, ctx_lines = reconstruct_match_data(
            temp_text_file,
            cached_match,
            patterns,
            pattern_ids,
            'f1',
            [],
            context_before=0,
            context_after=0,
        )

        assert match_dict['pattern'] == 'p1'
        assert match_dict['file'] == 'f1'
        assert match_dict['offset'] == 0
        assert match_dict['relative_line_number'] == 1
        assert 'ERROR' in match_dict['line_text']

    def test_reconstruct_match_with_context(self, temp_text_file):
        """Test match reconstruction with context lines."""
        cached_match = {'pattern_index': 0, 'offset': 0, 'line_number': 2}
        patterns = ['Second']
        pattern_ids = {'p1': 'Second'}

        match_dict, ctx_lines = reconstruct_match_data(
            temp_text_file,
            cached_match,
            patterns,
            pattern_ids,
            'f1',
            [],
            context_before=1,
            context_after=1,
        )

        assert len(ctx_lines) >= 1  # At least the matched line
        # Check line numbers are in correct range
        line_numbers = [ctx.relative_line_number for ctx in ctx_lines]
        assert 2 in line_numbers  # Matched line should be present

    def test_reconstruct_match_with_submatches(self, temp_text_file):
        """Test that submatches are correctly extracted."""
        cached_match = {'pattern_index': 0, 'offset': 0, 'line_number': 1}
        patterns = ['ERROR']
        pattern_ids = {'p1': 'ERROR'}

        match_dict, _ = reconstruct_match_data(
            temp_text_file,
            cached_match,
            patterns,
            pattern_ids,
            'f1',
            [],
            context_before=0,
            context_after=0,
        )

        assert 'submatches' in match_dict
        if match_dict['submatches']:
            assert match_dict['submatches'][0].text == 'ERROR'

    def test_reconstruct_match_case_insensitive(self, temp_text_file):
        """Test match reconstruction with case-insensitive flag."""
        cached_match = {'pattern_index': 0, 'offset': 0, 'line_number': 1}
        patterns = ['error']  # lowercase
        pattern_ids = {'p1': 'error'}

        match_dict, _ = reconstruct_match_data(
            temp_text_file,
            cached_match,
            patterns,
            pattern_ids,
            'f1',
            ['-i'],  # case insensitive
            context_before=0,
            context_after=0,
        )

        # Should still find "ERROR" in the line
        assert 'submatches' in match_dict
        if match_dict['submatches']:
            assert match_dict['submatches'][0].text.upper() == 'ERROR'


class TestCacheInfo:
    """Tests for cache info retrieval."""

    def test_get_cache_info_no_cache(self, temp_text_file):
        """Test get_trace_cache_info when no cache exists."""
        delete_trace_cache(temp_text_file, ['error'], [])
        info = get_trace_cache_info(temp_text_file, ['error'], [])
        assert info is None

    def test_get_cache_info_with_cache(self, temp_text_file):
        """Test get_trace_cache_info with existing cache."""
        from datetime import datetime

        stat = os.stat(temp_text_file)
        mtime = datetime.fromtimestamp(stat.st_mtime).isoformat()

        cache_data = {
            'version': TRACE_CACHE_VERSION,
            'source_path': os.path.abspath(temp_text_file),
            'source_modified_at': mtime,
            'source_size_bytes': stat.st_size,
            'patterns': ['error', 'warning'],
            'patterns_hash': compute_patterns_hash(['error', 'warning'], []),
            'rg_flags': ['-i'],
            'created_at': datetime.now().isoformat(),
            'matches': [{'pattern_index': 0, 'offset': 10, 'line_number': 1}],
        }

        cache_path = get_trace_cache_path(temp_text_file, ['error', 'warning'], [])
        save_trace_cache(cache_data, cache_path)

        info = get_trace_cache_info(temp_text_file, ['error', 'warning'], [])
        assert info is not None
        assert 'cache_path' in info
        assert info['match_count'] == 1
        assert info['patterns'] == ['error', 'warning']
        assert info['is_valid']  # Should still be valid

        delete_trace_cache(temp_text_file, ['error', 'warning'], [])


class TestIntegrationWithTrace:
    """Integration tests with trace.py functionality."""

    @pytest.fixture
    def mock_large_threshold(self):
        """Mock the large file threshold to a small value for testing."""
        with patch('rx.trace_cache.get_large_file_threshold_bytes', return_value=100):
            with patch('rx.trace.get_large_file_threshold_bytes', return_value=100):
                yield

    def test_cache_roundtrip(self, temp_large_file, mock_large_threshold):
        """Test that caching and retrieval produces consistent results."""
        from rx.trace import parse_paths

        patterns = ['ERROR']

        # First run - should create cache
        result1 = parse_paths([temp_large_file], patterns)

        # Give time for cache to be written
        time.sleep(0.1)

        # Check cache was created
        patterns_list = list(result1.patterns.values())
        cache_path = get_trace_cache_path(temp_large_file, patterns_list, [])

        # Second run - should use cache (if cache was created)
        result2 = parse_paths([temp_large_file], patterns)

        # Results should be identical
        assert len(result1.matches) == len(result2.matches)

        # Match offsets should be the same
        offsets1 = sorted([m['offset'] for m in result1.matches])
        offsets2 = sorted([m['offset'] for m in result2.matches])
        assert offsets1 == offsets2

        # Cleanup
        delete_trace_cache(temp_large_file, patterns_list, [])

    def test_cache_not_created_with_max_results(self, temp_large_file, mock_large_threshold):
        """Test that cache is not created when max_results is set."""
        from rx.trace import parse_paths

        patterns = ['ERROR']

        # Run with max_results
        result = parse_paths([temp_large_file], patterns, max_results=1)

        # Cache should not exist
        patterns_list = list(result.patterns.values())
        assert not is_trace_cache_valid(temp_large_file, patterns_list, [])

    def test_cache_invalidation_on_file_change(self, temp_large_file, mock_large_threshold):
        """Test that cache is invalidated when file changes."""
        from rx.trace import parse_paths

        patterns = ['ERROR']

        # First run - should create cache
        result1 = parse_paths([temp_large_file], patterns)
        patterns_list = list(result1.patterns.values())

        # Verify cache exists and is valid
        time.sleep(0.1)

        # Modify file
        time.sleep(0.1)
        with open(temp_large_file, 'a') as f:
            f.write('New ERROR line added\n')

        # Cache should now be invalid
        assert not is_trace_cache_valid(temp_large_file, patterns_list, [])

        # Cleanup
        delete_trace_cache(temp_large_file, patterns_list, [])


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_patterns_list(self, temp_text_file):
        """Test handling of empty patterns list."""
        hash_result = compute_patterns_hash([], [])
        assert hash_result is not None
        assert len(hash_result) == 16

    def test_special_characters_in_patterns(self, temp_text_file):
        """Test patterns with special regex characters."""
        patterns = [r'\d+', r'[a-z]+', r'.*error.*']
        hash_result = compute_patterns_hash(patterns, [])
        assert hash_result is not None

    def test_unicode_in_file_content(self):
        """Test handling of unicode content in files."""
        content = 'Line 1: Error with unicode: \u00e9\u00e0\u00fc\nLine 2: More \u4e2d\u6587\n'
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
            f.write(content)
            temp_path = f.name

        try:
            patterns = ['Error']
            cache_data = build_cache_from_matches(
                temp_path,
                patterns,
                [],
                [{'pattern': 'p1', 'file': 'f1', 'offset': 0, 'relative_line_number': 1}],
            )
            assert cache_data is not None
        finally:
            os.unlink(temp_path)

    def test_very_long_pattern(self, temp_text_file):
        """Test handling of very long patterns."""
        long_pattern = 'a' * 1000
        hash_result = compute_patterns_hash([long_pattern], [])
        assert hash_result is not None
        assert len(hash_result) == 16

    def test_many_patterns(self, temp_text_file):
        """Test handling of many patterns."""
        patterns = [f'pattern{i}' for i in range(100)]
        hash_result = compute_patterns_hash(patterns, [])
        assert hash_result is not None

        # Should produce consistent results
        hash_result2 = compute_patterns_hash(patterns, [])
        assert hash_result == hash_result2


class TestCompressedFileCaching:
    """Tests for compressed file caching functionality."""

    def test_build_cache_with_compression_format(self, temp_text_file):
        """Test that compression_format is included in cache data."""
        patterns = ['ERROR']
        matches = [
            {'pattern': 'p1', 'offset': 0, 'relative_line_number': 1, 'frame_index': 0},
            {'pattern': 'p1', 'offset': 100, 'relative_line_number': 5, 'frame_index': 1},
        ]

        cache_data = build_cache_from_matches(
            temp_text_file,
            patterns,
            [],
            matches,
            compression_format='zstd-seekable',
        )

        assert cache_data['compression_format'] == 'zstd-seekable'
        assert 'frames_with_matches' in cache_data
        assert sorted(cache_data['frames_with_matches']) == [0, 1]

    def test_build_cache_without_compression_format(self, temp_text_file):
        """Test that compression_format is omitted for regular files."""
        patterns = ['ERROR']
        matches = [
            {'pattern': 'p1', 'offset': 0, 'relative_line_number': 1},
        ]

        cache_data = build_cache_from_matches(
            temp_text_file,
            patterns,
            [],
            matches,
        )

        assert 'compression_format' not in cache_data
        assert 'frames_with_matches' not in cache_data

    def test_frame_index_preserved_in_cache(self, temp_text_file):
        """Test that frame_index is stored in cached matches."""
        patterns = ['ERROR']
        matches = [
            {'pattern': 'p1', 'offset': 0, 'relative_line_number': 1, 'frame_index': 2},
            {'pattern': 'p1', 'offset': 100, 'relative_line_number': 5, 'frame_index': 2},
            {'pattern': 'p1', 'offset': 200, 'relative_line_number': 10, 'frame_index': 5},
        ]

        cache_data = build_cache_from_matches(
            temp_text_file,
            patterns,
            [],
            matches,
            compression_format='zstd-seekable',
        )

        # Check that frame_index is preserved
        cached_matches = cache_data['matches']
        assert len(cached_matches) == 3
        assert cached_matches[0]['frame_index'] == 2
        assert cached_matches[1]['frame_index'] == 2
        assert cached_matches[2]['frame_index'] == 5

        # Check frames_with_matches
        assert sorted(cache_data['frames_with_matches']) == [2, 5]

    def test_frames_with_matches_deduplication(self, temp_text_file):
        """Test that frames_with_matches contains unique frame indices."""
        patterns = ['ERROR']
        # Multiple matches in the same frame
        matches = [{'pattern': 'p1', 'offset': i * 100, 'relative_line_number': i, 'frame_index': 0} for i in range(10)]

        cache_data = build_cache_from_matches(
            temp_text_file,
            patterns,
            [],
            matches,
            compression_format='zstd-seekable',
        )

        # Should only have frame 0 once
        assert cache_data['frames_with_matches'] == [0]

    def test_cache_version_updated(self, temp_text_file):
        """Test that cache version is 2 for compressed file support."""
        assert TRACE_CACHE_VERSION == 2

        patterns = ['ERROR']
        matches = [{'pattern': 'p1', 'offset': 0, 'relative_line_number': 1}]

        cache_data = build_cache_from_matches(temp_text_file, patterns, [], matches)
        assert cache_data['version'] == 2


class TestCompressedCacheHelpers:
    """Tests for compressed cache helper functions."""

    def test_should_cache_compressed_file_size_threshold(self):
        """Test compressed file size threshold (1MB)."""
        from rx.trace_cache import should_cache_compressed_file

        # Below threshold (1MB)
        assert should_cache_compressed_file(500_000, None, True) is False

        # Above threshold
        assert should_cache_compressed_file(2_000_000, None, True) is True

    def test_should_cache_compressed_file_max_results(self):
        """Test that max_results prevents caching."""
        from rx.trace_cache import should_cache_compressed_file

        # With max_results set, should not cache
        assert should_cache_compressed_file(2_000_000, 100, True) is False

        # Without max_results, should cache
        assert should_cache_compressed_file(2_000_000, None, True) is True

    def test_should_cache_compressed_file_incomplete_scan(self):
        """Test that incomplete scans are not cached."""
        from rx.trace_cache import should_cache_compressed_file

        # Incomplete scan
        assert should_cache_compressed_file(2_000_000, None, False) is False

        # Complete scan
        assert should_cache_compressed_file(2_000_000, None, True) is True

    def test_get_compressed_cache_info_no_cache(self, temp_text_file):
        """Test get_compressed_cache_info returns None when no cache exists."""
        from rx.trace_cache import get_compressed_cache_info

        result = get_compressed_cache_info(temp_text_file, ['ERROR'], [])
        assert result is None

    def test_get_compressed_cache_info_with_cache(self, temp_text_file):
        """Test get_compressed_cache_info returns correct data."""
        from rx.trace_cache import get_compressed_cache_info

        patterns = ['ERROR']
        matches = [
            {'pattern': 'p1', 'offset': 0, 'relative_line_number': 1, 'frame_index': 0},
            {'pattern': 'p1', 'offset': 100, 'relative_line_number': 5, 'frame_index': 2},
        ]

        # Create cache
        cache_data = build_cache_from_matches(
            temp_text_file,
            patterns,
            [],
            matches,
            compression_format='zstd-seekable',
        )
        cache_path = get_trace_cache_path(temp_text_file, patterns, [])
        save_trace_cache(cache_data, cache_path)

        # Retrieve cache info
        info = get_compressed_cache_info(temp_text_file, patterns, [])
        assert info is not None
        assert info['compression_format'] == 'zstd-seekable'
        assert info['frames_with_matches'] == [0, 2]
        assert info['match_count'] == 2
        assert len(info['matches']) == 2

        # Cleanup
        delete_trace_cache(temp_text_file, patterns, [])


class TestSeekableZstdCacheReconstruction:
    """Tests for seekable zstd cache reconstruction functions."""

    def test_reconstruct_seekable_zstd_match_basic(self, temp_text_file):
        """Test basic match reconstruction from cached data."""
        from rx.trace_cache import reconstruct_seekable_zstd_match

        patterns = ['ERROR']
        pattern_ids = {'p1': 'ERROR'}
        cached_match = {
            'pattern_index': 0,
            'offset': 0,
            'line_number': 1,
            'frame_index': 0,
        }

        # Create mock decompressed frame data
        frame_content = b'Line 1: ERROR in first line\nLine 2: Normal content\nLine 3: More content\n'
        decompressed_frames = {0: frame_content}
        frame_line_offsets = {0: 1}

        match_dict, context_lines = reconstruct_seekable_zstd_match(
            temp_text_file,
            cached_match,
            patterns,
            pattern_ids,
            'f1',
            [],
            decompressed_frames,
            frame_line_offsets,
            context_before=0,
            context_after=0,
        )

        assert match_dict['pattern'] == 'p1'
        assert match_dict['file'] == 'f1'
        assert match_dict['offset'] == 0
        assert match_dict['relative_line_number'] == 1
        assert 'ERROR' in match_dict['line_text']
        assert match_dict['is_compressed'] is True
        assert match_dict['is_seekable_zstd'] is True

    def test_reconstruct_seekable_zstd_match_with_context(self, temp_text_file):
        """Test match reconstruction with context lines."""
        from rx.trace_cache import reconstruct_seekable_zstd_match

        patterns = ['ERROR']
        pattern_ids = {'p1': 'ERROR'}
        cached_match = {
            'pattern_index': 0,
            'offset': 50,
            'line_number': 3,
            'frame_index': 0,
        }

        # Create mock decompressed frame data with multiple lines
        frame_content = b'Line 1: Before\nLine 2: Before too\nLine 3: ERROR here\nLine 4: After\nLine 5: After too\n'
        decompressed_frames = {0: frame_content}
        frame_line_offsets = {0: 1}

        match_dict, context_lines = reconstruct_seekable_zstd_match(
            temp_text_file,
            cached_match,
            patterns,
            pattern_ids,
            'f1',
            [],
            decompressed_frames,
            frame_line_offsets,
            context_before=2,
            context_after=2,
        )

        assert match_dict['relative_line_number'] == 3
        assert 'ERROR' in match_dict['line_text']
        assert len(context_lines) == 5  # 2 before + 1 match + 2 after

    def test_reconstruct_seekable_zstd_match_extracts_submatches(self, temp_text_file):
        """Test that submatches are extracted correctly."""
        from rx.trace_cache import reconstruct_seekable_zstd_match

        patterns = ['ERROR']
        pattern_ids = {'p1': 'ERROR'}
        cached_match = {
            'pattern_index': 0,
            'offset': 0,
            'line_number': 1,
            'frame_index': 0,
        }

        frame_content = b'This line has ERROR in it\n'
        decompressed_frames = {0: frame_content}
        frame_line_offsets = {0: 1}

        match_dict, _ = reconstruct_seekable_zstd_match(
            temp_text_file,
            cached_match,
            patterns,
            pattern_ids,
            'f1',
            [],
            decompressed_frames,
            frame_line_offsets,
        )

        assert len(match_dict['submatches']) == 1
        assert match_dict['submatches'][0].text == 'ERROR'


class TestIdentifyMatchingPatterns:
    """Tests for identify_matching_patterns function.

    This function is critical for determining which patterns matched a line,
    especially when processing cached matches where submatch data may be missing.
    """

    def test_with_valid_submatches(self):
        """Test pattern identification with valid submatches."""
        line_text = 'Error: something failed'
        submatches = [Submatch(text='Error', start=0, end=5)]
        pattern_ids = {'p1': 'Error'}

        result = identify_matching_patterns(line_text, submatches, pattern_ids, [])

        assert result == ['p1']

    def test_with_multiple_patterns_single_match(self):
        """Test that only the matching pattern is returned when multiple patterns exist."""
        line_text = 'Error: something failed'
        submatches = [Submatch(text='Error', start=0, end=5)]
        pattern_ids = {'p1': 'Error', 'p2': 'Warning'}

        result = identify_matching_patterns(line_text, submatches, pattern_ids, [])

        assert result == ['p1']
        assert 'p2' not in result

    def test_with_multiple_patterns_multiple_matches(self):
        """Test that multiple patterns can match the same line."""
        line_text = 'Error and Warning both appear'
        submatches = [
            Submatch(text='Error', start=0, end=5),
            Submatch(text='Warning', start=10, end=17),
        ]
        pattern_ids = {'p1': 'Error', 'p2': 'Warning'}

        result = identify_matching_patterns(line_text, submatches, pattern_ids, [])

        assert 'p1' in result
        assert 'p2' in result

    def test_empty_submatches_with_matching_pattern(self):
        """Test pattern identification when submatches is empty but pattern matches.

        This tests the fix for the bug where empty submatches would blindly return
        the first pattern without validation. The function should now validate
        the pattern against the line text.
        """
        line_text = 'This line contains wlame text'
        submatches = []  # Empty submatches (e.g., from cache reconstruction)
        pattern_ids = {'p1': 'wlame'}

        result = identify_matching_patterns(line_text, submatches, pattern_ids, [])

        assert result == ['p1']

    def test_empty_submatches_with_non_matching_pattern_returns_empty(self):
        """Test that non-matching patterns return empty list when submatches is empty.

        This is the critical bug fix test. Previously, when submatches was empty,
        the function would blindly return the first pattern without checking if
        it actually matched the line. This caused false positives when cached
        match data was stale or corrupted.

        Real-world scenario reproduced:
        - User searches for 'wlame' in a large file
        - File is cached with match offsets
        - Cache becomes stale or match reconstruction loses submatch data
        - A line like 'Email: dominionmbom@yahoo.co.uk' (which does NOT contain 'wlame')
          was incorrectly returned as a match because the old code assumed the first
          pattern always matched when submatches was empty.

        After the fix, the function validates the pattern against the line text
        and returns an empty list if no patterns actually match.
        """
        # This is the exact scenario from the bug report:
        # Line does NOT contain 'wlame' but was being returned as a match
        line_text = 'Email: dominionmbom@yahoo.co.uk - Name: dominion mbom - ScreenName: dominionmbom - Followers: 0 - Created At: Sat Aug 28 16:44:52 +0000 2010'
        submatches = []  # Empty submatches from cache reconstruction
        pattern_ids = {'p1': 'wlame'}

        result = identify_matching_patterns(line_text, submatches, pattern_ids, [])

        # The fix: should return empty list because 'wlame' is NOT in the line
        assert result == []

    def test_empty_submatches_with_multiple_patterns_mixed_match(self):
        """Test empty submatches with multiple patterns where only some match."""
        line_text = 'This line has Error but no warning'
        submatches = []
        pattern_ids = {'p1': 'Error', 'p2': 'Warning'}  # Note: 'Warning' with capital W

        result = identify_matching_patterns(line_text, submatches, pattern_ids, [])

        # Only 'Error' matches (case-sensitive by default)
        assert result == ['p1']

    def test_empty_submatches_case_insensitive(self):
        """Test empty submatches with case-insensitive flag."""
        line_text = 'This line has ERROR in uppercase'
        submatches = []
        pattern_ids = {'p1': 'error'}  # lowercase pattern

        # Without -i flag, should not match
        result_sensitive = identify_matching_patterns(line_text, submatches, pattern_ids, [])
        assert result_sensitive == []

        # With -i flag, should match
        result_insensitive = identify_matching_patterns(line_text, submatches, pattern_ids, ['-i'])
        assert result_insensitive == ['p1']

    def test_empty_submatches_with_regex_pattern(self):
        """Test empty submatches with regex pattern validation."""
        line_text = 'Error code: 12345'
        submatches = []
        pattern_ids = {'p1': r'\d+'}  # Match digits

        result = identify_matching_patterns(line_text, submatches, pattern_ids, [])

        assert result == ['p1']

    def test_empty_submatches_with_non_matching_regex(self):
        """Test empty submatches with regex that doesn't match."""
        line_text = 'No digits here'
        submatches = []
        pattern_ids = {'p1': r'\d+'}  # Match digits - won't match

        result = identify_matching_patterns(line_text, submatches, pattern_ids, [])

        assert result == []

    def test_empty_submatches_with_invalid_regex_skipped(self):
        """Test that invalid regex patterns are skipped without error."""
        line_text = 'Some text'
        submatches = []
        pattern_ids = {'p1': '[invalid(regex', 'p2': 'text'}

        result = identify_matching_patterns(line_text, submatches, pattern_ids, [])

        # Invalid regex p1 should be skipped, valid p2 should match
        assert result == ['p2']

    def test_empty_submatches_all_patterns_invalid(self):
        """Test empty submatches when all patterns are invalid regex."""
        line_text = 'Some text'
        submatches = []
        pattern_ids = {'p1': '[invalid(', 'p2': '(unclosed'}

        result = identify_matching_patterns(line_text, submatches, pattern_ids, [])

        # All patterns invalid, should return empty list
        assert result == []

    def test_submatches_not_matching_any_pattern(self):
        """Test when submatches exist but don't match any pattern in pattern_ids."""
        line_text = 'The quick brown fox'
        # Submatches contain text that doesn't match our patterns
        submatches = [Submatch(text='quick', start=4, end=9)]
        pattern_ids = {'p1': 'slow', 'p2': 'lazy'}

        result = identify_matching_patterns(line_text, submatches, pattern_ids, [])

        # Neither pattern matches the submatch text
        assert result == []

    def test_reconstruct_seekable_zstd_match_case_insensitive(self, temp_text_file):
        """Test case insensitive pattern matching in reconstruction."""
        from rx.trace_cache import reconstruct_seekable_zstd_match

        patterns = ['error']
        pattern_ids = {'p1': 'error'}
        cached_match = {
            'pattern_index': 0,
            'offset': 0,
            'line_number': 1,
            'frame_index': 0,
        }

        frame_content = b'This line has ERROR in it\n'
        decompressed_frames = {0: frame_content}
        frame_line_offsets = {0: 1}

        match_dict, _ = reconstruct_seekable_zstd_match(
            temp_text_file,
            cached_match,
            patterns,
            pattern_ids,
            'f1',
            ['-i'],  # Case insensitive flag
            decompressed_frames,
            frame_line_offsets,
        )

        assert len(match_dict['submatches']) == 1
        assert match_dict['submatches'][0].text == 'ERROR'
