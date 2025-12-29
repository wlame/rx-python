"""Tests for compressed file index management."""

import gzip
import os
import tempfile
from pathlib import Path

import pytest

from rx.compressed_index import (
    build_compressed_index,
    find_nearest_checkpoint,
    get_compressed_index_dir,
    get_compressed_index_path,
    get_decompressed_content_at_line,
    get_decompressed_lines,
)
from rx.unified_index import UNIFIED_INDEX_VERSION


@pytest.fixture
def temp_gzip_file():
    """Create a temporary gzip file with known content."""
    lines = [f'Line {i:04d}: Content for line {i}\n' for i in range(1, 101)]
    content = ''.join(lines).encode('utf-8')

    with tempfile.NamedTemporaryFile(delete=False, suffix='.gz') as f:
        temp_path = f.name

    with gzip.open(temp_path, 'wb') as gz:
        gz.write(content)

    yield temp_path, content, lines

    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_gzip_large():
    """Create a larger temporary gzip file for checkpoint testing."""
    lines = [f'Line {i:05d}: Content for line number {i}\n' for i in range(1, 5001)]
    content = ''.join(lines).encode('utf-8')

    with tempfile.NamedTemporaryFile(delete=False, suffix='.gz') as f:
        temp_path = f.name

    with gzip.open(temp_path, 'wb') as gz:
        gz.write(content)

    yield temp_path, content, lines

    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_index_dir(tmp_path, monkeypatch):
    """Create a temporary index directory using RX_CACHE_DIR env var."""
    monkeypatch.setenv('RX_CACHE_DIR', str(tmp_path))
    yield tmp_path / 'rx' / 'indexes'


class TestIndexPaths:
    """Tests for index path generation."""

    def test_get_compressed_index_dir_creates_directory(self, temp_index_dir):
        """Test that index directory is created."""
        index_dir = get_compressed_index_dir()
        assert index_dir.exists()
        assert index_dir.is_dir()

    def test_get_compressed_index_path_format(self, temp_gzip_file, temp_index_dir):
        """Test index path format."""
        temp_path, _, _ = temp_gzip_file
        index_path = get_compressed_index_path(temp_path)
        assert index_path.suffix == '.json'
        assert temp_path.split('/')[-1] in str(index_path)

    def test_get_compressed_index_path_consistent(self, temp_gzip_file, temp_index_dir):
        """Test that index path is consistent for same source."""
        temp_path, _, _ = temp_gzip_file
        path1 = get_compressed_index_path(temp_path)
        path2 = get_compressed_index_path(temp_path)
        assert path1 == path2


class TestIndexBuild:
    """Tests for building compressed file indexes."""

    def test_build_compressed_index(self, temp_gzip_file, temp_index_dir):
        """Test building an index for a compressed file."""
        temp_path, content, lines = temp_gzip_file
        index_data = build_compressed_index(temp_path)

        assert index_data['version'] == UNIFIED_INDEX_VERSION
        assert index_data['source_path'] == str(Path(temp_path).resolve())
        assert index_data['compression_format'] == 'gzip'
        assert index_data['decompressed_size_bytes'] == len(content)
        # Line count may be off by one due to final newline counting
        assert abs(index_data['total_lines'] - len(lines)) <= 1
        assert 'line_index' in index_data
        assert len(index_data['line_index']) > 0

    def test_build_index_has_first_line_checkpoint(self, temp_gzip_file, temp_index_dir):
        """Test that index always has checkpoint for line 1."""
        temp_path, _, _ = temp_gzip_file
        index_data = build_compressed_index(temp_path)
        assert index_data['line_index'][0] == [1, 0]

    def test_build_index_noncompressed_raises(self, temp_index_dir):
        """Test building index for non-compressed file raises ValueError."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as f:
            f.write(b'Plain text content\n')
            temp_path = f.name

        try:
            with pytest.raises(ValueError):
                build_compressed_index(temp_path)
        finally:
            os.unlink(temp_path)


class TestFindNearestCheckpoint:
    """Tests for checkpoint lookup."""

    def test_find_checkpoint_at_start(self):
        """Test finding checkpoint at start of file."""
        line_index = [[1, 0], [1000, 50000], [2000, 100000]]
        line, offset = find_nearest_checkpoint(line_index, 1)
        assert line == 1
        assert offset == 0

    def test_find_checkpoint_in_middle(self):
        """Test finding checkpoint in middle of file."""
        line_index = [[1, 0], [1000, 50000], [2000, 100000]]
        line, offset = find_nearest_checkpoint(line_index, 1500)
        assert line == 1000
        assert offset == 50000

    def test_find_checkpoint_at_exact_boundary(self):
        """Test finding checkpoint at exact checkpoint line."""
        line_index = [[1, 0], [1000, 50000], [2000, 100000]]
        line, offset = find_nearest_checkpoint(line_index, 2000)
        assert line == 2000
        assert offset == 100000

    def test_find_checkpoint_past_end(self):
        """Test finding checkpoint past last checkpoint."""
        line_index = [[1, 0], [1000, 50000], [2000, 100000]]
        line, offset = find_nearest_checkpoint(line_index, 5000)
        assert line == 2000
        assert offset == 100000


class TestGetDecompressedLines:
    """Tests for retrieving specific lines from compressed files."""

    def test_get_first_line(self, temp_gzip_file, temp_index_dir):
        """Test getting the first line."""
        temp_path, _, lines = temp_gzip_file
        result = get_decompressed_lines(temp_path, 1, 1)
        assert len(result) == 1
        assert result[0] == lines[0].rstrip('\n')

    def test_get_multiple_lines(self, temp_gzip_file, temp_index_dir):
        """Test getting multiple lines."""
        temp_path, _, lines = temp_gzip_file
        result = get_decompressed_lines(temp_path, 10, 5)
        assert len(result) == 5
        for i, line in enumerate(result):
            assert line == lines[9 + i].rstrip('\n')

    def test_get_lines_near_end(self, temp_gzip_file, temp_index_dir):
        """Test getting lines near end of file."""
        temp_path, _, lines = temp_gzip_file
        result = get_decompressed_lines(temp_path, 98, 3)
        assert len(result) == 3
        assert result[0] == lines[97].rstrip('\n')


class TestGetDecompressedContentAtLine:
    """Tests for getting content with context around a line."""

    def test_get_content_with_context(self, temp_gzip_file, temp_index_dir):
        """Test getting content with context lines."""
        temp_path, _, lines = temp_gzip_file
        result = get_decompressed_content_at_line(temp_path, 50, context_before=2, context_after=2)
        assert len(result) == 5  # 2 before + 1 target + 2 after
        assert result[2] == lines[49].rstrip('\n')  # Line 50 is at index 49

    def test_get_content_at_start(self, temp_gzip_file, temp_index_dir):
        """Test getting content at start of file."""
        temp_path, _, lines = temp_gzip_file
        result = get_decompressed_content_at_line(temp_path, 1, context_before=2, context_after=2)
        # Can't have 2 lines before line 1
        assert len(result) == 3  # Line 1, 2, 3
        assert result[0] == lines[0].rstrip('\n')

    def test_get_content_with_no_context(self, temp_gzip_file, temp_index_dir):
        """Test getting content with no context."""
        temp_path, _, lines = temp_gzip_file
        result = get_decompressed_content_at_line(temp_path, 50, context_before=0, context_after=0)
        assert len(result) == 1
        assert result[0] == lines[49].rstrip('\n')
