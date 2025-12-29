"""Tests for seekable index module."""

import pytest

from rx.seekable_zstd import check_t2sz_available, create_seekable_zstd


# Check if zstandard module is available
try:
    import zstandard

    ZSTANDARD_AVAILABLE = True
except ImportError:
    ZSTANDARD_AVAILABLE = False

# Check if either t2sz or zstandard is available for compression
CAN_CREATE_SEEKABLE = check_t2sz_available() or ZSTANDARD_AVAILABLE

from rx.seekable_index import (
    FrameLineInfo,
    SeekableIndex,
    build_index,
    delete_index,
    find_frame_for_line,
    find_frames_for_lines,
    get_index_dir,
    get_index_info,
    get_index_path,
    get_or_build_index,
    is_index_valid,
    load_index,
    save_index,
)
from rx.unified_index import UNIFIED_INDEX_VERSION


class TestFrameLineInfo:
    """Test FrameLineInfo dataclass."""

    def test_creation(self):
        """Test creating FrameLineInfo object."""
        info = FrameLineInfo(
            index=0,
            compressed_offset=0,
            compressed_size=1000,
            decompressed_offset=0,
            decompressed_size=5000,
            first_line=1,
            last_line=100,
            line_count=100,
        )
        assert info.index == 0
        assert info.first_line == 1
        assert info.last_line == 100
        assert info.line_count == 100


class TestSeekableIndex:
    """Test SeekableIndex dataclass."""

    def test_to_dict(self):
        """Test converting SeekableIndex to dictionary."""
        index = SeekableIndex(
            version=1,
            source_zst_path='/test/file.zst',
            source_zst_modified_at='2025-01-01T00:00:00',
            source_zst_size_bytes=1000,
            decompressed_size_bytes=5000,
            total_lines=100,
            frame_count=2,
            frame_size_target=4096,
            created_at='2025-01-01T00:00:00',
        )
        d = index.to_dict()
        assert d['version'] == 1
        assert d['source_zst_path'] == '/test/file.zst'
        assert d['total_lines'] == 100

    def test_from_dict(self):
        """Test creating SeekableIndex from dictionary."""
        d = {
            'version': 1,
            'source_zst_path': '/test/file.zst',
            'source_zst_modified_at': '2025-01-01T00:00:00',
            'source_zst_size_bytes': 1000,
            'decompressed_size_bytes': 5000,
            'total_lines': 100,
            'frame_count': 2,
            'frame_size_target': 4096,
            'frames': [],
            'line_index': [],
            'created_at': '2025-01-01T00:00:00',
        }
        index = SeekableIndex.from_dict(d)
        assert index.version == 1
        assert index.source_zst_path == '/test/file.zst'
        assert index.total_lines == 100


class TestIndexPath:
    """Test index path generation."""

    def test_get_index_dir_creates_directory(self, tmp_path, monkeypatch):
        """Test that get_index_dir creates directory if needed."""
        cache_dir = tmp_path / 'cache'
        monkeypatch.setenv('XDG_CACHE_HOME', str(cache_dir))

        index_dir = get_index_dir()
        assert index_dir.exists()
        assert 'indexes' in str(index_dir)

    def test_get_index_path_unique_for_different_files(self, tmp_path):
        """Test that different files get different index paths."""
        path1 = get_index_path('/path/to/file1.zst')
        path2 = get_index_path('/path/to/file2.zst')
        assert path1 != path2

    def test_get_index_path_includes_filename(self, tmp_path):
        """Test that index path includes original filename."""
        path = get_index_path('/path/to/myfile.zst')
        assert 'myfile.zst' in str(path)


@pytest.mark.skipif(not CAN_CREATE_SEEKABLE, reason='Neither t2sz nor zstandard available')
class TestBuildIndex:
    """Test index building."""

    @pytest.fixture
    def seekable_zstd_file(self, tmp_path):
        """Create a seekable zstd file for testing."""
        input_file = tmp_path / 'input.txt'
        # Create content with known number of lines
        lines = [f'Line number {i}\n' for i in range(500)]
        content = ''.join(lines)
        input_file.write_text(content)

        output_file = tmp_path / 'output.zst'
        info = create_seekable_zstd(
            input_file,
            output_file,
            frame_size_bytes=2 * 1024,  # 2KB frames for multiple frames
            compression_level=1,
        )
        return output_file, len(lines), content

    def test_build_index_basic(self, seekable_zstd_file, tmp_path, monkeypatch):
        """Test building index for seekable zstd file."""
        # Use temp directory for cache
        monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'cache'))

        zst_file, expected_lines, _ = seekable_zstd_file

        index = build_index(zst_file)

        assert index.version == UNIFIED_INDEX_VERSION
        assert index.source_zst_path == str(zst_file.resolve())
        assert index.total_lines == expected_lines
        assert len(index.frames) == index.frame_count
        assert index.frame_count > 0

    def test_build_index_frame_line_ranges(self, seekable_zstd_file, tmp_path, monkeypatch):
        """Test that frame line ranges are correct."""
        monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'cache'))

        zst_file, expected_lines, _ = seekable_zstd_file
        index = build_index(zst_file)

        # First frame should start at line 1
        assert index.frames[0].first_line == 1

        # Frames should be contiguous
        for i in range(1, len(index.frames)):
            assert index.frames[i].first_line == index.frames[i - 1].last_line + 1

        # Last frame should end at total_lines
        assert index.frames[-1].last_line == expected_lines

    def test_build_index_invalid_file(self, tmp_path):
        """Test building index for invalid file raises error."""
        invalid_file = tmp_path / 'invalid.zst'
        invalid_file.write_bytes(b'not valid')

        with pytest.raises(ValueError):
            build_index(invalid_file)


class TestFindFrameForLine:
    """Test finding frames for line numbers."""

    @pytest.fixture
    def sample_index(self):
        """Create sample index with known frame structure."""
        frames = [
            FrameLineInfo(
                index=0,
                compressed_offset=0,
                compressed_size=100,
                decompressed_offset=0,
                decompressed_size=1000,
                first_line=1,
                last_line=50,
                line_count=50,
            ),
            FrameLineInfo(
                index=1,
                compressed_offset=100,
                compressed_size=100,
                decompressed_offset=1000,
                decompressed_size=1000,
                first_line=51,
                last_line=100,
                line_count=50,
            ),
            FrameLineInfo(
                index=2,
                compressed_offset=200,
                compressed_size=100,
                decompressed_offset=2000,
                decompressed_size=1000,
                first_line=101,
                last_line=150,
                line_count=50,
            ),
        ]
        return SeekableIndex(
            version=1,
            source_zst_path='/test.zst',
            source_zst_modified_at='2025-01-01',
            source_zst_size_bytes=300,
            decompressed_size_bytes=3000,
            total_lines=150,
            frame_count=3,
            frame_size_target=1000,
            frames=frames,
        )

    def test_find_frame_first_frame(self, sample_index):
        """Test finding frame for line in first frame."""
        assert find_frame_for_line(sample_index, 1) == 0
        assert find_frame_for_line(sample_index, 25) == 0
        assert find_frame_for_line(sample_index, 50) == 0

    def test_find_frame_middle_frame(self, sample_index):
        """Test finding frame for line in middle frame."""
        assert find_frame_for_line(sample_index, 51) == 1
        assert find_frame_for_line(sample_index, 75) == 1
        assert find_frame_for_line(sample_index, 100) == 1

    def test_find_frame_last_frame(self, sample_index):
        """Test finding frame for line in last frame."""
        assert find_frame_for_line(sample_index, 101) == 2
        assert find_frame_for_line(sample_index, 125) == 2
        assert find_frame_for_line(sample_index, 150) == 2

    def test_find_frame_out_of_range(self, sample_index):
        """Test finding frame for out-of-range line raises error."""
        with pytest.raises(ValueError):
            find_frame_for_line(sample_index, 0)
        with pytest.raises(ValueError):
            find_frame_for_line(sample_index, 151)


class TestFindFramesForLines:
    """Test finding frames for multiple lines."""

    @pytest.fixture
    def sample_index(self):
        """Create sample index."""
        frames = [
            FrameLineInfo(
                index=0,
                compressed_offset=0,
                compressed_size=100,
                decompressed_offset=0,
                decompressed_size=1000,
                first_line=1,
                last_line=50,
                line_count=50,
            ),
            FrameLineInfo(
                index=1,
                compressed_offset=100,
                compressed_size=100,
                decompressed_offset=1000,
                decompressed_size=1000,
                first_line=51,
                last_line=100,
                line_count=50,
            ),
        ]
        return SeekableIndex(
            version=1,
            source_zst_path='/test.zst',
            source_zst_modified_at='2025-01-01',
            source_zst_size_bytes=200,
            decompressed_size_bytes=2000,
            total_lines=100,
            frame_count=2,
            frame_size_target=1000,
            frames=frames,
        )

    def test_find_frames_single_frame(self, sample_index):
        """Test finding frames for lines in same frame."""
        result = find_frames_for_lines(sample_index, [10, 20, 30])
        assert result == {0: [10, 20, 30]}

    def test_find_frames_multiple_frames(self, sample_index):
        """Test finding frames for lines in different frames."""
        result = find_frames_for_lines(sample_index, [10, 60, 90])
        assert 0 in result
        assert 1 in result
        assert 10 in result[0]
        assert 60 in result[1]
        assert 90 in result[1]

    def test_find_frames_ignores_invalid_lines(self, sample_index):
        """Test that invalid line numbers are ignored."""
        result = find_frames_for_lines(sample_index, [0, 10, 200])
        assert result == {0: [10]}


@pytest.mark.skipif(not CAN_CREATE_SEEKABLE, reason='Neither t2sz nor zstandard available')
class TestIndexValidation:
    """Test index validation."""

    @pytest.fixture
    def valid_index_setup(self, tmp_path, monkeypatch):
        """Create a valid index for testing."""
        monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'cache'))

        input_file = tmp_path / 'input.txt'
        input_file.write_text('line\n' * 100)

        zst_file = tmp_path / 'test.zst'
        create_seekable_zstd(input_file, zst_file, frame_size_bytes=1024)

        index = build_index(zst_file)
        return zst_file, index

    def test_is_index_valid_true(self, valid_index_setup):
        """Test valid index returns True."""
        zst_file, _ = valid_index_setup
        assert is_index_valid(zst_file) is True

    def test_is_index_valid_false_no_index(self, tmp_path):
        """Test missing index returns False."""
        nonexistent = tmp_path / 'nonexistent.zst'
        assert is_index_valid(nonexistent) is False

    def test_get_or_build_index_existing(self, valid_index_setup):
        """Test get_or_build_index returns existing index."""
        zst_file, original_index = valid_index_setup

        # Get should return cached index
        index = get_or_build_index(zst_file)
        assert index.total_lines == original_index.total_lines


class TestIndexPersistence:
    """Test index save/load."""

    def test_save_and_load_index(self, tmp_path):
        """Test saving and loading index."""
        frames = [
            FrameLineInfo(
                index=0,
                compressed_offset=0,
                compressed_size=100,
                decompressed_offset=0,
                decompressed_size=1000,
                first_line=1,
                last_line=50,
                line_count=50,
            ),
        ]
        original = SeekableIndex(
            version=1,
            source_zst_path='/test.zst',
            source_zst_modified_at='2025-01-01',
            source_zst_size_bytes=100,
            decompressed_size_bytes=1000,
            total_lines=50,
            frame_count=1,
            frame_size_target=1000,
            frames=frames,
            created_at='2025-01-01',
        )

        index_path = tmp_path / 'test.idx.json'
        save_index(original, index_path)

        loaded = load_index(index_path)
        assert loaded is not None
        assert loaded.total_lines == original.total_lines
        assert loaded.frame_count == original.frame_count
        assert len(loaded.frames) == len(original.frames)

    def test_load_index_nonexistent(self, tmp_path):
        """Test loading nonexistent index returns None."""
        result = load_index(tmp_path / 'nonexistent.json')
        assert result is None


@pytest.mark.skipif(not CAN_CREATE_SEEKABLE, reason='Neither t2sz nor zstandard available')
class TestDeleteIndex:
    """Test index deletion."""

    def test_delete_existing_index(self, tmp_path, monkeypatch):
        """Test deleting existing index."""
        monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'cache'))

        # Create a zstd file and build its index
        input_file = tmp_path / 'input.txt'
        input_file.write_text('test\n' * 100)

        zst_file = tmp_path / 'test.zst'
        create_seekable_zstd(input_file, zst_file, frame_size_bytes=1024)
        build_index(zst_file)

        # Verify index exists
        assert is_index_valid(zst_file)

        # Delete and verify
        result = delete_index(zst_file)
        assert result is True
        assert is_index_valid(zst_file) is False

    def test_delete_nonexistent_index(self, tmp_path):
        """Test deleting nonexistent index returns True."""
        result = delete_index(tmp_path / 'nonexistent.zst')
        assert result is True


@pytest.mark.skipif(not CAN_CREATE_SEEKABLE, reason='Neither t2sz nor zstandard available')
class TestGetIndexInfo:
    """Test getting index info."""

    def test_get_index_info_existing(self, tmp_path, monkeypatch):
        """Test getting info for existing index."""
        monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'cache'))

        input_file = tmp_path / 'input.txt'
        input_file.write_text('test line\n' * 100)

        zst_file = tmp_path / 'test.zst'
        create_seekable_zstd(input_file, zst_file, frame_size_bytes=1024)
        build_index(zst_file)

        info = get_index_info(zst_file)
        assert info is not None
        assert 'total_lines' in info
        assert 'frame_count' in info
        assert info['total_lines'] == 100

    def test_get_index_info_nonexistent(self, tmp_path):
        """Test getting info for nonexistent index returns None."""
        info = get_index_info(tmp_path / 'nonexistent.zst')
        assert info is None
