"""Tests for FileIndex and related Pydantic models.

These tests verify the serialization/deserialization of the unified FileIndex model
that replaces the previous dict-based approach for index data.
"""

import json
import os
import tempfile

import pytest

from rx.index import (
    INDEX_VERSION,
    create_index_file,
    delete_index,
    get_index_path,
    load_index,
)
from rx.models import (
    FileAnalysis,
    FileIndex,
    FrameLineInfo,
    IndexAnalysis,
    IndexType,
)


class TestIndexType:
    """Tests for IndexType enum."""

    def test_index_type_values(self):
        """Test IndexType enum has expected values."""
        assert IndexType.REGULAR.value == 'regular'
        assert IndexType.COMPRESSED.value == 'compressed'
        assert IndexType.SEEKABLE_ZSTD.value == 'seekable_zstd'

    def test_index_type_from_string(self):
        """Test IndexType can be created from string values."""
        assert IndexType('regular') == IndexType.REGULAR
        assert IndexType('compressed') == IndexType.COMPRESSED
        assert IndexType('seekable_zstd') == IndexType.SEEKABLE_ZSTD


class TestIndexAnalysis:
    """Tests for IndexAnalysis model."""

    def test_create_with_all_fields(self):
        """Test creating IndexAnalysis with all fields."""
        analysis = IndexAnalysis(
            line_count=100,
            empty_line_count=5,
            line_length_max=200,
            line_length_avg=50.5,
            line_length_median=45.0,
            line_length_p95=150.0,
            line_length_p99=180.0,
            line_length_stddev=25.0,
            line_length_max_line_number=42,
            line_length_max_byte_offset=1024,
            line_ending='LF',
        )

        assert analysis.line_count == 100
        assert analysis.empty_line_count == 5
        assert analysis.line_length_max == 200
        assert analysis.line_length_avg == 50.5
        assert analysis.line_ending == 'LF'

    def test_create_with_defaults(self):
        """Test creating IndexAnalysis with defaults (all None)."""
        analysis = IndexAnalysis()

        assert analysis.line_count is None
        assert analysis.empty_line_count is None
        assert analysis.line_length_max is None

    def test_model_dump_excludes_none(self):
        """Test that model_dump with exclude_none works."""
        analysis = IndexAnalysis(line_count=100, line_ending='LF')

        dumped = analysis.model_dump(exclude_none=True)

        assert 'line_count' in dumped
        assert 'line_ending' in dumped
        assert 'line_length_max' not in dumped
        assert 'empty_line_count' not in dumped

    def test_json_serialization_round_trip(self):
        """Test IndexAnalysis JSON serialization and deserialization."""
        original = IndexAnalysis(
            line_count=1000,
            empty_line_count=50,
            line_length_max=500,
            line_length_avg=75.5,
            line_ending='CRLF',
        )

        # Serialize to JSON
        json_str = original.model_dump_json()

        # Deserialize
        loaded = IndexAnalysis.model_validate_json(json_str)

        assert loaded.line_count == original.line_count
        assert loaded.empty_line_count == original.empty_line_count
        assert loaded.line_length_avg == original.line_length_avg
        assert loaded.line_ending == original.line_ending


class TestFileIndex:
    """Tests for FileIndex model."""

    def test_create_regular_index(self):
        """Test creating a regular file index."""
        index = FileIndex(
            version=INDEX_VERSION,
            index_type=IndexType.REGULAR,
            source_path='/path/to/file.txt',
            source_modified_at='2024-01-01T12:00:00',
            source_size_bytes=1024,
            line_index=[[1, 0], [100, 5000], [200, 10000]],
        )

        assert index.version == INDEX_VERSION
        assert index.index_type == IndexType.REGULAR
        assert index.source_path == '/path/to/file.txt'
        assert len(index.line_index) == 3

    def test_create_with_analysis(self):
        """Test creating index with embedded analysis."""
        analysis = IndexAnalysis(
            line_count=200,
            line_length_max=150,
            line_ending='LF',
        )

        index = FileIndex(
            version=INDEX_VERSION,
            source_path='/path/to/file.txt',
            source_modified_at='2024-01-01T12:00:00',
            source_size_bytes=2048,
            analysis=analysis,
            line_index=[[1, 0]],
        )

        assert index.analysis is not None
        assert index.analysis.line_count == 200
        assert index.analysis.line_ending == 'LF'

    def test_default_index_type_is_regular(self):
        """Test that default index_type is REGULAR."""
        index = FileIndex(
            version=1,
            source_path='/path/to/file.txt',
            source_modified_at='2024-01-01T12:00:00',
            source_size_bytes=100,
        )

        assert index.index_type == IndexType.REGULAR

    def test_get_line_count_from_total_lines(self):
        """Test get_line_count returns total_lines when set."""
        index = FileIndex(
            version=1,
            source_path='/path/to/file.txt',
            source_modified_at='2024-01-01T12:00:00',
            source_size_bytes=100,
            total_lines=500,
        )

        assert index.get_line_count() == 500

    def test_get_line_count_from_analysis(self):
        """Test get_line_count returns analysis.line_count when total_lines not set."""
        analysis = IndexAnalysis(line_count=300)
        index = FileIndex(
            version=1,
            source_path='/path/to/file.txt',
            source_modified_at='2024-01-01T12:00:00',
            source_size_bytes=100,
            analysis=analysis,
        )

        assert index.get_line_count() == 300

    def test_get_line_count_returns_none_when_unavailable(self):
        """Test get_line_count returns None when no line count available."""
        index = FileIndex(
            version=1,
            source_path='/path/to/file.txt',
            source_modified_at='2024-01-01T12:00:00',
            source_size_bytes=100,
        )

        assert index.get_line_count() is None


class TestFileIndexFromDict:
    """Tests for FileIndex.from_dict() method."""

    def test_from_dict_regular_index(self):
        """Test creating FileIndex from regular index dict."""
        data = {
            'version': 1,
            'source_path': '/path/to/file.txt',
            'source_modified_at': '2024-01-01T12:00:00',
            'source_size_bytes': 1024,
            'index_step_bytes': 2097152,
            'created_at': '2024-01-01T12:01:00',
            'line_index': [[1, 0], [50, 2500]],
            'analysis': {
                'line_count': 100,
                'empty_line_count': 5,
                'line_length_max': 200,
                'line_ending': 'LF',
            },
        }

        index = FileIndex.from_dict(data)

        assert index.version == 1
        assert index.index_type == IndexType.REGULAR
        assert index.source_path == '/path/to/file.txt'
        assert index.source_size_bytes == 1024
        assert len(index.line_index) == 2
        assert index.analysis is not None
        assert index.analysis.line_count == 100
        assert index.analysis.line_ending == 'LF'

    def test_from_dict_compressed_index(self):
        """Test creating FileIndex from compressed index dict."""
        data = {
            'version': 1,
            'source_path': '/path/to/file.txt.gz',
            'source_modified_at': '2024-01-01T12:00:00',
            'source_size_bytes': 512,
            'compression_format': 'gzip',
            'decompressed_size_bytes': 2048,
            'total_lines': 100,
            'line_sample_interval': 1000,
            'line_index': [[1, 0], [1000, 50000]],
            'created_at': '2024-01-01T12:01:00',
        }

        index = FileIndex.from_dict(data)

        assert index.index_type == IndexType.COMPRESSED
        assert index.compression_format == 'gzip'
        assert index.decompressed_size_bytes == 2048
        assert index.total_lines == 100
        assert index.get_line_count() == 100

    def test_from_dict_seekable_zstd_index(self):
        """Test creating FileIndex from seekable zstd index dict."""
        data = {
            'version': 1,
            'source_zst_path': '/path/to/file.zst',
            'source_zst_modified_at': '2024-01-01T12:00:00',
            'source_zst_size_bytes': 1024,
            'decompressed_size_bytes': 4096,
            'total_lines': 200,
            'frame_count': 4,
            'frames': [
                {
                    'index': 0,
                    'compressed_offset': 0,
                    'compressed_size': 256,
                    'decompressed_offset': 0,
                    'decompressed_size': 1024,
                    'first_line': 1,
                    'last_line': 50,
                    'line_count': 50,
                },
                {
                    'index': 1,
                    'compressed_offset': 256,
                    'compressed_size': 256,
                    'decompressed_offset': 1024,
                    'decompressed_size': 1024,
                    'first_line': 51,
                    'last_line': 100,
                    'line_count': 50,
                },
            ],
            'line_index': [[1, 0, 0], [51, 1024, 1]],
            'created_at': '2024-01-01T12:01:00',
        }

        index = FileIndex.from_dict(data)

        assert index.index_type == IndexType.SEEKABLE_ZSTD
        assert index.source_path == '/path/to/file.zst'
        assert index.frame_count == 4
        assert index.frames is not None
        assert len(index.frames) == 2
        assert index.frames[0].first_line == 1
        assert index.frames[1].first_line == 51

    def test_from_dict_with_missing_optional_fields(self):
        """Test from_dict handles missing optional fields gracefully."""
        data = {
            'version': 1,
            'source_path': '/path/to/file.txt',
            'source_modified_at': '2024-01-01T12:00:00',
            'source_size_bytes': 100,
        }

        index = FileIndex.from_dict(data)

        assert index.analysis is None
        assert index.compression_format is None
        assert index.frames is None
        assert index.line_index == []


class TestFileIndexToDict:
    """Tests for FileIndex.to_dict() method."""

    def test_to_dict_regular_index(self):
        """Test converting regular FileIndex to dict."""
        analysis = IndexAnalysis(
            line_count=100,
            line_length_max=200,
            line_ending='LF',
        )

        index = FileIndex(
            version=1,
            index_type=IndexType.REGULAR,
            source_path='/path/to/file.txt',
            source_modified_at='2024-01-01T12:00:00',
            source_size_bytes=1024,
            created_at='2024-01-01T12:01:00',
            index_step_bytes=2097152,
            analysis=analysis,
            line_index=[[1, 0], [50, 2500]],
        )

        data = index.to_dict()

        assert data['version'] == 1
        assert data['source_path'] == '/path/to/file.txt'
        assert data['source_size_bytes'] == 1024
        assert data['line_index'] == [[1, 0], [50, 2500]]
        assert 'analysis' in data
        assert data['analysis']['line_count'] == 100
        assert data['analysis']['line_ending'] == 'LF'

    def test_to_dict_compressed_index(self):
        """Test converting compressed FileIndex to dict."""
        index = FileIndex(
            version=1,
            index_type=IndexType.COMPRESSED,
            source_path='/path/to/file.gz',
            source_modified_at='2024-01-01T12:00:00',
            source_size_bytes=512,
            compression_format='gzip',
            decompressed_size_bytes=2048,
            total_lines=100,
            line_sample_interval=1000,
            line_index=[[1, 0]],
        )

        data = index.to_dict()

        assert data['compression_format'] == 'gzip'
        assert data['decompressed_size_bytes'] == 2048
        assert data['total_lines'] == 100

    def test_round_trip_regular_index(self):
        """Test that to_dict -> from_dict preserves data."""
        original = FileIndex(
            version=1,
            index_type=IndexType.REGULAR,
            source_path='/path/to/file.txt',
            source_modified_at='2024-01-01T12:00:00',
            source_size_bytes=1024,
            created_at='2024-01-01T12:01:00',
            index_step_bytes=2097152,
            analysis=IndexAnalysis(
                line_count=100,
                empty_line_count=5,
                line_length_max=200,
                line_length_avg=50.5,
                line_ending='LF',
            ),
            line_index=[[1, 0], [50, 2500], [100, 5000]],
        )

        # Round trip
        data = original.to_dict()
        restored = FileIndex.from_dict(data)

        assert restored.version == original.version
        assert restored.source_path == original.source_path
        assert restored.source_size_bytes == original.source_size_bytes
        assert restored.line_index == original.line_index
        assert restored.analysis.line_count == original.analysis.line_count
        assert restored.analysis.line_ending == original.analysis.line_ending


class TestFileIndexJsonPersistence:
    """Tests for FileIndex JSON file persistence."""

    def setup_method(self):
        """Create temp directory for test files."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up temp directory."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_and_load_creates_valid_json(self):
        """Test that saved index creates valid JSON file."""
        test_file = os.path.join(self.temp_dir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write('Line 1\nLine 2\nLine 3\n')

        # Create index
        file_index = create_index_file(test_file, force=True)
        assert file_index is not None

        # Verify JSON file exists and is valid
        index_path = get_index_path(test_file)
        assert index_path.exists()

        with open(index_path) as f:
            data = json.load(f)

        assert 'version' in data
        assert 'source_path' in data
        assert 'line_index' in data
        assert 'analysis' in data

        # Clean up
        delete_index(test_file)

    def test_json_structure_matches_expected_format(self):
        """Test that saved JSON has expected structure for backward compatibility."""
        test_file = os.path.join(self.temp_dir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write('Line 1\nLine 2\nLine 3\n')

        file_index = create_index_file(test_file, force=True)
        index_path = get_index_path(test_file)

        with open(index_path) as f:
            data = json.load(f)

        # Required top-level fields
        assert data['version'] == INDEX_VERSION
        assert data['source_path'] == os.path.abspath(test_file)
        assert isinstance(data['source_size_bytes'], int)
        assert isinstance(data['source_modified_at'], str)
        assert isinstance(data['created_at'], str)
        assert isinstance(data['line_index'], list)

        # Analysis structure
        assert 'analysis' in data
        analysis = data['analysis']
        assert 'line_count' in analysis
        assert 'line_ending' in analysis

        # Line index structure
        assert len(data['line_index']) >= 1
        first_entry = data['line_index'][0]
        assert first_entry == [1, 0]  # First line at offset 0

        delete_index(test_file)

    def test_loaded_index_has_correct_types(self):
        """Test that loaded FileIndex has correct Python types."""
        test_file = os.path.join(self.temp_dir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write('Line 1\nLine 2\nLine 3\n')

        create_index_file(test_file, force=True)
        loaded = load_index(get_index_path(test_file))

        assert isinstance(loaded, FileIndex)
        assert isinstance(loaded.version, int)
        assert isinstance(loaded.source_path, str)
        assert isinstance(loaded.source_size_bytes, int)
        assert isinstance(loaded.line_index, list)
        assert isinstance(loaded.analysis, IndexAnalysis)
        assert isinstance(loaded.analysis.line_count, int)

        delete_index(test_file)

    def test_analysis_statistics_are_correct(self):
        """Test that analysis statistics in saved index are accurate."""
        test_file = os.path.join(self.temp_dir, 'test.txt')
        # Create file with known content
        lines = [
            'Short',  # 5 chars
            'Medium length',  # 13 chars
            'A very long line that is much longer',  # 37 chars
            '',  # 0 chars (empty)
            'Normal',  # 6 chars
        ]
        with open(test_file, 'w') as f:
            f.write('\n'.join(lines) + '\n')

        file_index = create_index_file(test_file, force=True)
        analysis = file_index.analysis

        assert analysis.line_count == 5
        assert analysis.empty_line_count == 1
        assert analysis.line_length_max == 36  # "A very long line that is much longer"
        assert analysis.line_ending == 'LF'

        delete_index(test_file)


class TestFrameLineInfo:
    """Tests for FrameLineInfo model (seekable zstd)."""

    def test_create_frame_info(self):
        """Test creating FrameLineInfo."""
        frame = FrameLineInfo(
            index=0,
            compressed_offset=0,
            compressed_size=1024,
            decompressed_offset=0,
            decompressed_size=4096,
            first_line=1,
            last_line=100,
            line_count=100,
        )

        assert frame.index == 0
        assert frame.compressed_size == 1024
        assert frame.decompressed_size == 4096
        assert frame.first_line == 1
        assert frame.last_line == 100

    def test_frame_info_serialization(self):
        """Test FrameLineInfo serialization round-trip."""
        original = FrameLineInfo(
            index=2,
            compressed_offset=2048,
            compressed_size=512,
            decompressed_offset=8192,
            decompressed_size=2048,
            first_line=201,
            last_line=300,
            line_count=100,
        )

        # Round-trip through dict
        data = original.model_dump()
        restored = FrameLineInfo(**data)

        assert restored.index == original.index
        assert restored.compressed_offset == original.compressed_offset
        assert restored.first_line == original.first_line
        assert restored.line_count == original.line_count


class TestFileAnalysis:
    """Tests for FileAnalysis model."""

    def test_create_file_analysis(self):
        """Test creating FileAnalysis with typical fields."""
        analysis = FileAnalysis(
            file_id='f1',
            filepath='/path/to/file.txt',
            size_bytes=1024,
            size_human='1.00 KB',
            is_text=True,
            line_count=50,
            line_ending='LF',
        )

        assert analysis.file_id == 'f1'
        assert analysis.size_bytes == 1024
        assert analysis.is_text is True
        assert analysis.line_count == 50

    def test_file_analysis_with_compression_info(self):
        """Test FileAnalysis with compression fields."""
        analysis = FileAnalysis(
            file_id='f1',
            filepath='/path/to/file.txt.gz',
            size_bytes=512,
            size_human='512 B',
            is_text=True,
            is_compressed=True,
            compression_format='gzip',
            compressed_size=512,
            decompressed_size=2048,
            compression_ratio=4.0,
        )

        assert analysis.is_compressed is True
        assert analysis.compression_format == 'gzip'
        assert analysis.compression_ratio == 4.0

    def test_to_cache_dict(self):
        """Test FileAnalysis.to_cache_dict() method."""
        analysis = FileAnalysis(
            file_id='f1',
            filepath='/path/to/file.txt',
            size_bytes=1024,
            size_human='1.00 KB',
            is_text=True,
            line_count=50,
            line_ending='LF',
        )

        cache_dict = analysis.to_cache_dict()

        assert cache_dict['file'] == 'f1'
        assert cache_dict['size_bytes'] == 1024
        assert cache_dict['is_text'] is True
        assert cache_dict['line_count'] == 50

    def test_from_cache_dict(self):
        """Test FileAnalysis.from_cache_dict() method."""
        cache_data = {
            'size_bytes': 2048,
            'size_human': '2.00 KB',
            'is_text': True,
            'line_count': 100,
            'line_ending': 'CRLF',
            'has_index': True,
        }

        analysis = FileAnalysis.from_cache_dict(cache_data, 'f2', '/path/to/file.txt')

        assert analysis.file_id == 'f2'
        assert analysis.filepath == '/path/to/file.txt'
        assert analysis.size_bytes == 2048
        assert analysis.line_count == 100
        assert analysis.has_index is True


class TestIndexVersionCompatibility:
    """Tests for index version handling."""

    def setup_method(self):
        """Create temp directory."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_rejects_wrong_version(self):
        """Test that loading index with wrong version returns None."""
        # Create a file with wrong version
        index_file = os.path.join(self.temp_dir, 'wrong_version.json')
        data = {
            'version': 999,  # Wrong version
            'source_path': '/path/to/file.txt',
            'source_modified_at': '2024-01-01T12:00:00',
            'source_size_bytes': 100,
            'line_index': [[1, 0]],
        }

        with open(index_file, 'w') as f:
            json.dump(data, f)

        loaded = load_index(index_file)
        assert loaded is None

    def test_current_version_loads_successfully(self):
        """Test that index with current version loads successfully."""
        index_file = os.path.join(self.temp_dir, 'correct_version.json')
        data = {
            'version': INDEX_VERSION,
            'source_path': '/path/to/file.txt',
            'source_modified_at': '2024-01-01T12:00:00',
            'source_size_bytes': 100,
            'line_index': [[1, 0]],
        }

        with open(index_file, 'w') as f:
            json.dump(data, f)

        loaded = load_index(index_file)
        assert loaded is not None
        assert loaded.version == INDEX_VERSION


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
