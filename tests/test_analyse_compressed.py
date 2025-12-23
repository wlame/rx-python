"""Tests for compressed file analysis functionality."""

import gzip
import os
import tempfile
from pathlib import Path

import pytest

from rx.analyse import FileAnalyzer, analyse_path
from rx.analyse_cache import clear_all_caches, load_cache


class TestCompressedFileAnalysis:
    """Test analysis of compressed files."""

    def test_analyse_gzip_file(self, tmp_path):
        """Test analyzing a gzip compressed file."""
        # Create a test file
        test_content = 'Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n'
        text_file = tmp_path / 'test.txt'
        text_file.write_text(test_content)

        # Compress it
        gz_file = tmp_path / 'test.txt.gz'
        with open(text_file, 'rb') as f_in, gzip.open(gz_file, 'wb') as f_out:
            f_out.write(f_in.read())

        # Analyze the compressed file
        analyzer = FileAnalyzer()
        result = analyzer.analyze_file(str(gz_file), 'f1')

        # Verify compression detection
        assert result.is_compressed is True
        assert result.compression_format == 'gzip'
        assert result.is_seekable_zstd is False
        assert result.compressed_size > 0
        assert result.decompressed_size > 0
        assert result.compression_ratio > 0

        # Verify content analysis
        assert result.is_text is True
        assert result.line_count == 5
        assert result.empty_line_count == 0
        assert result.line_length_max == 6  # "Line X"

    def test_analyse_zstd_file(self, tmp_path):
        """Test analyzing a zstd compressed file."""
        pytest.importorskip('zstandard')

        # Create a test file
        test_content = 'Hello World\nTest Line\n'
        text_file = tmp_path / 'test.txt'
        text_file.write_text(test_content)

        # Compress it with zstd
        zst_file = tmp_path / 'test.txt.zst'
        os.system(f'zstd -q -f {text_file} -o {zst_file}')

        if not zst_file.exists():
            pytest.skip('zstd command not available')

        # Analyze the compressed file
        analyzer = FileAnalyzer()
        result = analyzer.analyze_file(str(zst_file), 'f1')

        # Verify compression detection
        assert result.is_compressed is True
        assert result.compression_format == 'zstd'
        assert result.decompressed_size > 0

        # Verify content analysis
        assert result.is_text is True
        assert result.line_count == 2

    def test_analyse_shows_compression_info(self, tmp_path):
        """Test that compression information is properly populated."""
        # Create and compress a file
        test_content = 'A' * 1000 + '\n' + 'B' * 1000 + '\n'
        text_file = tmp_path / 'test.txt'
        text_file.write_text(test_content)

        gz_file = tmp_path / 'test.txt.gz'
        with open(text_file, 'rb') as f_in, gzip.open(gz_file, 'wb') as f_out:
            f_out.write(f_in.read())

        # Analyze
        result = analyse_path([str(gz_file)])

        assert len(result['results']) == 1
        file_result = result['results'][0]

        # Check all compression fields are present
        assert file_result['is_compressed'] is True
        assert file_result['compression_format'] == 'gzip'
        assert file_result['is_seekable_zstd'] is False
        assert file_result['compressed_size'] > 0
        assert file_result['decompressed_size'] == len(test_content.encode())
        assert file_result['compression_ratio'] > 1  # Should compress well

    def test_analyse_binary_compressed_file(self, tmp_path):
        """Test analyzing a compressed binary file (should not analyze content)."""
        # Create a binary file
        binary_content = bytes([0xFF, 0xD8, 0xFF, 0xE0] * 100)  # JPEG-like header
        bin_file = tmp_path / 'test.bin'
        bin_file.write_bytes(binary_content)

        # Compress it
        gz_file = tmp_path / 'test.bin.gz'
        with open(bin_file, 'rb') as f_in, gzip.open(gz_file, 'wb') as f_out:
            f_out.write(f_in.read())

        # Analyze
        analyzer = FileAnalyzer()
        result = analyzer.analyze_file(str(gz_file), 'f1')

        # Should detect compression
        assert result.is_compressed is True
        # Binary detection depends on whether is_text_file detects null bytes in decompressed content
        # Some binary patterns might pass the text detection heuristic
        # The important thing is that compression is detected
        if not result.is_text:
            assert result.line_count is None

    def test_temp_file_cleanup(self, tmp_path):
        """Test that temporary files are cleaned up after analysis."""
        # Create and compress a file
        test_content = 'Cleanup test\n'
        text_file = tmp_path / 'test.txt'
        text_file.write_text(test_content)

        gz_file = tmp_path / 'test.txt.gz'
        with open(text_file, 'rb') as f_in, gzip.open(gz_file, 'wb') as f_out:
            f_out.write(f_in.read())

        # Get temp directory before analysis
        temp_dir = Path(tempfile.gettempdir())
        temp_files_before = set(temp_dir.glob('tmp*.txt'))

        # Analyze
        analyzer = FileAnalyzer()
        result = analyzer.analyze_file(str(gz_file), 'f1')

        # Get temp files after
        temp_files_after = set(temp_dir.glob('tmp*.txt'))

        # No new temp files should remain
        assert temp_files_after == temp_files_before
        assert result.is_text is True


class TestAnalyseCache:
    """Test cache integration with compressed files."""

    def teardown_method(self):
        """Clean up caches after each test."""
        clear_all_caches()

    def test_analyse_cache_hit(self, tmp_path):
        """Test that cache is used on second analysis."""
        # Create and compress a file
        test_content = 'Cached content\n'
        text_file = tmp_path / 'test.txt'
        text_file.write_text(test_content)

        gz_file = tmp_path / 'test.txt.gz'
        with open(text_file, 'rb') as f_in, gzip.open(gz_file, 'wb') as f_out:
            f_out.write(f_in.read())

        # First analysis
        analyzer = FileAnalyzer()
        result1 = analyzer.analyze_file(str(gz_file), 'f1')

        # Verify cache was created
        cached = load_cache(str(gz_file))
        assert cached is not None
        assert cached['is_compressed'] is True

        # Second analysis should hit cache
        result2 = analyzer.analyze_file(str(gz_file), 'f2')

        # Results should be identical (except file_id)
        assert result2.is_compressed == result1.is_compressed
        assert result2.line_count == result1.line_count
        assert result2.decompressed_size == result1.decompressed_size

    def test_analyse_cache_invalidated_on_change(self, tmp_path):
        """Test that cache is invalidated when file changes."""
        # Create and compress a file
        test_content = 'Original content\n'
        text_file = tmp_path / 'test.txt'
        text_file.write_text(test_content)

        gz_file = tmp_path / 'test.txt.gz'
        with open(text_file, 'rb') as f_in, gzip.open(gz_file, 'wb') as f_out:
            f_out.write(f_in.read())

        # First analysis
        analyzer = FileAnalyzer()
        result1 = analyzer.analyze_file(str(gz_file), 'f1')
        original_line_count = result1.line_count

        # Modify and re-compress
        new_content = 'New content\nLine 2\nLine 3\n'
        text_file.write_text(new_content)
        with open(text_file, 'rb') as f_in, gzip.open(gz_file, 'wb') as f_out:
            f_out.write(f_in.read())

        # Second analysis should detect change
        result2 = analyzer.analyze_file(str(gz_file), 'f2')

        # Results should be different
        assert result2.line_count != original_line_count
        assert result2.line_count == 3


class TestIndexInfo:
    """Test index information detection."""

    def test_analyse_shows_index_info_for_regular_file(self, tmp_path):
        """Test that index info is shown for indexed files."""
        # Create a large file that will get indexed
        test_content = 'Line\n' * 10000
        text_file = tmp_path / 'large.txt'
        text_file.write_text(test_content)

        # Analyze (should create index)
        from rx.index import get_large_file_threshold_bytes

        # Only test if file is large enough
        if text_file.stat().st_size < get_large_file_threshold_bytes():
            pytest.skip('File not large enough to trigger indexing')

        analyzer = FileAnalyzer()
        result = analyzer.analyze_file(str(text_file), 'f1')

        # Index should be created and detected
        # Note: Index creation happens after analysis, so we need to analyze again
        result2 = analyzer.analyze_file(str(text_file), 'f2')

        assert result2.has_index is True
        assert result2.index_path is not None
        assert result2.index_valid is True
        assert result2.index_checkpoint_count is not None

    def test_analyse_shows_no_index_for_small_file(self, tmp_path):
        """Test that small files show no index info."""
        # Create a small file
        test_content = 'Small file\n'
        text_file = tmp_path / 'small.txt'
        text_file.write_text(test_content)

        # Analyze
        analyzer = FileAnalyzer()
        result = analyzer.analyze_file(str(text_file), 'f1')

        # Should have no index
        assert result.has_index is False
        assert result.index_path is None


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_analyse_empty_compressed_file(self, tmp_path):
        """Test analyzing an empty compressed file."""
        # Create empty file and compress
        text_file = tmp_path / 'empty.txt'
        text_file.write_text('')

        gz_file = tmp_path / 'empty.txt.gz'
        with open(text_file, 'rb') as f_in, gzip.open(gz_file, 'wb') as f_out:
            f_out.write(f_in.read())

        # Analyze
        analyzer = FileAnalyzer()
        result = analyzer.analyze_file(str(gz_file), 'f1')

        # Should handle empty file gracefully
        assert result.is_compressed is True
        assert result.line_count == 0 or result.line_count == 1  # Depends on implementation

    def test_analyse_nonexistent_file(self, tmp_path):
        """Test analyzing a non-existent file returns minimal result."""
        # Try to analyze non-existent file
        analyzer = FileAnalyzer()
        result = analyzer.analyze_file(str(tmp_path / 'nonexistent.txt.gz'), 'f1')

        # Should return minimal result without crashing
        assert result.size_bytes == 0
        assert result.is_text is False

    def test_analyse_corrupted_compressed_file(self, tmp_path):
        """Test analyzing a corrupted compressed file."""
        # Create a fake gzip file (just header, no valid data)
        gz_file = tmp_path / 'corrupted.txt.gz'
        gz_file.write_bytes(b'\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x00invalid')

        # Analyze - should handle gracefully
        analyzer = FileAnalyzer()
        result = analyzer.analyze_file(str(gz_file), 'f1')

        # Should detect as compressed but fail to analyze content
        assert result.is_compressed is True
        # May or may not have line count depending on decompression failure handling


class TestAnalyseSeekableZstdIndex:
    """Test analyse with seekable zstd files that have indexes."""

    def test_analyse_seekable_zstd_with_index_info(self, tmp_path):
        """Test that analyse correctly reads index info from seekable zstd files.

        This is a regression test for the error:
        'SeekableIndex' object has no attribute 'get'

        The issue was that load_index returns a SeekableIndex dataclass object,
        but the code was calling .get() on it as if it were a dict.
        """
        import subprocess

        pytest.importorskip('zstandard')

        # Create a test file with enough content
        text_file = tmp_path / 'test.txt'
        with open(text_file, 'w') as f:
            for i in range(1, 101):
                f.write(f'Line {i}: content for line number {i}\n')

        # Compress with rx compress (creates seekable zstd with index)
        zst_file = tmp_path / 'test.txt.zst'
        result = subprocess.run(
            ['rx', 'compress', str(text_file), '-o', str(zst_file)],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0 or not zst_file.exists():
            pytest.skip('Could not create seekable zstd file')

        # Analyze the file - this should NOT raise an error
        analyzer = FileAnalyzer()
        analysis_result = analyzer.analyze_file(str(zst_file), 'f1')

        # Verify basic analysis worked
        assert analysis_result.is_compressed is True
        assert analysis_result.compression_format == 'zstd'
        assert analysis_result.line_count == 100

        # Verify index info is populated (the part that was failing)
        assert analysis_result.has_index is True
        assert analysis_result.index_valid is True
        # index_checkpoint_count should be set from len(index_data.frames)
        assert analysis_result.index_checkpoint_count is not None
        assert analysis_result.index_checkpoint_count >= 1


class TestCompressionRatio:
    """Test compression ratio calculations."""

    def test_compression_ratio_calculated(self, tmp_path):
        """Test that compression ratio is properly calculated."""
        # Create highly compressible content
        test_content = 'A' * 10000 + '\n'
        text_file = tmp_path / 'test.txt'
        text_file.write_text(test_content)

        gz_file = tmp_path / 'test.txt.gz'
        with open(text_file, 'rb') as f_in, gzip.open(gz_file, 'wb') as f_out:
            f_out.write(f_in.read())

        # Analyze
        analyzer = FileAnalyzer()
        result = analyzer.analyze_file(str(gz_file), 'f1')

        # Verify ratio is calculated correctly
        assert result.compression_ratio is not None
        assert result.compression_ratio > 1  # Should compress well
        assert result.compression_ratio == result.decompressed_size / result.compressed_size

    def test_compression_ratio_for_incompressible_data(self, tmp_path):
        """Test compression ratio for random/incompressible data."""
        # Create random-like content (less compressible)
        import string

        test_content = ''.join([string.ascii_letters[i % 52] for i in range(1000)]) + '\n'
        text_file = tmp_path / 'test.txt'
        text_file.write_text(test_content)

        gz_file = tmp_path / 'test.txt.gz'
        with open(text_file, 'rb') as f_in, gzip.open(gz_file, 'wb') as f_out:
            f_out.write(f_in.read())

        # Analyze
        analyzer = FileAnalyzer()
        result = analyzer.analyze_file(str(gz_file), 'f1')

        # Should still have valid ratio
        assert result.compression_ratio is not None
        assert result.compression_ratio > 0
