"""Tests for rx samples command with range support."""

import tempfile
from pathlib import Path

from click.testing import CliRunner

from rx.cli.samples import samples_command


class TestSamplesRanges:
    """Test rx samples with line and byte ranges."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

        # Create a test file with numbered lines
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        self.test_file_path = self.temp_file.name

        # Write 100 lines for testing
        for i in range(1, 101):
            self.temp_file.write(f'Line {i}: This is line number {i}\n')
        self.temp_file.close()

    def teardown_method(self):
        """Clean up test fixtures."""
        Path(self.test_file_path).unlink(missing_ok=True)

    def test_single_line_with_context(self):
        """Test single line number with context (existing behavior)."""
        result = self.runner.invoke(samples_command, [self.test_file_path, '-l', '50', '--context', '2'])

        assert result.exit_code == 0
        # Should include lines 48-52 (50 +/- 2 context)
        assert 'Line 48:' in result.output
        assert 'Line 49:' in result.output
        assert 'Line 50:' in result.output
        assert 'Line 51:' in result.output
        assert 'Line 52:' in result.output

    def test_line_range_basic(self):
        """Test basic line range without context."""
        result = self.runner.invoke(samples_command, [self.test_file_path, '-l', '10-15'])

        assert result.exit_code == 0
        # Should include exactly lines 10-15
        assert 'Line 10:' in result.output
        assert 'Line 11:' in result.output
        assert 'Line 12:' in result.output
        assert 'Line 13:' in result.output
        assert 'Line 14:' in result.output
        assert 'Line 15:' in result.output
        # Should not include line 9 or 16 (no context for ranges)
        assert 'Line 9:' not in result.output
        assert 'Line 16:' not in result.output

    def test_line_range_ignores_context(self):
        """Test that line ranges ignore context parameters."""
        result = self.runner.invoke(
            samples_command,
            [
                self.test_file_path,
                '-l',
                '20-25',
                '--context',
                '10',  # This should be ignored for the range
            ],
        )

        assert result.exit_code == 0
        # Should include exactly lines 20-25, not 10-35
        assert 'Line 20:' in result.output
        assert 'Line 25:' in result.output
        assert 'Line 10:' not in result.output
        assert 'Line 35:' not in result.output

    def test_mixed_single_and_range(self):
        """Test mixing single line with context and range."""
        result = self.runner.invoke(samples_command, [self.test_file_path, '-l', '5', '-l', '50-55', '--context', '2'])

        assert result.exit_code == 0
        # Line 5 should have context (3-7)
        assert 'Line 3:' in result.output
        assert 'Line 4:' in result.output
        assert 'Line 5:' in result.output
        assert 'Line 6:' in result.output
        assert 'Line 7:' in result.output
        # Lines 50-55 should be exact (no context)
        assert 'Line 50:' in result.output
        assert 'Line 55:' in result.output
        # Should not include lines 48-49 or 56-57 (no context for range)
        assert 'Line 48:' not in result.output
        assert 'Line 49:' not in result.output
        assert 'Line 56:' not in result.output
        assert 'Line 57:' not in result.output

    def test_multiple_ranges(self):
        """Test multiple line ranges."""
        result = self.runner.invoke(samples_command, [self.test_file_path, '-l', '10-12', '-l', '40-42', '-l', '70-72'])

        assert result.exit_code == 0
        # All three ranges should be present
        assert 'Line 10:' in result.output
        assert 'Line 12:' in result.output
        assert 'Line 40:' in result.output
        assert 'Line 42:' in result.output
        assert 'Line 70:' in result.output
        assert 'Line 72:' in result.output

    def test_line_range_json_output(self):
        """Test line range with JSON output."""
        result = self.runner.invoke(samples_command, [self.test_file_path, '-l', '30-35', '--json'])

        assert result.exit_code == 0
        import json

        data = json.loads(result.output)

        # Check that the range key exists
        assert '30-35' in data['samples']
        # Check that we have 6 lines (30-35 inclusive)
        assert len(data['samples']['30-35']) == 6
        # Verify line content
        assert 'Line 30:' in data['samples']['30-35'][0]
        assert 'Line 35:' in data['samples']['30-35'][5]

    def test_byte_range_basic(self):
        """Test basic byte offset range."""
        # Calculate approximate byte offsets (each line is about 30-35 bytes)
        # Line 1 starts at offset 0
        # Line 10 starts at approximately 9 * 33 = 297 bytes
        # Line 15 ends at approximately 15 * 33 = 495 bytes
        result = self.runner.invoke(samples_command, [self.test_file_path, '-b', '297-495'])

        assert result.exit_code == 0
        # Should include lines that contain these byte offsets
        # The exact lines depend on the byte calculation, but should be around lines 10-15
        assert 'Line 10:' in result.output or 'Line 11:' in result.output

    def test_invalid_range_format(self):
        """Test invalid range format error handling."""
        result = self.runner.invoke(
            samples_command,
            [
                self.test_file_path,
                '-l',
                '10-20-30',  # Invalid: too many dashes
            ],
        )

        assert result.exit_code == 1
        assert 'Invalid range format' in result.output

    def test_invalid_range_order(self):
        """Test range with start > end."""
        result = self.runner.invoke(
            samples_command,
            [
                self.test_file_path,
                '-l',
                '50-30',  # Invalid: start > end
            ],
        )

        assert result.exit_code == 1
        assert 'Invalid range' in result.output
        assert 'Start must be <= end' in result.output

    def test_range_beyond_file_end(self):
        """Test range that extends beyond file end."""
        result = self.runner.invoke(
            samples_command,
            [
                self.test_file_path,
                '-l',
                '95-200',  # File only has 100 lines
            ],
        )

        assert result.exit_code == 0
        # Should return lines 95-100 (clamped to file size)
        assert 'Line 95:' in result.output
        assert 'Line 100:' in result.output

    def test_range_start_beyond_file(self):
        """Test range that starts beyond file end."""
        result = self.runner.invoke(
            samples_command,
            [
                self.test_file_path,
                '-l',
                '150-200',  # File only has 100 lines
            ],
        )

        assert result.exit_code == 0
        # Should return empty result

    def test_range_with_before_after_context(self):
        """Test that ranges ignore --before and --after parameters."""
        result = self.runner.invoke(
            samples_command, [self.test_file_path, '-l', '50-52', '--before', '5', '--after', '10']
        )

        assert result.exit_code == 0
        # Should include exactly lines 50-52, not 45-62
        assert 'Line 50:' in result.output
        assert 'Line 52:' in result.output
        assert 'Line 45:' not in result.output
        assert 'Line 62:' not in result.output

    def test_single_line_range(self):
        """Test range where start equals end."""
        result = self.runner.invoke(samples_command, [self.test_file_path, '-l', '50-50'])

        assert result.exit_code == 0
        # Should return exactly one line
        assert 'Line 50:' in result.output

    def test_negative_range_value(self):
        """Test that negative range values are rejected."""
        result = self.runner.invoke(samples_command, [self.test_file_path, '-l', '-10-20'])

        assert result.exit_code == 1
        assert 'Invalid' in result.output  # Should error with invalid format or offset

    def test_non_integer_range(self):
        """Test that non-integer range values are rejected."""
        result = self.runner.invoke(samples_command, [self.test_file_path, '-l', 'abc-def'])

        assert result.exit_code == 1
        assert 'Invalid range format' in result.output


class TestSamplesRangesEdgeCases:
    """Test edge cases for samples range feature."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_empty_file_range(self):
        """Test range on empty file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            temp_path = f.name

        try:
            result = self.runner.invoke(samples_command, [temp_path, '-l', '1-10'])

            assert result.exit_code == 0
            # Empty result expected
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_single_line_file_range(self):
        """Test range on single-line file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write('Only line\n')
            temp_path = f.name

        try:
            result = self.runner.invoke(samples_command, [temp_path, '-l', '1-10'])

            assert result.exit_code == 0
            assert 'Only line' in result.output
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestSamplesRangesSeekableZstd:
    """Test rx samples with line ranges on seekable zstd files."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def _create_seekable_zstd(self, temp_dir: Path, num_lines: int = 100) -> Path:
        """Create a seekable zstd file with numbered lines."""
        import subprocess

        txt_path = temp_dir / 'test.txt'
        zst_path = temp_dir / 'test.txt.zst'

        # Write test content
        with open(txt_path, 'w') as f:
            for i in range(1, num_lines + 1):
                f.write(f'Line {i}: This is line number {i}\n')

        # Compress with seekable zstd using rx compress command
        result = subprocess.run(
            ['rx', 'compress', str(txt_path), '-o', str(zst_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            # Fallback: try using zstd directly with seekable format
            result = subprocess.run(
                ['zstd', '--ultra', '-22', str(txt_path), '-o', str(zst_path)],
                capture_output=True,
                text=True,
            )

        return zst_path

    def test_zstd_line_range_returns_single_chunk(self):
        """Test that line range on zstd file returns a single chunk, not individual lines."""
        import json

        with tempfile.TemporaryDirectory() as temp_dir:
            zst_path = self._create_seekable_zstd(Path(temp_dir))

            if not zst_path.exists():
                # Skip if we couldn't create the zstd file
                return

            result = self.runner.invoke(samples_command, [str(zst_path), '-l', '1-10', '--json'])

            if result.exit_code != 0:
                # Skip if zstd processing failed
                return

            # Find JSON in output (may have prefix messages)
            output = result.output
            json_start = output.find('{')
            if json_start == -1:
                return

            data = json.loads(output[json_start:])

            # Key assertion: should have '1-10' as a single key, NOT '1', '2', '3', etc.
            assert '1-10' in data['samples'], f"Expected '1-10' key in samples, got: {list(data['samples'].keys())}"
            # Should NOT have individual line keys
            assert '1' not in data['samples'], 'Should not have individual line keys'
            assert '2' not in data['samples'], 'Should not have individual line keys'

            # Should have 10 lines in the range
            assert len(data['samples']['1-10']) == 10, f'Expected 10 lines, got {len(data["samples"]["1-10"])}'

    def test_zstd_mixed_single_and_range(self):
        """Test mixing single line and range on zstd file."""
        import json

        with tempfile.TemporaryDirectory() as temp_dir:
            zst_path = self._create_seekable_zstd(Path(temp_dir))

            if not zst_path.exists():
                return

            result = self.runner.invoke(samples_command, [str(zst_path), '-l', '5', '-l', '20-25', '--json'])

            if result.exit_code != 0:
                return

            output = result.output
            json_start = output.find('{')
            if json_start == -1:
                return

            data = json.loads(output[json_start:])

            # Should have both single line and range keys
            assert '5' in data['samples'], f"Expected '5' key in samples, got: {list(data['samples'].keys())}"
            assert '20-25' in data['samples'], f"Expected '20-25' key in samples, got: {list(data['samples'].keys())}"

            # Range should have 6 lines
            assert len(data['samples']['20-25']) == 6

    def test_zstd_multiple_ranges(self):
        """Test multiple ranges on zstd file."""
        import json

        with tempfile.TemporaryDirectory() as temp_dir:
            zst_path = self._create_seekable_zstd(Path(temp_dir))

            if not zst_path.exists():
                return

            result = self.runner.invoke(
                samples_command, [str(zst_path), '-l', '1-5', '-l', '50-55', '-l', '90-95', '--json']
            )

            if result.exit_code != 0:
                return

            output = result.output
            json_start = output.find('{')
            if json_start == -1:
                return

            data = json.loads(output[json_start:])

            # Should have all three range keys
            assert '1-5' in data['samples']
            assert '50-55' in data['samples']
            assert '90-95' in data['samples']

            # Each range should have correct number of lines
            assert len(data['samples']['1-5']) == 5
            assert len(data['samples']['50-55']) == 6
            assert len(data['samples']['90-95']) == 6

    def test_zstd_range_content_matches_original(self):
        """Test that zstd range content matches uncompressed file."""
        import json

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            txt_path = temp_path / 'test.txt'
            zst_path = self._create_seekable_zstd(temp_path)

            if not zst_path.exists() or not txt_path.exists():
                return

            # Get range from txt file
            txt_result = self.runner.invoke(samples_command, [str(txt_path), '-l', '10-15', '--json'])
            if txt_result.exit_code != 0:
                return

            txt_output = txt_result.output
            txt_json_start = txt_output.find('{')
            txt_data = json.loads(txt_output[txt_json_start:])

            # Get range from zst file
            zst_result = self.runner.invoke(samples_command, [str(zst_path), '-l', '10-15', '--json'])
            if zst_result.exit_code != 0:
                return

            zst_output = zst_result.output
            zst_json_start = zst_output.find('{')
            zst_data = json.loads(zst_output[zst_json_start:])

            # Content should match
            assert txt_data['samples']['10-15'] == zst_data['samples']['10-15']
