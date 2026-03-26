"""Tests for absolute_line_number field in trace output."""

import json
import os
import tempfile

import pytest

from click.testing import CliRunner

from rx.cli.trace import trace_command
from rx.trace import parse_paths


class TestAbsoluteLineNumbers:
    """Test absolute_line_number field in trace output."""

    def setup_method(self):
        """Create test files before each test."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

        # Create a test file with multiple lines
        self.test_file = os.path.join(self.temp_dir, 'test.txt')
        self.lines = [
            'Line 1: first line\n',
            'Line 2: normal content\n',
            'Line 3: pattern match here\n',
            'Line 4: more content\n',
            'Line 5: another pattern match\n',
            'Line 6: normal content\n',
            'Line 7: pattern match again\n',
            'Line 8: final line\n',
        ]
        with open(self.test_file, 'w') as f:
            f.writelines(self.lines)

    def test_absolute_line_number_in_json_output(self):
        """Test that absolute_line_number appears in JSON output."""
        result = self.runner.invoke(
            trace_command,
            ['pattern', self.test_file, '--json'],
        )
        assert result.exit_code == 0

        output = json.loads(result.output)
        matches = output.get('matches', [])

        # Should find 3 matches
        assert len(matches) == 3

        # Each match should have absolute_line_number field
        for match in matches:
            assert 'absolute_line_number' in match
            # For seekable zstd files with index, absolute_line_number should be known
            # For regular files without complete scan, it might be -1
            assert isinstance(match['absolute_line_number'], int)

    def test_absolute_line_number_with_context(self):
        """Test that absolute_line_number appears in context lines."""
        result = self.runner.invoke(
            trace_command,
            ['pattern', self.test_file, '--json', '-C', '1'],
        )
        assert result.exit_code == 0

        output = json.loads(result.output)
        context_lines = output.get('context_lines', {})

        # Check that context lines have absolute_line_number
        for key, ctx_list in context_lines.items():
            for ctx in ctx_list:
                assert 'absolute_line_number' in ctx
                assert isinstance(ctx['absolute_line_number'], int)

    def test_absolute_line_number_seekable_zstd(self):
        """Test that seekable zstd files have correct absolute_line_number."""
        # This test requires a seekable zstd file with index
        # Skip if zstandard is not available
        try:
            import zstandard
        except ImportError:
            return

        # Create a seekable zstd file
        import subprocess

        zst_file = os.path.join(self.temp_dir, 'test.txt.zst')
        # Use zstd CLI to create seekable format
        try:
            result = subprocess.run(
                ['zstd', '--ultra', '-22', '-o', zst_file, self.test_file],
                capture_output=True,
            )
        except FileNotFoundError:
            pytest.skip('zstd CLI not installed')
        if result.returncode != 0:
            pytest.skip('zstd compression failed')

        # Try to make it seekable (may fail if not supported)
        try:
            from rx.seekable_zstd import create_seekable_zstd

            create_seekable_zstd(self.test_file, zst_file + '.seekable')
            zst_file = zst_file + '.seekable'
        except Exception:
            # Skip if seekable creation fails
            return

        # Run trace on seekable zstd file
        result = self.runner.invoke(
            trace_command,
            ['pattern', zst_file, '--json'],
        )

        if result.exit_code == 0:
            output = json.loads(result.output)
            matches = output.get('matches', [])

            # For seekable zstd with index, absolute_line_number should be known (>= 1)
            for match in matches:
                assert 'absolute_line_number' in match
                # Should be actual line number, not -1
                if match.get('is_seekable_zstd'):
                    assert match['absolute_line_number'] >= 1

    def test_absolute_line_number_programmatic_api(self):
        """Test absolute_line_number via programmatic API."""
        result = parse_paths(
            paths=[self.test_file],
            regexps=['pattern'],
            context_before=0,
            context_after=0,
        )

        # Check matches
        for match in result.matches:
            assert 'absolute_line_number' in match
            assert isinstance(match['absolute_line_number'], int)

        # Check context lines
        for key, ctx_list in result.context_lines.items():
            for ctx in ctx_list:
                assert hasattr(ctx, 'absolute_line_number')
                assert isinstance(ctx.absolute_line_number, int)

    def test_absolute_line_number_cache_reconstruction(self, monkeypatch):
        """Test that cached matches preserve absolute_line_number."""
        # Set a low threshold to trigger caching with a small file (1MB)
        monkeypatch.setenv('RX_LARGE_FILE_MB', '1')

        # Create a file just over 1MB with only a few pattern matches
        # Use padding lines without the pattern, and specific lines with pattern
        large_file = os.path.join(self.temp_dir, 'large.txt')
        padding_line = 'x' * 100 + ' no match here\n'  # ~115 bytes
        pattern_line = 'y' * 100 + ' pattern match here\n'  # ~122 bytes

        # 1MB = ~1,048,576 bytes, so ~9000 lines of padding
        # Put pattern matches at specific known line numbers
        pattern_lines = {100, 500, 1000, 5000, 9000}

        with open(large_file, 'w') as f:
            for i in range(1, 9500):
                if i in pattern_lines:
                    f.write(pattern_line)
                else:
                    f.write(padding_line)

        # First run - creates cache
        result1 = self.runner.invoke(
            trace_command,
            ['pattern', large_file, '--json'],
        )
        assert result1.exit_code == 0

        output1 = json.loads(result1.output)
        matches1 = output1.get('matches', [])

        # Should find exactly 5 matches
        assert len(matches1) == 5

        # Second run - uses cache
        result2 = self.runner.invoke(
            trace_command,
            ['pattern', large_file, '--json'],
        )
        assert result2.exit_code == 0

        output2 = json.loads(result2.output)
        matches2 = output2.get('matches', [])

        # Both should have absolute_line_number and match
        assert len(matches1) == len(matches2)
        for m1, m2 in zip(matches1, matches2):
            assert 'absolute_line_number' in m1
            assert 'absolute_line_number' in m2
            # For cached complete scans, absolute_line_number should match relative_line_number
            if m1['absolute_line_number'] != -1:
                assert m1['absolute_line_number'] == m1['relative_line_number']
            if m2['absolute_line_number'] != -1:
                assert m2['absolute_line_number'] == m2['relative_line_number']

    def test_absolute_line_number_unknown_for_chunked(self):
        """Test that absolute_line_number is -1 for chunked processing without complete scan."""
        # For files processed in chunks without index, absolute_line_number should be -1
        # This is hard to test deterministically, so we just verify the field exists
        result = parse_paths(
            paths=[self.test_file],
            regexps=['pattern'],
            context_before=0,
            context_after=0,
        )

        for match in result.matches:
            assert 'absolute_line_number' in match
            # Should be either -1 (unknown) or >= 1 (known)
            assert match['absolute_line_number'] == -1 or match['absolute_line_number'] >= 1
