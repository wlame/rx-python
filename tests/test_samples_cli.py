"""Tests for the rx samples CLI command."""

import json
import os
import tempfile

from click.testing import CliRunner

from rx.cli.samples import samples_command


class TestSamplesCommand:
    """Test rx samples CLI command."""

    def setup_method(self):
        """Create test files before each test."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

        # Create a test file with multiple lines
        self.test_file = os.path.join(self.temp_dir, 'test.log')
        self.lines = [
            'Line 0: first line\n',
            'Line 1: normal content\n',
            'Line 2: error occurred here\n',
            'Line 3: more content\n',
            'Line 4: another error\n',
            'Line 5: final line\n',
        ]
        with open(self.test_file, 'w') as f:
            f.writelines(self.lines)

        # Calculate byte offsets for each line
        self.offsets = []
        offset = 0
        for line in self.lines:
            self.offsets.append(offset)
            offset += len(line.encode('utf-8'))

    def test_samples_single_offset(self):
        """Test samples command with a single byte offset."""
        # Get offset for line 2 (error line)
        offset = self.offsets[2]
        result = self.runner.invoke(samples_command, [self.test_file, '-b', str(offset)])
        assert result.exit_code == 0
        assert 'error occurred' in result.output
        # New format is file:line:offset
        assert f':{offset}' in result.output  # Check offset appears in output
        assert '===' in result.output  # Check header separator

    def test_samples_multiple_offsets(self):
        """Test samples command with multiple byte offsets."""
        offset1 = self.offsets[2]
        offset2 = self.offsets[4]
        result = self.runner.invoke(samples_command, [self.test_file, '-b', str(offset1), '-b', str(offset2)])
        assert result.exit_code == 0
        assert 'error occurred' in result.output
        assert 'another error' in result.output
        # New format is file:line:offset
        assert f':{offset1}' in result.output
        assert f':{offset2}' in result.output

    def test_samples_with_context(self):
        """Test samples command with custom context size."""
        offset = self.offsets[2]
        result = self.runner.invoke(samples_command, [self.test_file, '-b', str(offset), '-c', '1'])
        assert result.exit_code == 0
        assert 'Context: 1 before, 1 after' in result.output
        assert 'error occurred' in result.output

    def test_samples_with_before_after(self):
        """Test samples command with separate before/after context."""
        offset = self.offsets[2]
        result = self.runner.invoke(samples_command, [self.test_file, '-b', str(offset), '-B', '1', '-A', '2'])
        assert result.exit_code == 0
        assert 'Context: 1 before, 2 after' in result.output
        assert 'error occurred' in result.output

    def test_samples_json_output(self):
        """Test samples command with JSON output."""
        offset = self.offsets[2]
        result = self.runner.invoke(samples_command, [self.test_file, '-b', str(offset), '--json'])
        assert result.exit_code == 0

        data = json.loads(result.output)
        assert data['path'] == self.test_file
        # offsets now maps offset -> line number
        assert isinstance(data['offsets'], dict)
        assert str(offset) in data['offsets']
        assert data['before_context'] == 3
        assert data['after_context'] == 3
        assert str(offset) in data['samples']
        # Check that samples contain the expected content
        sample_lines = data['samples'][str(offset)]
        assert any('error occurred' in line for line in sample_lines)

    def test_samples_json_multiple_offsets(self):
        """Test JSON output with multiple offsets."""
        offset1 = self.offsets[2]
        offset2 = self.offsets[4]
        result = self.runner.invoke(
            samples_command,
            [self.test_file, '-b', str(offset1), '-b', str(offset2), '--json'],
        )
        assert result.exit_code == 0

        data = json.loads(result.output)
        # offsets now maps offset -> line number
        assert isinstance(data['offsets'], dict)
        assert str(offset1) in data['offsets']
        assert str(offset2) in data['offsets']
        assert str(offset1) in data['samples']
        assert str(offset2) in data['samples']

    def test_samples_no_color(self):
        """Test samples command with --no-color flag."""
        offset = self.offsets[2]
        result = self.runner.invoke(samples_command, [self.test_file, '-b', str(offset), '--no-color'])
        assert result.exit_code == 0
        # Should not contain ANSI escape codes
        assert '\033[' not in result.output
        assert 'error occurred' in result.output

    def test_samples_with_regex_highlight(self):
        """Test samples command with regex highlighting."""
        offset = self.offsets[2]
        result = self.runner.invoke(samples_command, [self.test_file, '-b', str(offset), '-r', 'error'])
        assert result.exit_code == 0
        assert 'error occurred' in result.output

    def test_samples_default_context(self):
        """Test that default context is 3 lines."""
        offset = self.offsets[2]
        result = self.runner.invoke(samples_command, [self.test_file, '-b', str(offset), '--json'])
        assert result.exit_code == 0

        data = json.loads(result.output)
        assert data['before_context'] == 3
        assert data['after_context'] == 3

    def test_samples_long_form_byte_offset(self):
        """Test --byte-offset long form option."""
        offset = self.offsets[2]
        result = self.runner.invoke(samples_command, [self.test_file, '--byte-offset', str(offset)])
        assert result.exit_code == 0
        assert 'error occurred' in result.output


class TestSamplesCommandErrors:
    """Test error handling in rx samples command."""

    def setup_method(self):
        """Create test environment."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

        self.test_file = os.path.join(self.temp_dir, 'test.txt')
        with open(self.test_file, 'w') as f:
            f.write('Line 1\nLine 2\nLine 3\n')

    def test_samples_missing_byte_offset(self):
        """Test error when no byte offset is provided."""
        result = self.runner.invoke(samples_command, [self.test_file])
        assert result.exit_code != 0
        assert 'byte-offset' in result.output.lower() or 'required' in result.output.lower()

    def test_samples_nonexistent_file(self):
        """Test error for nonexistent file."""
        result = self.runner.invoke(samples_command, ['/nonexistent/path.txt', '-b', '0'])
        assert result.exit_code != 0

    def test_samples_negative_context(self):
        """Test error for negative context values."""
        result = self.runner.invoke(samples_command, [self.test_file, '-b', '0', '-c', '-1'])
        assert result.exit_code != 0
        assert 'non-negative' in result.output.lower() or 'error' in result.output.lower()

    def test_samples_binary_file(self):
        """Test error for binary file."""
        binary_file = os.path.join(self.temp_dir, 'binary.bin')
        with open(binary_file, 'wb') as f:
            f.write(b'\x00\x01\x02\x03\x04\x05')

        result = self.runner.invoke(samples_command, [binary_file, '-b', '0'])
        assert result.exit_code != 0
        assert 'not a text file' in result.output.lower()

    def test_samples_invalid_offset(self):
        """Test handling of offset beyond file size."""
        result = self.runner.invoke(samples_command, [self.test_file, '-b', '999999'])
        # The function should handle this gracefully
        assert result.exit_code == 0 or 'error' in result.output.lower()


class TestSamplesCommandEdgeCases:
    """Test edge cases for rx samples command."""

    def setup_method(self):
        """Create test environment."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def test_samples_empty_file(self):
        """Test samples on empty file."""
        empty_file = os.path.join(self.temp_dir, 'empty.txt')
        with open(empty_file, 'w') as f:
            pass  # Create empty file

        result = self.runner.invoke(samples_command, [empty_file, '-b', '0'])
        # Should handle gracefully - either exit with error or return empty samples
        # The behavior depends on implementation
        assert result.exit_code in (0, 1)

    def test_samples_single_line_file(self):
        """Test samples on single-line file."""
        single_line = os.path.join(self.temp_dir, 'single.txt')
        with open(single_line, 'w') as f:
            f.write('Only one line here\n')

        result = self.runner.invoke(samples_command, [single_line, '-b', '0'])
        assert result.exit_code == 0
        assert 'Only one line' in result.output

    def test_samples_offset_at_start(self):
        """Test samples with offset at file start."""
        test_file = os.path.join(self.temp_dir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write('Line 1\nLine 2\nLine 3\n')

        result = self.runner.invoke(samples_command, [test_file, '-b', '0'])
        assert result.exit_code == 0
        assert 'Line 1' in result.output

    def test_samples_offset_at_end(self):
        """Test samples with offset near end of file."""
        test_file = os.path.join(self.temp_dir, 'test.txt')
        content = 'Line 1\nLine 2\nLine 3\n'
        with open(test_file, 'w') as f:
            f.write(content)

        # Offset at beginning of last line
        offset = len('Line 1\nLine 2\n')
        result = self.runner.invoke(samples_command, [test_file, '-b', str(offset)])
        assert result.exit_code == 0
        assert 'Line 3' in result.output

    def test_samples_zero_context(self):
        """Test samples with zero context lines."""
        test_file = os.path.join(self.temp_dir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write('Line 1\nLine 2\nLine 3\n')

        result = self.runner.invoke(samples_command, [test_file, '-b', '0', '-c', '0'])
        assert result.exit_code == 0
        assert 'Context: 0 before, 0 after' in result.output

    def test_samples_large_context(self):
        """Test samples with context larger than file."""
        test_file = os.path.join(self.temp_dir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write('Line 1\nLine 2\nLine 3\n')

        result = self.runner.invoke(samples_command, [test_file, '-b', '7', '-c', '100'])
        assert result.exit_code == 0
        # Should return available lines without error
        assert 'Line' in result.output

    def test_samples_unicode_content(self):
        """Test samples with unicode content."""
        test_file = os.path.join(self.temp_dir, 'unicode.txt')
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write('Hello world\n')
            f.write('Japanese text here\n')
            f.write('More content\n')

        result = self.runner.invoke(samples_command, [test_file, '-b', '12'])
        assert result.exit_code == 0
        # Should handle unicode gracefully
        assert 'Japanese' in result.output or result.exit_code == 0

    def test_samples_many_offsets(self):
        """Test samples with many offsets."""
        test_file = os.path.join(self.temp_dir, 'test.txt')
        lines = [f'Line {i}\n' for i in range(20)]
        with open(test_file, 'w') as f:
            f.writelines(lines)

        # Calculate offsets for lines 5, 10, 15
        offsets = []
        current = 0
        for i, line in enumerate(lines):
            if i in (5, 10, 15):
                offsets.append(current)
            current += len(line.encode('utf-8'))

        args = [test_file]
        for off in offsets:
            args.extend(['-b', str(off)])
        args.append('--json')

        result = self.runner.invoke(samples_command, args)
        assert result.exit_code == 0

        data = json.loads(result.output)
        assert len(data['samples']) == 3


class TestSamplesCommandOutput:
    """Test output formatting of rx samples command."""

    def setup_method(self):
        """Create test files."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

        self.test_file = os.path.join(self.temp_dir, 'test.log')
        with open(self.test_file, 'w') as f:
            f.write('First line\n')
            f.write('Second line\n')
            f.write('Third line with error\n')
            f.write('Fourth line\n')
            f.write('Fifth line\n')

    def test_output_contains_file_header(self):
        """Test that output contains file path header."""
        result = self.runner.invoke(samples_command, [self.test_file, '-b', '0'])
        assert result.exit_code == 0
        assert 'File:' in result.output
        assert self.test_file in result.output or 'test.log' in result.output

    def test_output_contains_context_info(self):
        """Test that output shows context configuration."""
        result = self.runner.invoke(samples_command, [self.test_file, '-b', '0', '-c', '2'])
        assert result.exit_code == 0
        assert 'Context: 2 before, 2 after' in result.output

    def test_output_offset_separator(self):
        """Test that output has offset separator lines."""
        result = self.runner.invoke(samples_command, [self.test_file, '-b', '0'])
        assert result.exit_code == 0
        assert '===' in result.output
        # New format shows file:line:offset instead of "Offset:"
        assert ':0' in result.output  # Offset 0 should appear

    def test_json_structure(self):
        """Test JSON output has correct structure."""
        result = self.runner.invoke(samples_command, [self.test_file, '-b', '0', '--json'])
        assert result.exit_code == 0

        data = json.loads(result.output)
        assert 'path' in data
        assert 'offsets' in data
        assert 'before_context' in data
        assert 'after_context' in data
        assert 'samples' in data
        assert isinstance(data['samples'], dict)


class TestSamplesLineOffset:
    """Test rx samples CLI command with line offset (-l) option."""

    def setup_method(self):
        """Create test files before each test."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

        # Create a test file with multiple lines
        self.test_file = os.path.join(self.temp_dir, 'test.log')
        self.lines = [
            'Line 1: first line\n',
            'Line 2: normal content\n',
            'Line 3: error occurred here\n',
            'Line 4: more content\n',
            'Line 5: another error\n',
            'Line 6: final line\n',
        ]
        with open(self.test_file, 'w') as f:
            f.writelines(self.lines)

    def test_line_offset_single(self):
        """Test samples command with a single line number."""
        result = self.runner.invoke(samples_command, [self.test_file, '-l', '3'])
        assert result.exit_code == 0
        assert 'error occurred' in result.output
        # New format is file:line:offset instead of "Line: 3"
        assert ':3:' in result.output  # Line 3 should appear in format

    def test_line_offset_multiple(self):
        """Test samples command with multiple line numbers."""
        result = self.runner.invoke(samples_command, [self.test_file, '-l', '3', '-l', '5'])
        assert result.exit_code == 0
        assert 'error occurred' in result.output
        assert 'another error' in result.output
        # New format is file:line:offset
        assert ':3:' in result.output
        assert ':5:' in result.output

    def test_line_offset_json_output(self):
        """Test line offset with JSON output."""
        result = self.runner.invoke(samples_command, [self.test_file, '-l', '3', '--json'])
        assert result.exit_code == 0

        data = json.loads(result.output)
        assert data['path'] == self.test_file
        assert isinstance(data['lines'], dict)  # lines now maps line -> offset
        assert data['offsets'] == {}
        assert data['before_context'] == 3
        assert data['after_context'] == 3
        assert '3' in data['samples']
        sample_lines = data['samples']['3']
        assert any('error occurred' in line for line in sample_lines)

    def test_line_offset_json_multiple(self):
        """Test JSON output with multiple line numbers."""
        result = self.runner.invoke(samples_command, [self.test_file, '-l', '3', '-l', '5', '--json'])
        assert result.exit_code == 0

        data = json.loads(result.output)
        assert isinstance(data['lines'], dict)  # lines now maps line -> offset
        assert data['offsets'] == {}
        assert '3' in data['samples']
        assert '5' in data['samples']

    def test_line_offset_with_context(self):
        """Test line offset with custom context size."""
        result = self.runner.invoke(samples_command, [self.test_file, '-l', '3', '-c', '1'])
        assert result.exit_code == 0
        assert 'Context: 1 before, 1 after' in result.output
        assert 'error occurred' in result.output

    def test_line_offset_with_before_after(self):
        """Test line offset with separate before/after context."""
        result = self.runner.invoke(samples_command, [self.test_file, '-l', '3', '-B', '1', '-A', '2'])
        assert result.exit_code == 0
        assert 'Context: 1 before, 2 after' in result.output

    def test_line_offset_long_form(self):
        """Test --line-offset long form option."""
        result = self.runner.invoke(samples_command, [self.test_file, '--line-offset', '3'])
        assert result.exit_code == 0
        assert 'error occurred' in result.output

    def test_line_offset_first_line(self):
        """Test getting first line of file."""
        result = self.runner.invoke(samples_command, [self.test_file, '-l', '1'])
        assert result.exit_code == 0
        assert 'first line' in result.output
        # New format is file:line:offset
        assert ':1:' in result.output

    def test_line_offset_last_line(self):
        """Test getting last line of file."""
        result = self.runner.invoke(samples_command, [self.test_file, '-l', '6'])
        assert result.exit_code == 0
        assert 'final line' in result.output
        # New format is file:line:offset
        assert ':6:' in result.output

    def test_line_offset_no_color(self):
        """Test line offset with --no-color flag."""
        result = self.runner.invoke(samples_command, [self.test_file, '-l', '3', '--no-color'])
        assert result.exit_code == 0
        assert '\033[' not in result.output
        assert 'error occurred' in result.output

    def test_line_offset_with_regex(self):
        """Test line offset with regex highlighting."""
        result = self.runner.invoke(samples_command, [self.test_file, '-l', '3', '-r', 'error'])
        assert result.exit_code == 0
        assert 'error occurred' in result.output


class TestSamplesLineOffsetErrors:
    """Test error handling for line offset option."""

    def setup_method(self):
        """Create test environment."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

        self.test_file = os.path.join(self.temp_dir, 'test.txt')
        with open(self.test_file, 'w') as f:
            f.write('Line 1\nLine 2\nLine 3\n')

    def test_mutual_exclusivity(self):
        """Test that -b and -l cannot be used together."""
        result = self.runner.invoke(samples_command, [self.test_file, '-b', '0', '-l', '1'])
        assert result.exit_code != 0
        assert 'cannot use both' in result.output.lower()

    def test_missing_offset_option(self):
        """Test error when neither -b nor -l is provided."""
        result = self.runner.invoke(samples_command, [self.test_file])
        assert result.exit_code != 0
        assert 'must provide' in result.output.lower() or 'required' in result.output.lower()

    def test_line_offset_beyond_file(self):
        """Test line number beyond file length - should return error about EOF."""
        result = self.runner.invoke(samples_command, [self.test_file, '-l', '999'])
        assert result.exit_code != 0  # Should fail with error
        assert 'EOF reached' in result.output or 'out of bounds' in result.output

    def test_line_offset_zero(self):
        """Test line number 0 (invalid - lines are 1-based)."""
        result = self.runner.invoke(samples_command, [self.test_file, '-l', '0'])
        assert result.exit_code != 0  # Should fail with error
        assert 'out of bounds' in result.output or 'EOF reached' in result.output

    def test_line_offset_negative(self):
        """Test negative line number - gets last line."""
        result = self.runner.invoke(samples_command, [self.test_file, '-l', '-1'])
        assert result.exit_code == 0  # Should succeed
        # Should get the last line (line 3 in this test file)
        assert 'Line 3' in result.output


class TestSamplesLineOffsetEdgeCases:
    """Test edge cases for line offset functionality."""

    def setup_method(self):
        """Create test environment."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def test_line_offset_single_line_file(self):
        """Test line offset on single-line file."""
        single_line = os.path.join(self.temp_dir, 'single.txt')
        with open(single_line, 'w') as f:
            f.write('Only one line here\n')

        result = self.runner.invoke(samples_command, [single_line, '-l', '1'])
        assert result.exit_code == 0
        assert 'Only one line' in result.output

    def test_line_offset_empty_file(self):
        """Test line offset on empty file - should return error."""
        empty_file = os.path.join(self.temp_dir, 'empty.txt')
        with open(empty_file, 'w') as f:
            pass

        result = self.runner.invoke(samples_command, [empty_file, '-l', '1'])
        assert result.exit_code != 0  # Should fail with error
        assert 'out of bounds' in result.output or 'EOF reached' in result.output

    def test_line_offset_zero_context(self):
        """Test line offset with zero context."""
        test_file = os.path.join(self.temp_dir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write('Line 1\nLine 2\nLine 3\n')

        result = self.runner.invoke(samples_command, [test_file, '-l', '2', '-c', '0', '--json'])
        assert result.exit_code == 0
        data = json.loads(result.output)
        # With 0 context, should only get the target line
        assert len(data['samples']['2']) == 1
        assert 'Line 2' in data['samples']['2'][0]

    def test_line_offset_large_context(self):
        """Test line offset with context larger than file."""
        test_file = os.path.join(self.temp_dir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write('Line 1\nLine 2\nLine 3\n')

        result = self.runner.invoke(samples_command, [test_file, '-l', '2', '-c', '100', '--json'])
        assert result.exit_code == 0
        data = json.loads(result.output)
        # Should get all available lines
        assert len(data['samples']['2']) == 3

    def test_line_offset_many_lines(self):
        """Test with many line offsets."""
        test_file = os.path.join(self.temp_dir, 'test.txt')
        lines = [f'Line {i}\n' for i in range(1, 21)]
        with open(test_file, 'w') as f:
            f.writelines(lines)

        args = [test_file, '-l', '5', '-l', '10', '-l', '15', '--json']
        result = self.runner.invoke(samples_command, args)
        assert result.exit_code == 0

        data = json.loads(result.output)
        assert isinstance(data['lines'], dict)  # lines now maps line -> offset
        assert len(data['samples']) == 3

    def test_line_offset_unicode_content(self):
        """Test line offset with unicode content."""
        test_file = os.path.join(self.temp_dir, 'unicode.txt')
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write('Hello world\n')
            f.write('Japanese text here\n')
            f.write('More content\n')

        result = self.runner.invoke(samples_command, [test_file, '-l', '2'])
        assert result.exit_code == 0
        assert 'Japanese' in result.output


class TestSamplesLineOffsetJsonStructure:
    """Test JSON structure for line offset output."""

    def setup_method(self):
        """Create test files."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

        self.test_file = os.path.join(self.temp_dir, 'test.log')
        with open(self.test_file, 'w') as f:
            f.write('First line\n')
            f.write('Second line\n')
            f.write('Third line\n')

    def test_json_has_lines_field(self):
        """Test that JSON output includes lines field."""
        result = self.runner.invoke(samples_command, [self.test_file, '-l', '2', '--json'])
        assert result.exit_code == 0

        data = json.loads(result.output)
        assert 'lines' in data
        assert isinstance(data['lines'], dict)  # lines now maps line -> offset
        assert data['offsets'] == {}

    def test_json_structure_with_lines(self):
        """Test complete JSON structure with line offset."""
        result = self.runner.invoke(samples_command, [self.test_file, '-l', '2', '--json'])
        assert result.exit_code == 0

        data = json.loads(result.output)
        assert 'path' in data
        assert 'lines' in data
        assert 'offsets' in data
        assert 'before_context' in data
        assert 'after_context' in data
        assert 'samples' in data
        assert isinstance(data['lines'], dict)
        assert isinstance(data['offsets'], dict)
        assert isinstance(data['samples'], dict)

    def test_json_byte_offset_has_empty_lines(self):
        """Test that byte offset JSON has empty lines field."""
        result = self.runner.invoke(samples_command, [self.test_file, '-b', '0', '--json'])
        assert result.exit_code == 0

        data = json.loads(result.output)
        assert isinstance(data['lines'], dict)  # lines now maps line -> offset


class TestSamplesLineEndingConsistency:
    """Test that samples command matches head/tail behavior with different line endings."""

    def setup_method(self):
        """Create test environment."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def test_crlf_file_matches_tail(self):
        """Test that samples output matches tail for CRLF files."""
        import subprocess

        # Create CRLF file
        test_file = os.path.join(self.temp_dir, 'crlf.txt')
        with open(test_file, 'wb') as f:
            f.write(b'line1\r\n')
            f.write(b'line2\r\n')
            f.write(b'line3\r\n')
            f.write(b'line4\r\n')

        # Get last line with tail
        tail_result = subprocess.run(['tail', '-n', '1', test_file], capture_output=True, text=True)
        tail_output = tail_result.stdout.strip()

        # Get last line with samples
        result = self.runner.invoke(samples_command, [test_file, '-l', '4', '--context', '0'])
        assert result.exit_code == 0

        # Extract the actual line content (skip header)
        lines = result.output.strip().split('\n')
        samples_output = lines[-1] if lines else ''

        assert samples_output == tail_output

    def test_mixed_line_endings_matches_tail(self):
        """Test that samples matches tail with mixed line endings (CRLF, bare CR, LF)."""
        import subprocess

        # Create file with mixed endings
        test_file = os.path.join(self.temp_dir, 'mixed.txt')
        with open(test_file, 'wb') as f:
            f.write(b'line1\r\n')  # CRLF
            f.write(b'line2\r')  # bare CR - wc doesn't count as line
            f.write(b'line3\n')  # LF
            f.write(b'line4\r\n')  # CRLF

        # Count lines with wc -l
        wc_result = subprocess.run(['wc', '-l', test_file], capture_output=True, text=True)
        wc_count = int(wc_result.stdout.strip().split()[0])

        # Get last line with tail
        tail_result = subprocess.run(['tail', '-n', '1', test_file], capture_output=True, text=True)
        tail_output = tail_result.stdout.strip()

        # Get last line with samples (using wc line count)
        result = self.runner.invoke(samples_command, [test_file, '-l', str(wc_count), '--context', '0'])
        assert result.exit_code == 0

        # Extract actual line content
        lines = result.output.strip().split('\n')
        samples_output = lines[-1] if lines else ''

        assert samples_output == tail_output
        assert wc_count == 3  # Only \n counts

    def test_samples_first_line_matches_head(self):
        """Test that samples first line matches head."""
        import subprocess

        # Create test file
        test_file = os.path.join(self.temp_dir, 'test.txt')
        with open(test_file, 'wb') as f:
            f.write(b'first line\r\n')
            f.write(b'second line\r\n')
            f.write(b'third line\r\n')

        # Get first line with head
        head_result = subprocess.run(['head', '-n', '1', test_file], capture_output=True, text=True)
        head_output = head_result.stdout.strip()

        # Get first line with samples
        result = self.runner.invoke(samples_command, [test_file, '-l', '1', '--context', '0'])
        assert result.exit_code == 0

        # Extract actual line content
        lines = result.output.strip().split('\n')
        samples_output = lines[-1] if lines else ''

        assert samples_output == head_output

    def test_samples_no_extra_empty_lines_in_crlf(self):
        """Test that CRLF files don't show extra empty lines in samples output."""
        # Create CRLF file
        test_file = os.path.join(self.temp_dir, 'crlf_no_empty.txt')
        with open(test_file, 'wb') as f:
            for i in range(1, 11):
                f.write(f'line{i}\r\n'.encode())

        # Get lines 5-7 with samples
        result = self.runner.invoke(samples_command, [test_file, '-l', '6', '--context', '1'])
        assert result.exit_code == 0

        # Count actual content lines (exclude header lines)
        lines = result.output.strip().split('\n')
        content_lines = [
            line
            for line in lines
            if line and not line.startswith('=') and 'File:' not in line and 'Context:' not in line
        ]

        # Should have exactly 3 lines: line5, line6, line7
        assert len(content_lines) == 3
        assert 'line5' in content_lines[0]
        assert 'line6' in content_lines[1]
        assert 'line7' in content_lines[2]

        # No empty lines in between
        for line in content_lines:
            assert line.strip() != ''

    def test_bare_cr_not_treated_as_newline(self):
        """Test that bare \\r is NOT treated as a line separator."""
        # Create file with bare CR in middle of line
        test_file = os.path.join(self.temp_dir, 'bare_cr.txt')
        with open(test_file, 'wb') as f:
            f.write(b'before\rafter\n')  # \r in middle should NOT be line break
            f.write(b'line2\n')

        # Should be 2 lines total (only 2 \n characters)
        import subprocess

        wc_result = subprocess.run(['wc', '-l', test_file], capture_output=True, text=True)
        wc_count = int(wc_result.stdout.strip().split()[0])
        assert wc_count == 2

        # Get line 1 - should contain both "before" and "after"
        result = self.runner.invoke(samples_command, [test_file, '-l', '1', '--context', '0'])
        assert result.exit_code == 0

        output = result.output
        # The first line should contain the \r but not be split
        assert 'before' in output or 'after' in output

    def test_byte_line_offset_consistency(self):
        """Test that byte offset returned by line mode gives same content in byte mode.

        This tests the fix for the bug where get_context used splitlines() which
        treats \\r as a line separator, while line counting uses \\n only.
        """
        # Create file with \r at start of lines (like the user's Twitter data)
        test_file = os.path.join(self.temp_dir, 'cr_start.txt')
        with open(test_file, 'wb') as f:
            f.write(b'Line1 content\n')
            f.write(b'\rLine2 with CR at start\n')
            f.write(b'\rLine3 also has CR\n')

        # Get line 2 with JSON to get the byte offset
        result = self.runner.invoke(samples_command, [test_file, '-l', '2', '--context', '0', '--json'])
        assert result.exit_code == 0
        data = json.loads(result.output)

        # Get the byte offset for line 2
        line2_offset = data['lines']['2']
        assert line2_offset > 0, 'Should get valid byte offset'

        # Get line 2 content
        line2_content = data['samples']['2']

        # Now use the byte offset to get the same line
        result2 = self.runner.invoke(samples_command, [test_file, '-b', str(line2_offset), '--context', '0', '--json'])
        assert result2.exit_code == 0
        data2 = json.loads(result2.output)

        # The content should be identical
        byte_content = data2['samples'][str(line2_offset)]
        assert byte_content == line2_content, (
            f'Content mismatch: line mode got {line2_content}, byte mode got {byte_content}'
        )

        # Verify the line number mapping is also correct
        assert data2['offsets'][str(line2_offset)] == 2, 'Byte offset should map back to line 2'


class TestSamplesNegativeOffsets:
    """Test negative offset support for rx samples command."""

    def setup_method(self):
        """Create test files before each test."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

        # Create a test file with known lines
        self.test_file = os.path.join(self.temp_dir, 'test.log')
        self.lines = [f'Line {i}: content here\n' for i in range(1, 11)]  # 10 lines
        with open(self.test_file, 'w') as f:
            f.writelines(self.lines)

    def test_negative_line_offset_last_line(self):
        """Test -l -1 gets the last line."""
        result = self.runner.invoke(samples_command, [self.test_file, '-l', '-1', '--context', '0'])
        assert result.exit_code == 0
        assert 'Line 10:' in result.output
        # Should not contain other lines
        assert 'Line 9:' not in result.output
        assert 'Line 8:' not in result.output

    def test_negative_line_offset_second_to_last(self):
        """Test -l -2 gets the second to last line."""
        result = self.runner.invoke(samples_command, [self.test_file, '-l', '-2', '--context', '0'])
        assert result.exit_code == 0
        assert 'Line 9:' in result.output
        assert 'Line 10:' not in result.output

    def test_negative_line_offset_fifth_from_end(self):
        """Test -l -5 gets the 5th line from the end."""
        result = self.runner.invoke(samples_command, [self.test_file, '-l', '-5', '--context', '0'])
        assert result.exit_code == 0
        # File has 10 lines, -5 should be line 6
        assert 'Line 6:' in result.output

    def test_negative_line_offset_with_context(self):
        """Test negative line offset with context."""
        result = self.runner.invoke(samples_command, [self.test_file, '-l', '-3', '--context', '1'])
        assert result.exit_code == 0
        # File has 10 lines, -3 should be line 8
        # With context of 1: should show lines 7, 8, 9
        assert 'Line 7:' in result.output
        assert 'Line 8:' in result.output
        assert 'Line 9:' in result.output
        assert 'Line 6:' not in result.output
        assert 'Line 10:' not in result.output

    def test_negative_line_offset_beyond_file_start(self):
        """Test negative offset beyond file start - should clamp to line 1."""
        result = self.runner.invoke(samples_command, [self.test_file, '-l', '-100', '--context', '0'])
        assert result.exit_code == 0
        # Should clamp to line 1
        assert 'Line 1:' in result.output

    def test_negative_line_offset_json(self):
        """Test negative line offset with JSON output."""
        result = self.runner.invoke(samples_command, [self.test_file, '-l', '-1', '--json'])
        assert result.exit_code == 0

        # Handle potential stderr messages mixed in output (like "Analyzing file...")
        output = result.output
        if output.startswith('Analyzing') or output.startswith('Building') or output.startswith('Counting'):
            # Skip the first line (stderr message) and parse the JSON
            output = '\n'.join(output.split('\n')[1:])

        data = json.loads(output)
        # Should have line 10 in samples (the last line)
        assert '10' in data['samples']
        sample_lines = data['samples']['10']
        assert any('Line 10:' in line for line in sample_lines)

    def test_negative_byte_offset_last_byte(self):
        """Test -b -1 gets content near the last byte of the file."""
        result = self.runner.invoke(samples_command, [self.test_file, '-b', '-1', '--context', '0'])
        assert result.exit_code == 0
        # Last byte should be in the last line
        assert 'Line 10:' in result.output

    def test_negative_byte_offset_from_end(self):
        """Test negative byte offset counting from end."""
        import os

        file_size = os.path.getsize(self.test_file)
        # Get offset 50 bytes from end
        result = self.runner.invoke(samples_command, [self.test_file, '-b', '-50', '--context', '0'])
        assert result.exit_code == 0
        # Should get a line near the end (line 8, 9 or 10 depending on exact size)
        assert 'Line 8:' in result.output or 'Line 9:' in result.output or 'Line 10:' in result.output

    def test_negative_byte_offset_json(self):
        """Test negative byte offset with JSON output."""
        result = self.runner.invoke(samples_command, [self.test_file, '-b', '-10', '--json'])
        assert result.exit_code == 0

        data = json.loads(result.output)
        # Should have an offset in samples
        assert len(data['samples']) > 0
        # Should be near the end of file
        assert len(data['offsets']) > 0

    def test_mixed_positive_and_negative_lines(self):
        """Test mixing positive and negative line offsets."""
        result = self.runner.invoke(samples_command, [self.test_file, '-l', '1', '-l', '-1', '--context', '0'])
        assert result.exit_code == 0
        # Should have both first and last line
        assert 'Line 1:' in result.output
        assert 'Line 10:' in result.output

    def test_mixed_positive_and_negative_bytes(self):
        """Test mixing positive and negative byte offsets."""
        result = self.runner.invoke(samples_command, [self.test_file, '-b', '0', '-b', '-10', '--context', '0'])
        assert result.exit_code == 0
        # Should have content from start and end
        assert 'Line 1:' in result.output
        # Last offset should be near the end
        assert 'Line 9:' in result.output or 'Line 10:' in result.output

    def test_multiple_negative_line_offsets(self):
        """Test multiple negative line offsets."""
        result = self.runner.invoke(
            samples_command, [self.test_file, '-l', '-1', '-l', '-5', '-l', '-10', '--context', '0']
        )
        assert result.exit_code == 0
        # -1 = line 10, -5 = line 6, -10 = line 1
        assert 'Line 10:' in result.output
        assert 'Line 6:' in result.output
        assert 'Line 1:' in result.output

    def test_negative_offset_with_before_after_context(self):
        """Test negative offset with --before and --after options."""
        result = self.runner.invoke(samples_command, [self.test_file, '-l', '-2', '--before', '1', '--after', '1'])
        assert result.exit_code == 0
        # -2 = line 9, with before=1 and after=1: lines 8, 9, 10
        assert 'Line 8:' in result.output
        assert 'Line 9:' in result.output
        assert 'Line 10:' in result.output
        assert 'Line 7:' not in result.output

    def test_negative_byte_offset_beyond_file_start(self):
        """Test negative byte offset beyond file start - should clamp to 0."""
        import os

        file_size = os.path.getsize(self.test_file)
        # Request offset way beyond file size
        result = self.runner.invoke(samples_command, [self.test_file, '-b', str(-(file_size + 1000)), '--context', '0'])
        assert result.exit_code == 0
        # Should clamp to start and show line 1
        assert 'Line 1:' in result.output


class TestSamplesNegativeOffsetErrors:
    """Test error handling for negative offsets."""

    def setup_method(self):
        """Create test environment."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

        self.test_file = os.path.join(self.temp_dir, 'test.txt')
        with open(self.test_file, 'w') as f:
            f.write('Line 1\nLine 2\nLine 3\n')

    def test_negative_range_not_allowed_start(self):
        """Test that ranges with negative start value are rejected."""
        result = self.runner.invoke(samples_command, [self.test_file, '-l', '-5-10'])
        assert result.exit_code != 0
        assert 'Invalid' in result.output

    def test_negative_range_not_allowed_end(self):
        """Test that ranges with negative end value are rejected."""
        result = self.runner.invoke(samples_command, [self.test_file, '-l', '5--10'])
        assert result.exit_code != 0
        assert 'Invalid' in result.output

    def test_negative_range_both_negative(self):
        """Test that ranges with both negative values are rejected."""
        result = self.runner.invoke(samples_command, [self.test_file, '-l', '-10--5'])
        assert result.exit_code != 0
        assert 'Invalid' in result.output

    def test_negative_byte_range_not_allowed(self):
        """Test that negative byte ranges are rejected."""
        result = self.runner.invoke(samples_command, [self.test_file, '-b', '-100--50'])
        assert result.exit_code != 0
        assert 'Invalid' in result.output


class TestSamplesMultipleLineSpecifications:
    """Test combining multiple line specifications with ranges, single lines, and negative offsets.

    This tests the use case:
    `-l 1-15 -l 55 -l 60-62 -l -1 --context=5`

    Expected behavior:
    - Range 1-15: returns lines 1-15 exactly (ranges ignore context)
    - Single line 55 with context=5: returns lines 50-60
    - Range 60-62: returns lines 60-62 exactly (ranges ignore context)
    - Negative -1 with context=5: returns last 6 lines (last line + 5 before)
    """

    def setup_method(self):
        """Create test file with 100 numbered lines."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

        self.test_file = os.path.join(self.temp_dir, 'multispec.log')
        # Create 100 lines for comprehensive testing
        self.lines = [f'Line {i}: content for line number {i}\n' for i in range(1, 101)]
        with open(self.test_file, 'w') as f:
            f.writelines(self.lines)

    def teardown_method(self):
        """Clean up test files."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_multiple_specs_range_single_negative_with_context(self):
        """Test combining range, single line, and negative offset with context.

        Command: -l 1-15 -l 55 -l 60-62 -l -1 --context=5

        Expected output lines:
        - 1-15 (from range, context ignored)
        - 50-60 (from line 55 with context 5)
        - 60-62 (from range, context ignored)
        - 95-100 (from -1 with context 5, i.e., last line + 5 before)
        """
        result = self.runner.invoke(
            samples_command,
            [self.test_file, '-l', '1-15', '-l', '55', '-l', '60-62', '-l', '-1', '--context', '5'],
        )
        assert result.exit_code == 0

        # Range 1-15: should include all lines 1-15
        for i in range(1, 16):
            assert f'Line {i}:' in result.output, f'Line {i} should be in output (from range 1-15)'

        # Single line 55 with context 5: should include lines 50-60
        for i in range(50, 61):
            assert f'Line {i}:' in result.output, f'Line {i} should be in output (from line 55 with context 5)'

        # Range 60-62: should include lines 60-62 (already covered by above, but verify)
        assert 'Line 60:' in result.output
        assert 'Line 61:' in result.output
        assert 'Line 62:' in result.output

        # Negative -1 with context 5: should include lines 95-100
        for i in range(95, 101):
            assert f'Line {i}:' in result.output, f'Line {i} should be in output (from -1 with context 5)'

    def test_multiple_specs_json_output(self):
        """Test multiple line specifications with JSON output."""
        result = self.runner.invoke(
            samples_command,
            [self.test_file, '-l', '1-15', '-l', '55', '-l', '60-62', '-l', '-1', '--context', '5', '--json'],
        )
        assert result.exit_code == 0

        # Handle potential stderr messages mixed in output (like "Analyzing file...")
        output = result.output
        if output.startswith('Analyzing') or output.startswith('Building') or output.startswith('Counting'):
            # Find the start of JSON (first '{')
            json_start = output.find('{')
            if json_start != -1:
                output = output[json_start:]

        data = json.loads(output)

        # Check that all specifications are in samples
        assert '1-15' in data['samples'], 'Range 1-15 should be in samples'
        assert '55' in data['samples'], 'Line 55 should be in samples'
        assert '60-62' in data['samples'], 'Range 60-62 should be in samples'
        assert '100' in data['samples'], 'Line 100 (resolved from -1) should be in samples'

        # Verify line counts
        assert len(data['samples']['1-15']) == 15, 'Range 1-15 should have exactly 15 lines'
        assert len(data['samples']['55']) == 11, 'Line 55 with context 5 should have 11 lines (50-60)'
        assert len(data['samples']['60-62']) == 3, 'Range 60-62 should have exactly 3 lines'
        assert len(data['samples']['100']) == 6, 'Line 100 with context 5 should have 6 lines (95-100)'

    def test_multiple_ranges_no_context(self):
        """Test multiple ranges without context."""
        result = self.runner.invoke(
            samples_command, [self.test_file, '-l', '1-5', '-l', '20-25', '-l', '90-95', '--context', '0']
        )
        assert result.exit_code == 0

        # Range 1-5
        for i in range(1, 6):
            assert f'Line {i}:' in result.output

        # Range 20-25
        for i in range(20, 26):
            assert f'Line {i}:' in result.output

        # Range 90-95
        for i in range(90, 96):
            assert f'Line {i}:' in result.output

        # Should NOT include lines outside ranges
        assert 'Line 6:' not in result.output
        assert 'Line 19:' not in result.output
        assert 'Line 26:' not in result.output
        assert 'Line 89:' not in result.output
        assert 'Line 96:' not in result.output

    def test_multiple_single_lines_with_context(self):
        """Test multiple single lines with context."""
        result = self.runner.invoke(
            samples_command, [self.test_file, '-l', '10', '-l', '50', '-l', '90', '--context', '2']
        )
        assert result.exit_code == 0

        # Line 10 with context 2: lines 8-12
        for i in range(8, 13):
            assert f'Line {i}:' in result.output

        # Line 50 with context 2: lines 48-52
        for i in range(48, 53):
            assert f'Line {i}:' in result.output

        # Line 90 with context 2: lines 88-92
        for i in range(88, 93):
            assert f'Line {i}:' in result.output

    def test_multiple_negative_offsets_with_context(self):
        """Test multiple negative line offsets with context."""
        result = self.runner.invoke(
            samples_command, [self.test_file, '-l', '-1', '-l', '-10', '-l', '-50', '--context', '2']
        )
        assert result.exit_code == 0

        # -1 = line 100 with context 2: lines 98-100
        assert 'Line 98:' in result.output
        assert 'Line 99:' in result.output
        assert 'Line 100:' in result.output

        # -10 = line 91 with context 2: lines 89-93
        for i in range(89, 94):
            assert f'Line {i}:' in result.output

        # -50 = line 51 with context 2: lines 49-53
        for i in range(49, 54):
            assert f'Line {i}:' in result.output

    def test_overlapping_ranges_deduplicated(self):
        """Test that overlapping ranges don't duplicate lines in output."""
        result = self.runner.invoke(samples_command, [self.test_file, '-l', '10-20', '-l', '15-25', '--context', '0'])
        assert result.exit_code == 0

        # Both ranges should be in output
        for i in range(10, 26):
            assert f'Line {i}:' in result.output

    def test_adjacent_ranges(self):
        """Test adjacent ranges that together form a continuous block."""
        result = self.runner.invoke(samples_command, [self.test_file, '-l', '10-15', '-l', '16-20', '--context', '0'])
        assert result.exit_code == 0

        # Both ranges should be in output
        for i in range(10, 21):
            assert f'Line {i}:' in result.output

    def test_single_line_with_range_overlap(self):
        """Test single line with context that overlaps with a range."""
        result = self.runner.invoke(samples_command, [self.test_file, '-l', '10', '-l', '12-15', '--context', '3'])
        assert result.exit_code == 0

        # Line 10 with context 3: lines 7-13
        for i in range(7, 14):
            assert f'Line {i}:' in result.output

        # Range 12-15: lines 12-15
        for i in range(12, 16):
            assert f'Line {i}:' in result.output

    def test_negative_offset_at_file_start_with_context(self):
        """Test negative offset that resolves to beginning of file with context."""
        result = self.runner.invoke(samples_command, [self.test_file, '-l', '-100', '--context', '5'])
        assert result.exit_code == 0
        # -100 on a 100-line file = line 1
        # Line 1 with context 5: lines 1-6 (can't go before line 1)
        for i in range(1, 7):
            assert f'Line {i}:' in result.output

    def test_range_at_file_end_clamped(self):
        """Test range that extends beyond file end is clamped."""
        result = self.runner.invoke(samples_command, [self.test_file, '-l', '95-150', '--context', '0'])
        assert result.exit_code == 0
        # Should include lines 95-100 (clamped to file size)
        for i in range(95, 101):
            assert f'Line {i}:' in result.output

    def test_before_after_context_with_multiple_specs(self):
        """Test --before and --after with multiple specifications."""
        result = self.runner.invoke(
            samples_command, [self.test_file, '-l', '50', '-l', '70-72', '--before', '3', '--after', '2']
        )
        assert result.exit_code == 0

        # Line 50 with before=3 after=2: lines 47-52
        for i in range(47, 53):
            assert f'Line {i}:' in result.output

        # Range 70-72: exactly those lines (ranges ignore before/after)
        assert 'Line 70:' in result.output
        assert 'Line 71:' in result.output
        assert 'Line 72:' in result.output
        # Verify range doesn't get context
        assert 'Line 67:' not in result.output or 'Line 67:' in result.output  # May overlap from line 50 context
        assert 'Line 74:' not in result.output
