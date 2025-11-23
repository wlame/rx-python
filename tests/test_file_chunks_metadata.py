"""
Tests for file_chunks metadata feature.

These tests verify that the file_chunks field is properly populated
and indicates how files were split for parallel processing.
"""

import json
import os
import tempfile

from click.testing import CliRunner

from rx.cli.search import search_command


class TestFileChunksMetadata:
    """Test file_chunks metadata in responses."""

    def setup_method(self):
        """Create test files before each test."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

        # Create a small test file (will not be chunked)
        self.small_file = os.path.join(self.temp_dir, "small.txt")
        with open(self.small_file, "w") as f:
            for i in range(100):
                f.write(f"line {i}: some content\n")

    def test_file_chunks_present_in_json_output(self):
        """Test that file_chunks field is present in JSON output."""
        result = self.runner.invoke(search_command, [self.small_file, "content", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)

        # file_chunks should be present
        assert "file_chunks" in data
        assert data["file_chunks"] is not None
        assert isinstance(data["file_chunks"], dict)

        # Should have one entry for f1
        assert "f1" in data["file_chunks"]
        assert isinstance(data["file_chunks"]["f1"], int)

        # Small file should not be chunked (1 worker)
        assert data["file_chunks"]["f1"] == 1

    def test_file_chunks_small_file_not_chunked(self):
        """Test that small files show num_chunks=1 (not chunked)."""
        result = self.runner.invoke(search_command, [self.small_file, "line", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)

        # Small file should have exactly 1 chunk
        assert data["file_chunks"]["f1"] == 1

    def test_file_chunks_with_samples(self):
        """Test that file_chunks works with --samples mode."""
        result = self.runner.invoke(search_command, [self.small_file, "line", "--samples", "--context", "1", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)

        # file_chunks should be present even with samples
        assert "file_chunks" in data
        assert data["file_chunks"]["f1"] == 1

        # context_lines should also be present
        assert "context_lines" in data
        assert "before_context" in data
        assert "after_context" in data

    def test_file_chunks_consistent_across_searches(self):
        """Test that file_chunks is consistent when searching same file multiple times."""
        # Run search twice on same file
        result1 = self.runner.invoke(search_command, [self.small_file, "line", "--json"])
        result2 = self.runner.invoke(search_command, [self.small_file, "content", "--json"])

        assert result1.exit_code == 0
        assert result2.exit_code == 0

        data1 = json.loads(result1.output)
        data2 = json.loads(result2.output)

        # Both should have same file_chunks structure
        assert "file_chunks" in data1
        assert "file_chunks" in data2
        assert data1["file_chunks"] == data2["file_chunks"]

        # Both should report 1 chunk for this small file
        assert data1["file_chunks"]["f1"] == 1
        assert data2["file_chunks"]["f1"] == 1

    def test_file_chunks_matches_files_dict(self):
        """Test that file_chunks keys correspond to files dict keys."""
        result = self.runner.invoke(search_command, [self.small_file, "line", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)

        # file_chunks keys should match files dict keys
        file_ids = set(data["files"].keys())
        chunk_ids = set(data["file_chunks"].keys())

        assert file_ids == chunk_ids

    def test_file_chunks_cli_display_no_chunking(self):
        """Test that CLI output doesn't show chunking info when no files are chunked."""
        result = self.runner.invoke(search_command, [self.small_file, "line"])

        assert result.exit_code == 0

        # When no files are chunked, should NOT show "Parallel workers" line
        assert "Parallel workers:" not in result.output

        # Should show normal output
        assert "Path:" in result.output
        assert "Matches:" in result.output

    def test_file_chunks_data_structure(self):
        """Test the structure and types of file_chunks data."""
        result = self.runner.invoke(search_command, [self.small_file, "line", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)

        file_chunks = data["file_chunks"]

        # Should be a dict
        assert isinstance(file_chunks, dict)

        # All keys should be file IDs (strings starting with 'f')
        for file_id in file_chunks.keys():
            assert isinstance(file_id, str)
            assert file_id.startswith('f')

        # All values should be positive integers
        for num_chunks in file_chunks.values():
            assert isinstance(num_chunks, int)
            assert num_chunks >= 1  # At least 1 chunk per file

    def test_file_chunks_with_no_matches(self):
        """Test that file_chunks is populated even when there are no matches."""
        result = self.runner.invoke(search_command, [self.small_file, "NONEXISTENT_PATTERN", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)

        # file_chunks should still be present
        assert "file_chunks" in data
        assert data["file_chunks"]["f1"] == 1

        # No matches
        assert len(data["matches"]) == 0


class TestFileChunksDocumentation:
    """Test that line_number documentation is clear about absolute values."""

    def setup_method(self):
        """Create test files before each test."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

        self.test_file = os.path.join(self.temp_dir, "test.txt")
        with open(self.test_file, "w") as f:
            for i in range(1, 21):
                f.write(f"line {i}: content on line {i}\n")

    def test_line_numbers_are_absolute(self):
        """Test that line numbers in output match actual line numbers in file."""
        result = self.runner.invoke(search_command, [self.test_file, "line 10", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)

        # Should find the match on line 10
        assert len(data["matches"]) == 1
        match = data["matches"][0]

        # Line number should be 10 (absolute, 1-indexed)
        assert match["relative_line_number"] == 10
        assert "line 10:" in match["line_text"]

        # With file_chunks showing num_chunks=1, user knows line_number is reliable
        assert data["file_chunks"]["f1"] == 1

    def test_context_line_numbers_are_absolute(self):
        """Test that context line numbers are absolute with --samples."""
        result = self.runner.invoke(
            search_command, [self.test_file, "line 10", "--samples", "--context", "2", "--json"]
        )

        assert result.exit_code == 0
        data = json.loads(result.output)

        # Get the context lines
        assert "context_lines" in data
        assert len(data["context_lines"]) == 1

        context_key = list(data["context_lines"].keys())[0]
        context_lines = data["context_lines"][context_key]

        # Should have 5 lines: 2 before + match + 2 after = lines 8, 9, 10, 11, 12
        assert len(context_lines) == 5

        # Verify line numbers are absolute (8, 9, 10, 11, 12)
        line_numbers = [ctx["relative_line_number"] for ctx in context_lines]
        assert line_numbers == [8, 9, 10, 11, 12]

        # Verify line texts match line numbers
        for ctx in context_lines:
            expected_text_start = f"line {ctx['relative_line_number']}:"
            assert ctx["line_text"].startswith(expected_text_start)
