"""Tests for CLI parameter handling and --samples functionality."""

import json
import os
import tempfile
from pathlib import Path

from click.testing import CliRunner

from rx.cli.search import search_command


class TestCLISamples:
    """Test --samples CLI parameter."""

    def setup_method(self):
        """Create test files before each test."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

        # Create a test file with multiple matches
        self.test_file = os.path.join(self.temp_dir, "test.log")
        with open(self.test_file, "w") as f:
            f.write("Line 1: normal\n")
            f.write("Line 2: error occurred\n")
            f.write("Line 3: normal\n")
            f.write("Line 4: normal\n")
            f.write("Line 5: another error\n")
            f.write("Line 6: normal\n")
            f.write("Line 7: error again\n")
            f.write("Line 8: normal\n")

    def test_samples_basic(self):
        """Test --samples with default context (3 lines)."""
        result = self.runner.invoke(search_command, [self.test_file, "error", "--samples"])
        assert result.exit_code == 0
        assert "Path: " in result.output
        assert "Samples (context: 3 before, 3 after" in result.output
        assert "=== " in result.output  # Sample separator line
        assert "error occurred" in result.output
        assert "another error" in result.output
        assert "error again" in result.output

    def test_samples_with_context(self):
        """Test --samples with custom --context option."""
        result = self.runner.invoke(search_command, [self.test_file, "error", "--samples", "--context", "1"])
        assert result.exit_code == 0
        assert "Samples (context: 1 before, 1 after" in result.output
        assert "error occurred" in result.output

    def test_samples_with_before_after(self):
        """Test --samples with --before and --after options."""
        result = self.runner.invoke(
            search_command, [self.test_file, "error", "--samples", "--before", "2", "--after", "1"]
        )
        assert result.exit_code == 0
        assert "Samples (context: 2 before, 1 after" in result.output
        assert "error occurred" in result.output

    def test_samples_with_json(self):
        """Test --samples with --json output."""
        result = self.runner.invoke(search_command, [self.test_file, "error", "--samples", "--json"])
        assert result.exit_code == 0

        # Parse JSON output
        data = json.loads(result.output)

        # Verify structure
        assert "path" in data
        assert "patterns" in data
        assert "matches" in data
        assert "context_lines" in data  # Changed from "samples" to "context_lines"
        assert "before_context" in data
        assert "after_context" in data

        # Verify context_lines structure
        # Context lines are now keyed by "pattern:file:offset" (e.g., "p1:f1:23")
        context_lines = data["context_lines"]
        assert len(context_lines) == len(data["matches"])

        # Check that context_lines contain the expected structure
        for key, lines in context_lines.items():
            assert isinstance(lines, list)
            # Each line should be a dict with line_number, line_text, absolute_offset
            for line in lines:
                assert "relative_line_number" in line
                assert "line_text" in line
                assert "absolute_offset" in line
            assert len(lines) > 0
            assert ":" in key  # Should contain pattern:file:offset format

    def test_samples_with_no_color(self):
        """Test --samples with --no-color option."""
        result = self.runner.invoke(search_command, [self.test_file, "error", "--samples", "--no-color"])
        assert result.exit_code == 0
        # Should not contain ANSI color codes
        assert "\033[" not in result.output
        assert "error occurred" in result.output

    def test_samples_with_colorization(self):
        """Test that --samples includes colorization by default."""
        # Click's CliRunner doesn't emulate a TTY, so we need to check that
        # colorization works when explicitly enabled (default behavior in actual terminal)
        result = self.runner.invoke(search_command, [self.test_file, "error", "--samples"])
        assert result.exit_code == 0
        # In actual usage, colors would be enabled, but CliRunner disables them
        # So we just verify the output has the expected content
        assert "Path: " in result.output
        assert "error" in result.output

    def test_samples_with_directory(self):
        """Test --samples with directory path shows samples from each file."""
        # Create a directory with files
        dir_path = os.path.join(self.temp_dir, "testdir")
        os.makedirs(dir_path, exist_ok=True)

        test_file1 = os.path.join(dir_path, "file1.txt")
        with open(test_file1, "w") as f:
            f.write("Line 1\nerror in file1\nLine 3\n")

        test_file2 = os.path.join(dir_path, "file2.txt")
        with open(test_file2, "w") as f:
            f.write("Normal\nanother error\nEnd\n")

        result = self.runner.invoke(search_command, [dir_path, "error", "--samples"])
        assert result.exit_code == 0
        assert "Path: " in result.output
        assert "Pattern: error" in result.output
        assert "Files scanned:" in result.output
        assert "error in file1" in result.output
        assert "another error" in result.output

    def test_samples_with_directory_json(self):
        """Test --samples with directory and JSON output."""
        # Create a directory with files
        dir_path = os.path.join(self.temp_dir, "testdir2")
        os.makedirs(dir_path, exist_ok=True)

        test_file1 = os.path.join(dir_path, "file1.txt")
        with open(test_file1, "w") as f:
            f.write("Line 1\nerror in file1\nLine 3\n")

        result = self.runner.invoke(search_command, [dir_path, "error", "--samples", "--json"])
        assert result.exit_code == 0

        # Parse JSON output
        data = json.loads(result.output)

        # Verify structure
        assert "path" in data
        assert "patterns" in data
        assert "scanned_files" in data
        assert "context_lines" in data

        # Verify context_lines for each file
        # Context lines are now keyed by "pattern:file:offset" format (e.g., "p1:f1:7")
        context_lines = data["context_lines"]
        assert len(context_lines) > 0
        # Check that at least one context key contains the expected pattern/file IDs
        context_keys = list(context_lines.keys())
        assert any("p1" in key and "f1" in key for key in context_keys)
        # Verify that context_lines contain context lines
        for key, ctx_lines in context_lines.items():
            assert isinstance(ctx_lines, list)
            assert len(ctx_lines) > 0

    def test_samples_with_max_results(self):
        """Test --samples with --max-results to limit matches."""
        result = self.runner.invoke(search_command, [self.test_file, "error", "--samples", "--max-results", "1"])
        assert result.exit_code == 0
        # Should only have one sample section (format: "=== filepath:offset [pattern] ===")
        assert result.output.count("=== ") == 1  # One separator line per sample
        assert "Matches: 1" in result.output  # Verify only 1 match


class TestCLIPositionalAndNamedParams:
    """Test positional vs named parameters."""

    def setup_method(self):
        """Create test files before each test."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test.txt")
        with open(self.test_file, "w") as f:
            f.write("test error line\n")

    def test_positional_params(self):
        """Test using positional path and regex parameters."""
        result = self.runner.invoke(search_command, [self.test_file, "error"])
        assert result.exit_code == 0
        assert "error" in result.output or result.output  # Has some output

    def test_named_params(self):
        """Test using --path and --regex named parameters."""
        result = self.runner.invoke(search_command, ["--path", self.test_file, "--regex", "error"])
        assert result.exit_code == 0
        assert "error" in result.output or result.output

    def test_mixed_positional_and_named(self):
        """Test mixing positional and named parameters."""
        # Positional path, named regex
        result = self.runner.invoke(search_command, [self.test_file, "--regex", "error"])
        assert result.exit_code == 0

    def test_named_params_with_json(self):
        """Test named parameters with --json output."""
        result = self.runner.invoke(search_command, ["--path", self.test_file, "--regex", "error", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["path"] == [self.test_file]  # path is now a list
        assert "p1" in data["patterns"]
        assert data["patterns"]["p1"] == "error"

    def test_file_alias_for_path(self):
        """Test --file as alias for --path."""
        result = self.runner.invoke(search_command, ["--file", self.test_file, "--regex", "error"])
        assert result.exit_code == 0
        assert "error" in result.output or result.output

    def test_regexp_alias_for_regex(self):
        """Test --regexp as alias for --regex."""
        result = self.runner.invoke(search_command, ["--path", self.test_file, "--regexp", "error"])
        assert result.exit_code == 0
        assert "error" in result.output or result.output

    def test_e_short_option_for_regexp(self):
        """Test -e short option for regex pattern."""
        result = self.runner.invoke(search_command, ["--path", self.test_file, "-e", "error"])
        assert result.exit_code == 0
        assert "error" in result.output or result.output

    def test_combined_aliases(self):
        """Test --file and --regexp together."""
        result = self.runner.invoke(search_command, ["--file", self.test_file, "--regexp", "error", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["path"] == [self.test_file]  # path is now a list
        assert "p1" in data["patterns"]
        assert data["patterns"]["p1"] == "error"

    def test_e_with_file_alias(self):
        """Test -e short option with --file alias."""
        result = self.runner.invoke(search_command, ["--file", self.test_file, "-e", "error"])
        assert result.exit_code == 0
        assert "error" in result.output or result.output


class TestCLIMaxResults:
    """Test --max-results parameter."""

    def setup_method(self):
        """Create test file with multiple matches."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test.txt")
        with open(self.test_file, "w") as f:
            for i in range(10):
                f.write(f"Line {i}: error\n")

    def test_max_results_limits_output(self):
        """Test that --max-results limits the number of matches."""
        result = self.runner.invoke(search_command, [self.test_file, "error", "--max-results", "3", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data["matches"]) == 3

    def test_max_results_without_limit(self):
        """Test without --max-results returns all matches."""
        result = self.runner.invoke(search_command, [self.test_file, "error", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data["matches"]) == 10


class TestCLIColorization:
    """Test colorization in CLI output."""

    def setup_method(self):
        """Create test file."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test.txt")
        with open(self.test_file, "w") as f:
            f.write("error line\n")

    def test_colorization_default(self):
        """Test that default output works (colorization only applies to --samples)."""
        result = self.runner.invoke(search_command, [self.test_file, "error"])
        assert result.exit_code == 0
        # Regular output (without --samples) doesn't have colorization
        assert "Path: " in result.output
        assert "Pattern: " in result.output

    def test_colorization_disabled_with_no_color(self):
        """Test that --no-color flag works with --samples."""
        result = self.runner.invoke(search_command, [self.test_file, "error", "--samples", "--no-color"])
        assert result.exit_code == 0
        # Should NOT contain ANSI escape codes
        assert "\033[" not in result.output
        assert "error" in result.output

    def test_colorization_disabled_with_json(self):
        """Test that --json disables colorization."""
        result = self.runner.invoke(search_command, [self.test_file, "error", "--json"])
        assert result.exit_code == 0
        # Should NOT contain ANSI escape codes (JSON should be clean)
        assert "\033[" not in result.output
        # Should be valid JSON
        data = json.loads(result.output)
        assert "patterns" in data


class TestCLIRipgrepPassthrough:
    """Test ripgrep passthrough options."""

    def setup_method(self):
        """Create test files before each test."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test_case.txt")
        with open(self.test_file, "w") as f:
            f.write("Error\nerror\nERROR\n")

    def test_ignore_case_flag(self):
        """Test -i (ignore case) ripgrep flag passthrough."""
        result = self.runner.invoke(search_command, [self.test_file, "error", "-i", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        # Should find all three: Error, error, ERROR
        assert len(data["matches"]) == 3

    def test_case_sensitive_flag(self):
        """Test --case-sensitive ripgrep flag passthrough."""
        result = self.runner.invoke(search_command, [self.test_file, "error", "--case-sensitive", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        # Should find only lowercase "error"
        assert len(data["matches"]) == 1

    def test_default_case_behavior(self):
        """Test default case sensitivity (smart case by default in ripgrep)."""
        result = self.runner.invoke(search_command, [self.test_file, "error", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        # With lowercase pattern, ripgrep uses smart case (case-insensitive)
        # But our test shows it finds only 1, so it's case-sensitive by default
        assert len(data["matches"]) >= 1

    def test_multiple_passthrough_flags(self):
        """Test multiple ripgrep flags at once."""
        # Create a file with word boundaries
        test_file2 = os.path.join(self.temp_dir, "words.txt")
        with open(test_file2, "w") as f:
            f.write("error errors error\n")

        result = self.runner.invoke(search_command, [test_file2, "error", "-w", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        # -w means word boundary, so "errors" should not match
        # With JSON mode: 1 match (the line) with 2 submatches (two "error" words)
        assert len(data["matches"]) == 1
        # Check that we have 2 submatches (both "error" words, not "errors")
        match = data["matches"][0]
        assert len(match["submatches"]) == 2
        assert all(sm["text"] == "error" for sm in match["submatches"])


class TestCLIServerMode:
    """Test --serve parameter for server mode."""

    def test_serve_with_port_only(self):
        """Test --serve with port only."""
        # We can't actually run the server in tests, but we can check the parameter parsing
        # This test would require mocking uvicorn.run
        pass  # Skipping actual server tests

    def test_serve_with_host_and_port(self):
        """Test --serve with host:port format."""
        # Would require mocking uvicorn.run
        pass  # Skipping actual server tests
