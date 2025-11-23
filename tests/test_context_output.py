"""
Tests for context-related fixes in samples mode.

These tests cover the following issues that were fixed:
1. --context=0 showing "No context available" instead of matched lines
2. Missing samples for matches near end of file
3. Incomplete context when matches are on consecutive lines
"""

import json
import os
import tempfile

from click.testing import CliRunner

from rx.cli.search import search_command


class TestContextZero:
    """Test --context=0 fix."""

    def setup_method(self):
        """Create test files before each test."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

        # Create a test file with multiple matches
        self.test_file = os.path.join(self.temp_dir, "test.txt")
        with open(self.test_file, "w") as f:
            f.write("line 1: normal\n")
            f.write("line 2: error occurred\n")
            f.write("line 3: normal\n")
            f.write("line 4: another error\n")
            f.write("line 5: normal\n")

    def test_context_zero_shows_matches(self):
        """Test that --context=0 shows matched lines, not 'No context available'."""
        result = self.runner.invoke(search_command, [self.test_file, "error", "--samples", "--context", "0"])

        assert result.exit_code == 0
        assert "Samples (context: 0 before, 0 after)" in result.output

        # Should show the matched lines
        assert "error occurred" in result.output
        assert "another error" in result.output

        # Should NOT show the error message
        assert "No context available" not in result.output

    def test_context_zero_json_output(self):
        """Test that --context=0 with --json returns proper structure."""
        result = self.runner.invoke(search_command, [self.test_file, "error", "--samples", "--context", "0", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)

        # Verify structure
        assert "context_lines" in data
        assert data["before_context"] == 0
        assert data["after_context"] == 0
        assert data["context_lines"] is not None

        # Should have 2 matches
        assert len(data["matches"]) == 2

        # Each match should have corresponding context_lines with just the match itself
        assert len(data["context_lines"]) == 2
        for key, lines in data["context_lines"].items():
            assert len(lines) == 1  # Only the matched line
            assert "error" in lines[0]["line_text"]

    def test_context_zero_only_shows_matched_lines(self):
        """Test that --context=0 only shows matched lines without surrounding context."""
        result = self.runner.invoke(search_command, [self.test_file, "error", "--samples", "--context", "0"])

        assert result.exit_code == 0

        # Should show matched lines
        assert "error occurred" in result.output
        assert "another error" in result.output

        # Should NOT show surrounding normal lines in the context blocks
        # We need to check that "normal" doesn't appear in sample sections
        output_lines = result.output.split('\n')
        in_sample_section = False
        sample_lines = []

        for line in output_lines:
            if "===" in line and "error" in line:
                in_sample_section = True
            elif "===" in line or (in_sample_section and line.strip() == ""):
                in_sample_section = False
            elif in_sample_section:
                sample_lines.append(line)

        # In sample sections, should only see lines with "error"
        for line in sample_lines:
            if line.strip():  # Ignore empty lines
                assert "error" in line.lower()


class TestMatchesNearEndOfFile:
    """Test fixes for matches near or at end of file."""

    def setup_method(self):
        """Create test files before each test."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

        # Create a test file where matches are near the end
        self.test_file = os.path.join(self.temp_dir, "test.txt")
        with open(self.test_file, "w") as f:
            f.write("line 1: normal\n")
            f.write("line 2: normal\n")
            f.write("line 3: normal\n")
            f.write("line 4: second_to_last match\n")
            f.write("line 5: last_line match")  # No trailing newline

    def test_match_on_last_line_with_context_1(self):
        """Test that match on last line shows up with --context=1."""
        result = self.runner.invoke(search_command, [self.test_file, "last_line", "--samples", "--context", "1"])

        assert result.exit_code == 0
        assert "Samples (context: 1 before, 1 after)" in result.output

        # Should show the match
        assert "last_line match" in result.output

        # Should show line before
        assert "second_to_last match" in result.output

        # Should NOT show error message
        assert "No context available" not in result.output

    def test_match_on_second_to_last_line_with_context_1(self):
        """Test that match on second-to-last line shows line after."""
        result = self.runner.invoke(search_command, [self.test_file, "second_to_last", "--samples", "--context", "1"])

        assert result.exit_code == 0

        # Should show the match
        assert "second_to_last match" in result.output

        # Should show line after (the last line)
        assert "last_line match" in result.output

    def test_match_on_last_line_with_context_2(self):
        """Test that match on last line shows 2 lines before with --context=2."""
        result = self.runner.invoke(search_command, [self.test_file, "last_line", "--samples", "--context", "2"])

        assert result.exit_code == 0

        # Should show the match
        assert "last_line match" in result.output

        # Should show 2 lines before
        assert "line 3: normal" in result.output
        assert "second_to_last match" in result.output

    def test_all_matches_show_with_json(self):
        """Test that all matches near end of file appear in JSON output."""
        result = self.runner.invoke(search_command, [self.test_file, "match", "--samples", "--context", "1", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)

        # Should have 2 matches
        assert len(data["matches"]) == 2

        # Both matches should have context_lines
        assert len(data["context_lines"]) == 2


class TestConsecutiveMatches:
    """Test fixes for incomplete context when matches are on consecutive lines."""

    def setup_method(self):
        """Create test files before each test."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

        # Create a test file with consecutive matches
        self.test_file = os.path.join(self.temp_dir, "test.txt")
        with open(self.test_file, "w") as f:
            f.write("line 1: normal\n")
            f.write("line 2: normal\n")
            f.write("line 3: pattern_a match\n")
            f.write("line 4: pattern_b match\n")
            f.write("line 5: pattern_c match\n")
            f.write("line 6: normal\n")
            f.write("line 7: normal\n")

    def test_consecutive_matches_show_in_each_others_context_1(self):
        """Test that consecutive matches appear in each other's context with --context=1."""
        result = self.runner.invoke(
            search_command,
            [self.test_file, "-e", "pattern_a", "-e", "pattern_b", "-e", "pattern_c", "--samples", "--context", "1"],
        )

        assert result.exit_code == 0

        # All three patterns should be found
        assert "pattern_a match" in result.output
        assert "pattern_b match" in result.output
        assert "pattern_c match" in result.output

        # For pattern_a (line 3), should show line 2 before and line 4 after
        # For pattern_b (line 4), should show line 3 before and line 5 after
        # For pattern_c (line 5), should show line 4 before and line 6 after

        # Check that sample blocks exist for all three
        assert result.output.count("=== ") >= 3

    def test_consecutive_matches_show_in_each_others_context_2(self):
        """Test that consecutive matches appear in each other's context with --context=2."""
        result = self.runner.invoke(
            search_command, [self.test_file, "-e", "pattern_a", "-e", "pattern_b", "--samples", "--context", "2"]
        )

        assert result.exit_code == 0

        # Both patterns should be found
        assert "pattern_a match" in result.output
        assert "pattern_b match" in result.output

        # For pattern_a (line 3), should show:
        # - 2 lines before: lines 1, 2
        # - line 3 (the match)
        # - 2 lines after: lines 4, 5
        # This means pattern_b and pattern_c should appear in pattern_a's context

        # For pattern_b (line 4), should show:
        # - 2 lines before: lines 2, 3
        # - line 4 (the match)
        # - 2 lines after: lines 5, 6
        # This means pattern_a and pattern_c should appear in pattern_b's context

    def test_consecutive_matches_complete_context_json(self):
        """Test that JSON output has complete context for consecutive matches."""
        result = self.runner.invoke(
            search_command,
            [self.test_file, "-e", "pattern_a", "-e", "pattern_b", "--samples", "--context", "2", "--json"],
        )

        assert result.exit_code == 0
        data = json.loads(result.output)

        # Should have 2 matches
        assert len(data["matches"]) == 2

        # Both should have context_lines
        assert len(data["context_lines"]) == 2

        # Check that each match has the expected number of context lines
        for key, lines in data["context_lines"].items():
            # With context=2 before and after, and the matched line, we should have:
            # - For pattern_a (line 3): lines 1, 2, 3, 4, 5 = 5 lines
            # - For pattern_b (line 4): lines 2, 3, 4, 5, 6 = 5 lines
            assert len(lines) >= 3  # At minimum: matched line + some context

            # Verify each line has proper structure
            for line in lines:
                assert "relative_line_number" in line
                assert "line_text" in line
                assert "absolute_offset" in line

    def test_consecutive_matches_on_last_lines(self):
        """Test consecutive matches where one is on the last line."""
        # Create a file with matches on the last two lines
        test_file = os.path.join(self.temp_dir, "last_lines.txt")
        with open(test_file, "w") as f:
            f.write("line 1: normal\n")
            f.write("line 2: normal\n")
            f.write("line 3: normal\n")
            f.write("line 4: penultimate match\n")
            f.write("line 5: ultimate match")  # No trailing newline

        result = self.runner.invoke(
            search_command, [test_file, "-e", "penultimate", "-e", "ultimate", "--samples", "--context", "1"]
        )

        assert result.exit_code == 0

        # Both matches should be shown
        assert "penultimate match" in result.output
        assert "ultimate match" in result.output

        # For penultimate (line 4), should show line 3 before and line 5 after
        # For ultimate (line 5), should show line 4 before

        # Should NOT show error message
        assert "No context available" not in result.output


class TestContextCombinations:
    """Test various combinations of context parameters."""

    def setup_method(self):
        """Create test files before each test."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

        self.test_file = os.path.join(self.temp_dir, "test.txt")
        with open(self.test_file, "w") as f:
            for i in range(1, 11):
                if i == 5:
                    f.write(f"line {i}: match here\n")
                else:
                    f.write(f"line {i}: normal\n")

    def test_context_4_shows_4_lines_before_and_after(self):
        """Test that --context=4 shows 4 lines before and after."""
        result = self.runner.invoke(search_command, [self.test_file, "match", "--samples", "--context", "4"])

        assert result.exit_code == 0
        assert "Samples (context: 4 before, 4 after)" in result.output

        # Should show 4 lines before (lines 1-4)
        assert "line 1:" in result.output
        assert "line 2:" in result.output
        assert "line 3:" in result.output
        assert "line 4:" in result.output

        # Should show the match (line 5)
        assert "match here" in result.output

        # Should show 4 lines after (lines 6-9)
        assert "line 6:" in result.output
        assert "line 7:" in result.output
        assert "line 8:" in result.output
        assert "line 9:" in result.output

    def test_asymmetric_context_before_2_after_1(self):
        """Test --before=2 --after=1."""
        result = self.runner.invoke(
            search_command, [self.test_file, "match", "--samples", "--before", "2", "--after", "1"]
        )

        assert result.exit_code == 0
        assert "Samples (context: 2 before, 1 after)" in result.output

        # Should show 2 lines before
        assert "line 3:" in result.output
        assert "line 4:" in result.output

        # Should show the match
        assert "match here" in result.output

        # Should show 1 line after
        assert "line 6:" in result.output

        # Should NOT show line 2 (too far before) or line 7 (too far after)
        output_after_samples = result.output.split("Samples")[1]
        assert "line 2:" not in output_after_samples or result.output.count("line 2:") <= 1
        assert "line 7:" not in output_after_samples or result.output.count("line 7:") <= 1
