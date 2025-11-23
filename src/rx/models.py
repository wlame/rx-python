"""Pydantic models for API requests and responses"""

import re
from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response with system introspection data"""

    status: str = Field(..., example="ok")
    ripgrep_available: bool = Field(..., example=True)
    app_version: str = Field(..., example="0.1.0", description="Application version")
    python_version: str = Field(..., example="3.13.1", description="Python interpreter version")
    os_info: dict[str, str] = Field(
        ...,
        example={"system": "Darwin", "release": "23.0.0", "version": "Darwin Kernel Version 23.0.0"},
        description="Operating system information",
    )
    system_resources: dict[str, Any] = Field(
        ...,
        example={"cpu_cores": 8, "cpu_cores_physical": 4, "ram_total_gb": 16.0, "ram_available_gb": 8.5},
        description="System resources (CPU cores and RAM)",
    )
    python_packages: dict[str, str] = Field(
        default_factory=dict,
        example={"fastapi": "0.115.6", "pydantic": "2.11.0", "uvicorn": "0.34.0"},
        description="Key Python package versions",
    )
    constants: dict[str, Any] = Field(
        default_factory=dict,
        example={"LOG_LEVEL": "INFO", "MIN_CHUNK_SIZE_MB": 20, "MAX_SUBPROCESSES": 20},
        description="Application configuration constants",
    )
    environment: dict[str, str] = Field(
        default_factory=dict, example={"RX_LOG_LEVEL": "INFO"}, description="Application-related environment variables"
    )
    docs_url: str = Field(..., example="https://github.com/wlame/rx", description="Link to application documentation")


class Submatch(BaseModel):
    """A submatch within a matched line

    Attributes:
        text: The actual matched text
        start: Byte offset from start of line where match begins
        end: Byte offset from start of line where match ends
    """

    text: str = Field(..., example="error", description="The matched text")
    start: int = Field(..., example=10, description="Start position in line (bytes)")
    end: int = Field(..., example=15, description="End position in line (bytes)")


class ContextLine(BaseModel):
    """A context line (non-matching line shown for context)

    Attributes:
        relative_line_number: Line number relative to file start (1-indexed).
                             When file_chunks=1, this is the absolute line number.
                             When file_chunks>1, this may be relative to chunk boundary.
        line_text: The actual line content
        absolute_offset: Byte offset from start of file to start of this line
    """

    relative_line_number: int = Field(
        ..., example=42, description="Line number in file (1-indexed, see file_chunks to determine if absolute)"
    )
    line_text: str = Field(..., example="  previous line", description="Line content")
    absolute_offset: int = Field(..., example=1234, description="Byte offset in file")


class Match(BaseModel):
    """A single match with pattern ID, file ID, and rich metadata

    Attributes:
        pattern: Pattern ID (e.g., 'p1', 'p2')
        file: File ID (e.g., 'f1', 'f2')
        offset: Absolute byte offset in file where the matched LINE starts
        relative_line_number: Line number relative to file start (1-indexed, optional).
                             When file_chunks=1, this is the absolute line number.
                             When file_chunks>1, this may be relative to chunk boundary.
        line_text: The actual line content that matched (optional)
        submatches: List of submatch details with positions (optional)
    """

    pattern: str = Field(..., example="p1", description="Pattern ID (p1, p2, ...)")
    file: str = Field(..., example="f1", description="File ID (f1, f2, ...)")
    offset: int = Field(..., example=123, description="Byte offset in file where matched line starts")
    relative_line_number: int | None = Field(
        None, example=42, description="Line number in file (1-indexed, see file_chunks to determine if absolute)"
    )
    line_text: str | None = Field(None, example="error: something failed", description="The matched line")
    submatches: list[Submatch] | None = Field(None, description="Submatch details with positions")


class TraceResponse(BaseModel):
    """Response from trace endpoint using ID-based structure for multi-pattern support

    Attributes:
        path: Path(s) that were searched (list of paths)
        time: Search duration in seconds
        patterns: Mapping of pattern IDs to pattern strings
        files: Mapping of file IDs to file paths
        matches: List of matches with rich metadata
        scanned_files: List of files that were scanned
        skipped_files: List of files that were skipped
        file_chunks: Optional mapping of file IDs to number of chunks/workers used
        context_lines: Optional mapping of match composite keys to context lines
        before_context: Number of context lines shown before matches (if context requested)
        after_context: Number of context lines shown after matches (if context requested)
        max_results: Maximum number of results requested (None if not specified)
    """

    path: list[str] = Field(..., example=["/path/to/dir"], description="List of paths that were searched")
    time: float = Field(..., example=0.123)
    patterns: dict[str, str] = Field(..., example={"p1": "error", "p2": "warning"})
    files: dict[str, str] = Field(..., example={"f1": "/path/file.log"})
    matches: list[Match] = Field(default=[], example=[])
    scanned_files: list[str] = Field(default=[], example=[])
    skipped_files: list[str] = Field(default=[], example=[])
    max_results: int | None = Field(None, description="Maximum number of results requested (None if not specified)")

    # File chunking metadata - shows how files were processed
    file_chunks: dict[str, int] | None = Field(
        None,
        description="Number of chunks/workers used per file (file_id -> num_chunks). "
        "1 = single worker (no chunking), >1 = parallel processing with multiple workers",
        example={"f1": 1, "f2": 5, "f3": 20},
    )

    # Context lines are stored with composite key: "p1:f1:100" -> [ContextLine, ...]
    context_lines: dict[str, list[ContextLine]] | None = Field(None, description="Context lines around matches")
    before_context: int | None = Field(None, example=3, description="Lines shown before matches")
    after_context: int | None = Field(None, example=3, description="Lines shown after matches")

    def to_cli(self, colorize: bool = False) -> str:
        """Format response for CLI output (human-readable, uses values instead of IDs)"""
        # ANSI color codes
        GREY = '\033[90m'
        CYAN = '\033[36m'
        BOLD_CYAN = '\033[1;36m'
        YELLOW = '\033[33m'
        MAGENTA = '\033[35m'
        BOLD_MAGENTA = '\033[1;35m'
        GREEN = '\033[32m'
        BOLD_GREEN = '\033[1;32m'
        BLUE = '\033[34m'
        RESET = '\033[0m'

        lines = []

        # Path(s) in bold cyan - handle single or multiple paths
        path_display = ", ".join(self.path) if len(self.path) > 1 else self.path[0]
        if colorize:
            lines.append(f"{GREY}Path:{RESET} {BOLD_CYAN}{path_display}{RESET}")
        else:
            lines.append(f"Path: {path_display}")

        # Show patterns with magenta color
        if len(self.patterns) == 1:
            pattern_val = list(self.patterns.values())[0]
            if colorize:
                lines.append(f"{GREY}Pattern:{RESET} {BOLD_MAGENTA}{pattern_val}{RESET}")
            else:
                lines.append(f"Pattern: {pattern_val}")
        else:
            if colorize:
                lines.append(f"{GREY}Patterns ({len(self.patterns)}):{RESET}")
                for pid, pattern in sorted(self.patterns.items()):
                    lines.append(f"  {BLUE}{pid}{RESET}: {MAGENTA}{pattern}{RESET}")
            else:
                lines.append(f"Patterns ({len(self.patterns)}):")
                for pid, pattern in sorted(self.patterns.items()):
                    lines.append(f"  {pid}: {pattern}")

        # Time in yellow
        if colorize:
            lines.append(f"{GREY}Time:{RESET} {YELLOW}{self.time:.3f}s{RESET}")
        else:
            lines.append(f"Time: {self.time:.3f}s")

        # File stats in green
        if self.scanned_files:
            if colorize:
                lines.append(f"{GREY}Files scanned:{RESET} {GREEN}{len(self.scanned_files)}{RESET}")
            else:
                lines.append(f"Files scanned: {len(self.scanned_files)}")
        if self.skipped_files:
            if colorize:
                lines.append(f"{GREY}Files skipped:{RESET} {GREY}{len(self.skipped_files)}{RESET}")
            else:
                lines.append(f"Files skipped: {len(self.skipped_files)}")

        # File chunking info (show if any files were chunked)
        if self.file_chunks:
            chunked_files = [fid for fid, count in self.file_chunks.items() if count > 1]
            if chunked_files:
                total_chunks = sum(self.file_chunks.values())
                if colorize:
                    lines.append(
                        f"{GREY}Parallel workers:{RESET} {CYAN}{total_chunks}{RESET} "
                        f"{GREY}({len(chunked_files)} file(s) chunked){RESET}"
                    )
                else:
                    lines.append(f"Parallel workers: {total_chunks} ({len(chunked_files)} file(s) chunked)")

        # Matches count in bold green
        if colorize:
            lines.append(f"{GREY}Matches:{RESET} {BOLD_GREEN}{len(self.matches)}{RESET}")
        else:
            lines.append(f"Matches: {len(self.matches)}")

        if self.matches:
            lines.append("")
            if colorize:
                lines.append(f"{GREY}Matches:{RESET}")
            else:
                lines.append("Matches:")

            for match in self.matches:
                pattern_val = self.patterns.get(match.pattern, match.pattern)
                file_val = self.files.get(match.file, match.file)

                if colorize:
                    lines.append(
                        f"  {CYAN}{file_val}{RESET}"
                        f"{GREY}:{RESET}"
                        f"{YELLOW}{match.offset}{RESET} "
                        f"{GREY}[{RESET}{MAGENTA}{pattern_val}{RESET}{GREY}]{RESET}"
                    )
                else:
                    lines.append(f"  {file_val}:{match.offset} [{pattern_val}]")

        return "\n".join(lines)


class FileAnalysisResult(BaseModel):
    """Analysis result for a single file."""

    file: str = Field(..., description="File ID (e.g., 'f1')")
    size_bytes: int = Field(..., description="File size in bytes")
    size_human: str = Field(..., description="Human-readable file size")
    is_text: bool = Field(..., description="Whether the file is a text file")

    created_at: str | None = Field(None, description="File creation timestamp (ISO format)")
    modified_at: str | None = Field(None, description="File modification timestamp (ISO format)")
    permissions: str | None = Field(None, description="File permissions (octal)")
    owner: str | None = Field(None, description="File owner")

    line_count: int | None = Field(None, description="Total number of lines (text files only)")
    empty_line_count: int | None = Field(None, description="Number of empty lines")
    max_line_length: int | None = Field(None, description="Maximum line length")
    avg_line_length: float | None = Field(None, description="Average line length (excluding empty lines)")
    median_line_length: float | None = Field(None, description="Median line length")
    line_length_stddev: float | None = Field(None, description="Standard deviation of line lengths")

    custom_metrics: dict = Field(default_factory=dict, description="Custom metrics from plugins")


class AnalyseResponse(BaseModel):
    """Response for file analysis endpoint."""

    path: str = Field(..., description="Analyzed path(s)")
    time: float = Field(..., description="Analysis time in seconds")
    files: dict[str, str] = Field(..., description="File ID to filepath mapping")
    results: list[FileAnalysisResult] = Field(..., description="Analysis results for each file")
    scanned_files: list[str] = Field(..., description="List of successfully scanned files")
    skipped_files: list[str] = Field(..., description="List of skipped files")

    def to_cli(self, colorize: bool = False) -> str:
        """Format analysis response for CLI output."""
        BOLD = '\033[1m'
        GREEN = '\033[32m'
        YELLOW = '\033[33m'
        CYAN = '\033[36m'
        GREY = '\033[90m'
        RESET = '\033[0m'

        lines = []

        # Header
        if colorize:
            lines.append(f"{BOLD}File Analysis{RESET}")
        else:
            lines.append("File Analysis")

        lines.append(f"Path: {self.path}")
        lines.append(f"Time: {self.time:.3f}s")
        lines.append(f"Files analyzed: {len(self.results)}")
        lines.append("")

        # Results for each file
        for result in self.results:
            filepath = self.files.get(result.file, result.file)

            if colorize:
                lines.append(f"{CYAN}{filepath}{RESET}")
            else:
                lines.append(filepath)

            lines.append(f"  Size: {result.size_human} ({result.size_bytes:,} bytes)")
            lines.append(f"  Type: {'Text' if result.is_text else 'Binary'}")

            if result.modified_at:
                lines.append(f"  Modified: {result.modified_at}")
            if result.permissions:
                lines.append(f"  Permissions: {result.permissions}")
            if result.owner:
                lines.append(f"  Owner: {result.owner}")

            if result.is_text and result.line_count is not None:
                lines.append(f"  Lines: {result.line_count:,} total, {result.empty_line_count:,} empty")
                if result.max_line_length:
                    lines.append(
                        f"  Line length: max={result.max_line_length}, "
                        f"avg={result.avg_line_length:.1f}, "
                        f"median={result.median_line_length:.1f}, "
                        f"stddev={result.line_length_stddev:.1f}"
                    )

            if result.custom_metrics:
                lines.append(f"  Custom metrics: {result.custom_metrics}")

            lines.append("")

        return "\n".join(lines)


class ComplexityDetails(BaseModel):
    """Breakdown of complexity scoring components"""

    nested_quantifiers: int | None = Field(None, example=50)
    greedy_sequences: int | None = Field(None, example=25)
    lookarounds: int | None = Field(None, example=15)
    backreferences: int | None = Field(None, example=20)
    alternation: int | None = Field(None, example=10)
    character_classes: int | None = Field(None, example=2)
    quantifiers: int | None = Field(None, example=6)
    anchors: int | None = Field(None, example=2)
    literals: float | None = Field(None, example=0.5)
    star_height_multiplier: float | None = Field(None, example=1.5)
    star_height_depth: int | None = Field(None, example=2)
    length_multiplier: float | None = Field(None, example=1.2)


class ComplexityResponse(BaseModel):
    """Response from complexity analysis endpoint"""

    regex: str = Field(..., example="(a+)+")
    score: float = Field(..., example=58.5)
    level: str = Field(..., example="moderate")
    risk: str = Field(..., example="Medium - reasonable performance expected")
    warnings: list[str] = Field(..., example=["Found 1 nested quantifier(s) - CRITICAL ReDoS risk"])
    details: ComplexityDetails
    pattern_length: int = Field(..., example=5)

    def to_cli(self, colorize: bool = False) -> str:
        """Format complexity response for CLI output"""
        # ANSI color codes
        BOLD = '\033[1m'
        RED = '\033[91m'
        YELLOW = '\033[33m'
        GREEN = '\033[32m'
        CYAN = '\033[36m'
        MAGENTA = '\033[35m'
        GREY = '\033[90m'
        RESET = '\033[0m'

        lines = []

        # Header
        if colorize:
            lines.append(f"{BOLD}Regex Complexity Analysis{RESET}")
        else:
            lines.append("Regex Complexity Analysis")
        lines.append("")

        # Pattern
        if colorize:
            lines.append(f"{GREY}Pattern:{RESET} {CYAN}{self.regex}{RESET}")
        else:
            lines.append(f"Pattern: {self.regex}")

        # Score with color based on level
        score_color = ""
        if colorize:
            if self.level in ["dangerous", "very_complex"]:
                score_color = RED
            elif self.level == "complex":
                score_color = YELLOW
            else:
                score_color = GREEN

        if colorize:
            lines.append(f"{GREY}Score:{RESET} {score_color}{self.score:.1f}{RESET}")
        else:
            lines.append(f"Score: {self.score:.1f}")

        # Level
        level_display = self.level.replace("_", " ").title()
        if colorize:
            lines.append(f"{GREY}Level:{RESET} {score_color}{level_display}{RESET}")
        else:
            lines.append(f"Level: {level_display}")

        # Risk
        if colorize:
            lines.append(f"{GREY}Risk:{RESET} {self.risk}")
        else:
            lines.append(f"Risk: {self.risk}")

        # Warnings
        if self.warnings:
            lines.append("")
            if colorize:
                lines.append(f"{BOLD}Warnings:{RESET}")
            else:
                lines.append("Warnings:")
            for warning in self.warnings:
                if colorize:
                    if "CRITICAL" in warning:
                        lines.append(f"  {RED}⚠{RESET}  {warning}")
                    else:
                        lines.append(f"  {YELLOW}•{RESET}  {warning}")
                else:
                    lines.append(f"  - {warning}")

        # Details
        lines.append("")
        if colorize:
            lines.append(f"{BOLD}Details:{RESET}")
        else:
            lines.append("Details:")

        details_dict = self.details.model_dump()
        for key, value in details_dict.items():
            key_display = key.replace("_", " ").title()
            if colorize:
                lines.append(f"  {GREY}{key_display}:{RESET} {value}")
            else:
                lines.append(f"  {key_display}: {value}")

        return "\n".join(lines)


class SamplesResponse(BaseModel):
    """Response from samples endpoint"""

    path: str = Field(..., example="/path/to/file.txt")
    offsets: list[int] = Field(..., example=[123, 456])
    before_context: int = Field(..., example=3)
    after_context: int = Field(..., example=3)
    samples: dict[str, list[str]] = Field(
        ...,
        example={
            "123": ["Line before", "Line with match", "Line after"],
            "456": ["Another before", "Another match", "Another after"],
        },
    )

    def to_cli(self, colorize: bool = False, regex: str = None) -> str:
        """Format response for CLI output

        Args:
            colorize: Whether to apply color formatting
            regex: Regex pattern to highlight in output
        """
        lines = []
        lines.append(f"File: {self.path}")
        lines.append(f"Context: {self.before_context} before, {self.after_context} after")
        lines.append("")

        # ANSI color codes
        GREY = '\033[90m'
        RED = '\033[91m'
        RESET = '\033[0m'

        for offset in self.offsets:
            offset_str = str(offset)
            if offset_str in self.samples:
                # Format offset header
                header = f"=== Offset: {offset} ==="
                if colorize:
                    header = f"{GREY}{header}{RESET}"
                lines.append(header)

                context_lines = self.samples[offset_str]
                for line in context_lines:
                    # Highlight regex matches if colorize is enabled and regex is provided
                    if colorize and regex:
                        try:
                            # Use re.sub to replace matches with colored version
                            highlighted = re.sub(f'({regex})', f'{RED}\\1{RESET}', line)
                            lines.append(highlighted)
                        except re.error:
                            # If regex is invalid, just show the line without highlighting
                            lines.append(line)
                    else:
                        lines.append(line)
                lines.append("")

        return "\n".join(lines)
