"""Ripgrep JSON event parser and models

This module provides Pydantic models for parsing ripgrep's --json output format.
Ripgrep emits newline-delimited JSON events that describe matches, context lines,
and metadata about the search operation.

Event types:
- begin: Indicates the start of searching a file
- match: A pattern match with line number, offset, and submatches
- context: Context lines around matches (when -A/-B/-C flags are used)
- end: Indicates the end of searching a file with statistics
- summary: Overall search statistics

Example usage:
    for line in rg_process.stdout:
        event = parse_rg_json_event(line)
        if isinstance(event, RgMatchEvent):
            print(f"Match at line {event.data.line_number}: {event.data.lines.text}")
"""

import json
import logging
from typing import Literal, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============================================================================
# Submatch Models
# ============================================================================


class RgSubmatch(BaseModel):
    """A submatch within a line (represents one pattern match)

    Attributes:
        match: The matched text wrapped in {"text": "..."} format
        start: Byte offset from start of line where match begins
        end: Byte offset from start of line where match ends
    """

    match: dict  # {"text": "matched string"}
    start: int
    end: int

    @property
    def text(self) -> str:
        """Convenience property for accessing matched text"""
        if isinstance(self.match, dict):
            return self.match.get('text', '')
        return str(self.match)


# ============================================================================
# Path and Text Wrapper Models
# ============================================================================


class RgText(BaseModel):
    """Wrapper for text fields in ripgrep JSON output

    Ripgrep wraps strings in {"text": "..."} objects to allow for
    future extension with encoding information.
    """

    text: str


class RgPath(BaseModel):
    """Path information in ripgrep JSON output

    Can be a text path or null for stdin/binary data.
    """

    text: str | None = None


# ============================================================================
# Data Payload Models for Each Event Type
# ============================================================================


class RgBeginData(BaseModel):
    """Data payload for 'begin' events

    Emitted when ripgrep starts searching a file.
    """

    path: RgPath


class RgMatchData(BaseModel):
    """Data payload for 'match' events

    Contains all information about a pattern match including:
    - File path
    - Line number (1-indexed)
    - Absolute byte offset in file
    - The matched line text
    - Submatches with positions

    Attributes:
        path: File path being searched
        lines: The line(s) containing the match
        line_number: Line number (1-indexed) where match occurs
        absolute_offset: Byte offset from start of file to start of this line
        submatches: List of pattern matches within the line with their positions
    """

    path: RgPath
    lines: RgText
    line_number: int
    absolute_offset: int
    submatches: list[RgSubmatch]

    def get_match_absolute_offsets(self) -> list[int]:
        """Calculate absolute byte offsets for each submatch in the file

        Returns:
            List of absolute byte offsets (from file start) for each submatch
        """
        return [self.absolute_offset + sm.start for sm in self.submatches]


class RgContextData(BaseModel):
    """Data payload for 'context' events

    Emitted when context lines are requested via -A/-B/-C flags.
    These are non-matching lines that appear before/after matches.

    Attributes:
        path: File path being searched
        lines: The context line text
        line_number: Line number (1-indexed)
        absolute_offset: Byte offset from start of file to start of this line
        submatches: Always empty list for context lines
    """

    path: RgPath
    lines: RgText
    line_number: int
    absolute_offset: int
    submatches: list[RgSubmatch] = Field(default_factory=list)


class RgStats(BaseModel):
    """Search statistics for a single file"""

    elapsed: dict  # {"secs": 0, "nanos": 123, "human": "0.000123s"}
    searches: int
    searches_with_match: int
    bytes_searched: int
    bytes_printed: int
    matched_lines: int
    matches: int


class RgEndData(BaseModel):
    """Data payload for 'end' events

    Emitted when ripgrep finishes searching a file.
    Contains statistics about the search operation.
    """

    path: RgPath
    binary_offset: int | None = None
    stats: RgStats


class RgSummaryData(BaseModel):
    """Data payload for 'summary' events

    Emitted at the very end with overall statistics across all files.
    """

    elapsed_total: dict
    stats: RgStats


# ============================================================================
# Event Models
# ============================================================================


class RgBeginEvent(BaseModel):
    """Begin event - ripgrep started searching a file"""

    type: Literal["begin"]
    data: RgBeginData


class RgMatchEvent(BaseModel):
    """Match event - pattern matched a line"""

    type: Literal["match"]
    data: RgMatchData


class RgContextEvent(BaseModel):
    """Context event - non-matching line shown for context"""

    type: Literal["context"]
    data: RgContextData


class RgEndEvent(BaseModel):
    """End event - ripgrep finished searching a file"""

    type: Literal["end"]
    data: RgEndData


class RgSummaryEvent(BaseModel):
    """Summary event - overall statistics"""

    type: Literal["summary"]
    data: RgSummaryData


# Union type for all possible events
RgEvent = Union[RgBeginEvent, RgMatchEvent, RgContextEvent, RgEndEvent, RgSummaryEvent]


# ============================================================================
# Parsing Functions
# ============================================================================


def parse_rg_json_event(json_line: str | bytes) -> RgEvent | None:
    """Parse a single line of ripgrep JSON output into a typed event

    Args:
        json_line: A single line from ripgrep's --json output

    Returns:
        Parsed event object, or None if parsing fails

    Example:
        >>> line = '{"type":"match","data":{"path":{"text":"file.txt"},...}}'
        >>> event = parse_rg_json_event(line)
        >>> if isinstance(event, RgMatchEvent):
        ...     print(event.data.line_number)
    """
    try:
        if isinstance(json_line, bytes):
            json_line = json_line.decode('utf-8')

        json_line = json_line.strip()
        if not json_line:
            return None

        data = json.loads(json_line)
        event_type = data.get('type')

        event_models = {
            'begin': RgBeginEvent,
            'match': RgMatchEvent,
            'context': RgContextEvent,
            'end': RgEndEvent,
            'summary': RgSummaryEvent,
        }
        model = event_models[event_type]
        return model(**data)

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from ripgrep: {e}")
        return None
    except Exception as e:
        logger.error(f"Error parsing ripgrep event: {e}")
        return None


def parse_rg_json_stream(lines: list[str | bytes]) -> list[RgEvent]:
    """Parse multiple lines of ripgrep JSON output

    Args:
        lines: List of JSON lines from ripgrep output

    Returns:
        List of successfully parsed events (skips invalid lines)
    """
    events = []
    for line in lines:
        event = parse_rg_json_event(line)
        if event is not None:
            events.append(event)
    return events
