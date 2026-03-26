"""Worker functions for the trace engine.

This module contains the worker-level processing logic:
- HookCallbacks dataclass for event callbacks during parsing
- identify_matching_patterns() for determining which patterns matched a line
- process_task_worker() for running dd|rg pipelines on individual file chunks

Extracted from trace.py to keep the main orchestrator focused on
coordination logic while this module handles per-task processing.
"""

import logging
import re
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from rx.cli import prometheus as prom
from rx.file_utils import FileTask
from rx.models import ContextLine, Submatch
from rx.rg_json import RgContextEvent, RgMatchEvent, parse_rg_json_event
from rx.utils import NEWLINE_SYMBOL


logger = logging.getLogger(__name__)


@dataclass
class HookCallbacks:
    """Callbacks for hook events during parsing.

    These callbacks are called synchronously during parsing.
    The caller is responsible for making them async/non-blocking if needed.
    """

    on_match_found: Callable[[dict], None] | None = None
    on_file_scanned: Callable[[dict], None] | None = None

    # Request metadata for hook payloads
    request_id: str = ''
    patterns: dict = field(default_factory=dict)  # pattern_id -> pattern string
    files: dict = field(default_factory=dict)  # file_id -> filepath


def identify_matching_patterns(
    line_text: str, submatches: list[Submatch], pattern_ids: dict[str, str], rg_extra_args: list[str]
) -> list[str]:
    """
    Identify which patterns actually matched the line by testing each pattern.

    Since ripgrep doesn't tell us which pattern matched when multiple patterns are used,
    we need to determine this in Python by checking which patterns match the submatches.

    A line may match multiple different patterns, so we return a list of pattern IDs.

    Args:
        line_text: The matched line text
        submatches: List of Submatch objects with matched text and positions
        pattern_ids: Dictionary mapping pattern_id -> pattern string
        rg_extra_args: List of extra arguments passed to ripgrep (e.g., ['-i'] for case-insensitive)

    Returns:
        List of pattern_ids that matched (may contain multiple patterns).
        Returns empty list if no patterns match (this can happen with stale cache data).
    """
    # Determine regex flags from rg_extra_args
    flags = re.NOFLAG
    if rg_extra_args and '-i' in rg_extra_args:
        flags |= re.IGNORECASE

    if not submatches:
        # No submatches available - we must validate each pattern against the line directly.
        # This can happen when matches are reconstructed from cache where submatch data
        # wasn't preserved. We cannot blindly assume the first pattern matches.
        matching_pattern_ids = []
        for pattern_id, pattern in pattern_ids.items():
            try:
                regex = re.compile(pattern, flags)
                if regex.search(line_text):
                    matching_pattern_ids.append(pattern_id)
            except re.error:
                # Invalid regex, skip
                continue
        return matching_pattern_ids

    # Get the matched text from submatches
    matched_texts = set(sm.text for sm in submatches)

    matching_pattern_ids = []

    # Try each pattern to see which ones match
    for pattern_id, pattern in pattern_ids.items():
        try:
            regex = re.compile(pattern, flags)

            # Find all matches in the line
            pattern_matches = set(m.group() for m in regex.finditer(line_text))

            # Check if any of this pattern's matches are in the submatches
            if pattern_matches & matched_texts:  # Set intersection
                matching_pattern_ids.append(pattern_id)

        except re.error:
            # Invalid regex, skip
            continue

    return matching_pattern_ids


def process_task_worker(
    task: FileTask,
    pattern_ids: dict[str, str],
    rg_extra_args: list | None = None,
    context_before: int = 0,
    context_after: int = 0,
) -> tuple[FileTask, list[dict], list[ContextLine], float]:
    """
    Worker function to process a single FileTask with multiple patterns using ripgrep.
    Runs dd | rg --json pipeline and returns rich match data with optional context.

    Args:
        task: FileTask to process
        pattern_ids: Dictionary mapping pattern_id -> pattern string
        rg_extra_args: Optional list of extra arguments to pass to ripgrep
        context_before: Number of context lines before each match (0 = disabled)
        context_after: Number of context lines after each match (0 = disabled)

    Returns:
        Tuple of (task, list_of_match_dicts, list_of_context_lines, execution_time)

        Match dict structure:
        {
            'offset': int,           # Absolute byte offset of matched line start
            'pattern_ids': [str],    # List of pattern IDs that could have matched
            'line_number': int,      # Line number (1-indexed)
            'line_text': str,        # The matched line content
            'submatches': [Submatch] # Detailed submatch info
        }
    """
    # Import DEBUG_MODE from trace module (stays there, referenced by web.py)
    from rx.trace import DEBUG_MODE

    if rg_extra_args is None:
        rg_extra_args = []

    start_time = time.time()
    thread_id = threading.current_thread().name

    # Track active workers
    prom.active_workers.inc()

    logger.debug(
        f'[WORKER {thread_id}] Starting JSON task {task.task_id}: '
        f'file={task.filepath}, offset={task.offset}, count={task.count}'
    )

    try:
        # Calculate dd block parameters
        # We use 1MB blocks for dd to balance performance and memory
        bs = 1024 * 1024  # 1MB block size

        # Split task.offset into complete blocks and remainder bytes
        # Example: task.offset=75,000,000, bs=1,048,576
        #   skip_blocks = 71 (skip 71 complete MB blocks)
        #   skip_remainder = 552,448 bytes (the "extra" bytes into block 72)
        skip_blocks = task.offset // bs
        skip_remainder = task.offset % bs

        # dd actually starts reading at the last complete block boundary
        # This is BEFORE our desired task.offset by skip_remainder bytes
        # Example: actual_dd_offset = 71 * 1,048,576 = 74,448,896
        #   (which is 552,448 bytes before task.offset of 75,000,000)
        actual_dd_offset = skip_blocks * bs

        # Calculate how many blocks dd needs to read to ensure we get all task.count bytes
        # We add skip_remainder because dd starts before task.offset
        # We add (bs - 1) for ceiling division to ensure we read enough
        # Example: if task.count=20MB and skip_remainder=552,448:
        #   count_blocks = ceil((20MB + 552,448) / 1MB) = 21 blocks
        count_blocks = (task.count + skip_remainder + bs - 1) // bs

        logger.debug(
            f'[WORKER {thread_id}] Task {task.task_id}: '
            f'dd bs={bs} skip={skip_blocks} count={count_blocks}, '
            f'actual_dd_offset={actual_dd_offset}'
        )

        # Run dd | rg with --json mode
        dd_proc = subprocess.Popen(
            ['dd', f'if={task.filepath}', f'bs={bs}', f'skip={skip_blocks}', f'count={count_blocks}', 'status=none'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Build ripgrep command with --json and multiple -e patterns
        rg_cmd = ['rg', '--json', '--no-heading', '--color=never']

        # Add context flags if requested
        if context_before > 0:
            rg_cmd.extend(['-B', str(context_before)])
        if context_after > 0:
            rg_cmd.extend(['-A', str(context_after)])

        # Add all patterns with -e flag
        for pattern in pattern_ids.values():
            rg_cmd.extend(['-e', pattern])

        # Add extra args (but filter out incompatible ones)
        filtered_extra_args = [arg for arg in rg_extra_args if arg not in ['--byte-offset', '--only-matching']]
        rg_cmd.extend(filtered_extra_args)

        rg_cmd.append('-')  # Read from stdin

        logger.debug(f'[WORKER {thread_id}] Running: {" ".join(rg_cmd)}')

        rg_proc = subprocess.Popen(
            rg_cmd,
            stdin=dd_proc.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if dd_proc.stdout:
            dd_proc.stdout.close()

        # Debug mode: capture all output for debugging
        debug_output = []
        debug_file = None

        if DEBUG_MODE:
            # Create debug file with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]  # milliseconds
            debug_file = f'.debug_{timestamp}_thread{thread_id}_task{task.task_id}.txt'

            # Write header with command and metadata
            with open(debug_file, 'w') as f:
                f.write(f'Timestamp: {datetime.now().isoformat()}\n')
                f.write(f'Thread ID: {thread_id}\n')
                f.write(f'Task ID: {task.task_id}\n')
                f.write(f'File: {task.filepath}\n')
                f.write(f'Offset: {task.offset}\n')
                f.write(f'Count: {task.count}\n')
                f.write('\nDD Command:\n')
                f.write(f'  dd if={task.filepath} bs={bs} skip={skip_blocks} count={count_blocks} status=none\n')
                f.write('\nRipgrep Command:\n')
                f.write(f'  {" ".join(rg_cmd)}\n')
                f.write('\nRipgrep JSON Output:\n')
                f.write('=' * 80 + '\n')

        # Parse JSON events from ripgrep output
        # Collect matches and context lines
        matches = []
        context_lines = []

        for line in rg_proc.stdout or []:
            # Save to debug file if enabled
            if DEBUG_MODE and line:
                debug_output.append(line)

            event = parse_rg_json_event(line)

            if isinstance(event, RgMatchEvent):
                # Match event - extract rich data
                match_data = event.data

                # ripgrep's absolute_offset is relative to dd's output (which starts at actual_dd_offset)
                # Convert to true absolute offset in the original file
                absolute_offset = actual_dd_offset + match_data.absolute_offset

                # Critical: Only include matches that fall within THIS task's designated range
                # task.offset is where this task should start (inclusive)
                # task.offset + task.count is where this task should end (exclusive)
                # We read extra bytes (skip_remainder) at the start via dd, so we must filter them out
                # We also read extra bytes at the end (up to bs-1), so we must filter those too
                # This prevents duplicate matches across adjacent tasks
                if task.offset <= absolute_offset < task.offset + task.count:
                    # Extract submatches
                    submatches = [Submatch(text=sm.text, start=sm.start, end=sm.end) for sm in match_data.submatches]

                    # Since ripgrep can't tell us which specific pattern matched when using
                    # multiple patterns with flags like -i, we record all pattern IDs
                    matching_pattern_ids = list(pattern_ids.keys())

                    # For regular files, we don't track absolute line numbers during chunked processing
                    # Line numbers from rg are relative to the chunk, not the whole file
                    matches.append(
                        {
                            'offset': absolute_offset,
                            'pattern_ids': matching_pattern_ids,
                            'line_number': match_data.line_number,
                            'absolute_line_number': -1,  # Unknown for chunked processing
                            'line_text': match_data.lines.text.rstrip(NEWLINE_SYMBOL),
                            'submatches': submatches,
                        }
                    )

                    logger.debug(
                        f'[WORKER {thread_id}] Match: line={match_data.line_number}, '
                        f'offset={absolute_offset}, submatches={len(submatches)}'
                    )

            elif isinstance(event, RgContextEvent):
                # Context event - only include if within task range
                context_data = event.data
                absolute_offset = actual_dd_offset + context_data.absolute_offset

                if task.offset <= absolute_offset < task.offset + task.count:
                    context_lines.append(
                        ContextLine(
                            relative_line_number=context_data.line_number,
                            absolute_line_number=-1,  # Unknown for chunked processing
                            line_text=context_data.lines.text.rstrip(NEWLINE_SYMBOL),
                            absolute_offset=absolute_offset,
                        )
                    )

                    logger.debug(
                        f'[WORKER {thread_id}] Context: line={context_data.line_number}, offset={absolute_offset}'
                    )

        rg_proc.wait()
        dd_proc.wait()

        elapsed = time.time() - start_time

        # Write debug output to file if enabled
        if DEBUG_MODE and debug_file and debug_output:
            try:
                with open(debug_file, 'a') as f:
                    for output_line in debug_output:
                        f.write(output_line.decode('utf-8', errors='replace'))
                    f.write('\n' + '=' * 80 + '\n')
                    f.write('\nSummary:\n')
                    f.write(f'  Matches found: {len(matches)}\n')
                    f.write(f'  Context lines: {len(context_lines)}\n')
                    f.write(f'  Duration: {elapsed:.3f}s\n')
                    f.write(f'  Return code (rg): {rg_proc.returncode}\n')
                    f.write(f'  Return code (dd): {dd_proc.returncode}\n')
                logger.info(f'[WORKER {thread_id}] Debug output written to {debug_file}')
            except Exception as e:
                logger.warning(f'[WORKER {thread_id}] Failed to write debug file: {e}')

        logger.debug(
            f'[WORKER {thread_id}] Task {task.task_id} completed: '
            f'found {len(matches)} matches, {len(context_lines)} context lines in {elapsed:.3f}s'
        )

        # Track task completion
        prom.worker_tasks_completed.inc()
        prom.active_workers.dec()

        return (task, matches, context_lines, elapsed)

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f'[WORKER {thread_id}] Task {task.task_id} failed after {elapsed:.3f}s: {e}')

        # Track task failure
        prom.worker_tasks_failed.inc()
        prom.active_workers.dec()

        return (task, [], [], elapsed)
