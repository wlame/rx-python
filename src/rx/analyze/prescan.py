"""Parallel ripgrep prescan for fast anomaly detection."""

import json
import logging
import re
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from time import time

from rx.file_utils import MAX_SUBPROCESSES, FileTask, create_file_tasks

from .detectors import (
    AnomalyDetector,
    ErrorKeywordDetector,
    HighEntropyDetector,
    JsonDumpDetector,
    TracebackDetector,
    WarningKeywordDetector,
)


logger = logging.getLogger(__name__)


@dataclass
class PrescanMatch:
    """A match from rg prescan with detector info."""

    line_num: int
    byte_offset: int
    detector_name: str
    severity: float
    line_text: str


def rg_prescan_keywords(filepath: str, patterns: list[tuple[re.Pattern, float]]) -> set[int]:
    """Use ripgrep to quickly find lines matching error keywords.

    This is MUCH faster than checking each line with Python regex,
    especially for large files. Ripgrep uses SIMD and parallel processing.

    Args:
        filepath: Path to the file to scan
        patterns: List of (compiled_pattern, severity) tuples

    Returns:
        Set of line numbers (1-based) that match any keyword pattern
    """
    if not patterns:
        return set()

    # Build rg command with all patterns
    # Use -n for line numbers, -o for only matching (faster)
    rg_cmd = ['rg', '--line-number', '--no-heading', '--color=never']

    # Add each pattern - convert Python regex to rg pattern
    for pattern, _ in patterns:
        # Get the pattern string from compiled regex
        rg_cmd.extend(['-e', pattern.pattern])

    rg_cmd.append(filepath)

    try:
        result = subprocess.run(
            rg_cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout for very large files
        )

        matching_lines: set[int] = set()

        # Parse output: "line_num:matched_text"
        for line in result.stdout.splitlines():
            if ':' in line:
                try:
                    line_num_str = line.split(':', 1)[0]
                    matching_lines.add(int(line_num_str))
                except (ValueError, IndexError):
                    continue

        logger.info(f'rg prescan found {len(matching_lines)} lines with error keywords in {filepath}')
        return matching_lines

    except subprocess.TimeoutExpired:
        logger.warning(f'rg prescan timed out for {filepath}, falling back to line-by-line')
        return set()
    except FileNotFoundError:
        logger.warning('ripgrep (rg) not found, falling back to line-by-line keyword detection')
        return set()
    except Exception as e:
        logger.warning(f'rg prescan failed: {e}, falling back to line-by-line')
        return set()


def _prescan_chunk_worker(
    task: FileTask,
    patterns: list[str],
    pattern_to_detector: dict[str, tuple[str, float]],
) -> tuple[FileTask, list[PrescanMatch], float]:
    """Worker function to prescan a single file chunk with ripgrep.

    Uses dd | rg --json pipeline similar to rx trace for parallel processing.

    Args:
        task: FileTask defining the chunk to process
        patterns: List of regex pattern strings
        pattern_to_detector: Mapping from pattern -> (detector_name, severity)

    Returns:
        Tuple of (task, list_of_matches, execution_time)
    """
    start_time = time()
    thread_id = threading.current_thread().name

    logger.debug(f'[PRESCAN {thread_id}] Processing chunk {task.task_id}: offset={task.offset}, count={task.count}')

    try:
        # Calculate dd block parameters (same approach as rx trace)
        bs = 1024 * 1024  # 1MB block size
        skip_blocks = task.offset // bs
        skip_remainder = task.offset % bs
        actual_dd_offset = skip_blocks * bs
        count_blocks = (task.count + skip_remainder + bs - 1) // bs

        # Run dd | rg --json pipeline
        dd_proc = subprocess.Popen(
            ['dd', f'if={task.filepath}', f'bs={bs}', f'skip={skip_blocks}', f'count={count_blocks}', 'status=none'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Build ripgrep command with --json for structured output
        rg_cmd = ['rg', '--json', '--no-heading', '--color=never']
        for pattern in patterns:
            rg_cmd.extend(['-e', pattern])
        rg_cmd.append('-')  # Read from stdin

        rg_proc = subprocess.Popen(
            rg_cmd,
            stdin=dd_proc.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if dd_proc.stdout:
            dd_proc.stdout.close()

        # Parse JSON output
        matches: list[PrescanMatch] = []

        for line in rg_proc.stdout or []:
            try:
                data = json.loads(line)
                if data.get('type') != 'match':
                    continue

                match_data = data.get('data', {})
                rg_offset = match_data.get('absolute_offset', 0)

                # Convert to absolute offset in file
                absolute_offset = actual_dd_offset + rg_offset

                # Only include matches within this task's range
                if not (task.offset <= absolute_offset < task.offset + task.count):
                    continue

                line_text = match_data.get('lines', {}).get('text', '')

                # Find which detector this pattern belongs to
                for pattern, (detector_name, severity) in pattern_to_detector.items():
                    if re.search(pattern, line_text):
                        matches.append(
                            PrescanMatch(
                                line_num=-1,  # Will be calculated from offset
                                byte_offset=absolute_offset,
                                detector_name=detector_name,
                                severity=severity,
                                line_text=line_text.rstrip('\n\r'),
                            )
                        )
                        break

            except json.JSONDecodeError:
                continue

        rg_proc.wait()
        dd_proc.wait()

        elapsed = time() - start_time
        logger.debug(f'[PRESCAN {thread_id}] Chunk {task.task_id} completed: {len(matches)} matches in {elapsed:.3f}s')

        return (task, matches, elapsed)

    except Exception as e:
        elapsed = time() - start_time
        logger.error(f'[PRESCAN {thread_id}] Chunk {task.task_id} failed: {e}')
        return (task, [], elapsed)


def rg_prescan_all_detectors(
    filepath: str,
    detectors: list[AnomalyDetector],
) -> dict[str, list[PrescanMatch]]:
    """Use ripgrep to prescan file for ALL regex-based anomaly patterns in PARALLEL.

    This uses the same chunked parallel processing approach as rx trace:
    - Split file into chunks using create_file_tasks()
    - Process each chunk in parallel with ThreadPoolExecutor
    - Each worker runs dd | rg --json pipeline
    - Merge results from all workers

    This is MUCH faster than single-threaded rg for very large files (75GB+).

    Args:
        filepath: Path to the file to scan
        detectors: List of anomaly detectors to extract patterns from

    Returns:
        Dict mapping detector_name -> list of PrescanMatch objects
    """
    # Collect all patterns from regex-based detectors
    pattern_to_detector: dict[str, tuple[str, float]] = {}

    for detector in detectors:
        if isinstance(detector, TracebackDetector):
            for lang_patterns in detector.TRACEBACK_START_PATTERNS.values():
                for pattern in lang_patterns:
                    pattern_to_detector[pattern.pattern] = (detector.name, 0.9)
        elif isinstance(detector, ErrorKeywordDetector):
            for pattern, severity in detector.ERROR_KEYWORDS:
                pattern_to_detector[pattern.pattern] = (detector.name, severity)
        elif isinstance(detector, WarningKeywordDetector):
            for pattern, severity in detector.WARNING_KEYWORDS:
                pattern_to_detector[pattern.pattern] = (detector.name, severity)
        elif isinstance(detector, HighEntropyDetector):
            for pattern in detector.SECRET_CONTEXT_PATTERNS:
                pattern_to_detector[pattern.pattern] = (detector.name, 0.6)
        elif isinstance(detector, JsonDumpDetector):
            for pattern in detector.JSON_START_PATTERNS:
                pattern_to_detector[pattern.pattern] = (detector.name, 0.3)

    if not pattern_to_detector:
        return {}

    patterns = list(pattern_to_detector.keys())

    try:
        start_time = time()

        # Create file tasks for parallel processing
        tasks = create_file_tasks(filepath)
        logger.info(f'[PRESCAN] Created {len(tasks)} parallel tasks for {filepath}')

        # Process chunks in parallel
        all_matches: list[PrescanMatch] = []
        total_worker_time = 0.0

        with ThreadPoolExecutor(max_workers=MAX_SUBPROCESSES, thread_name_prefix='Prescan') as executor:
            future_to_task = {
                executor.submit(_prescan_chunk_worker, task, patterns, pattern_to_detector): task for task in tasks
            }

            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    _, matches, elapsed = future.result()
                    all_matches.extend(matches)
                    total_worker_time += elapsed
                except Exception as e:
                    logger.error(f'[PRESCAN] Task {task.task_id} failed: {e}')

        # Sort by byte offset
        all_matches.sort(key=lambda m: m.byte_offset)

        # Group by detector
        matches_by_detector: dict[str, list[PrescanMatch]] = {}
        for match in all_matches:
            if match.detector_name not in matches_by_detector:
                matches_by_detector[match.detector_name] = []
            matches_by_detector[match.detector_name].append(match)

        elapsed = time() - start_time
        total_matches = len(all_matches)
        logger.info(
            f'[PRESCAN] Completed in {elapsed:.1f}s (worker time: {total_worker_time:.1f}s): '
            f'{total_matches} matches across {len(matches_by_detector)} detectors'
        )

        return matches_by_detector

    except FileNotFoundError:
        logger.warning('ripgrep (rg) not found')
        return {}
    except Exception as e:
        logger.warning(f'rg prescan failed: {e}')
        return {}
