"""Compressed file processing for the trace engine.

This module handles searching within compressed files:
- Standard compressed files (gzip, xz, bz2) via decompressor|rg pipelines
- Seekable zstd files via parallel frame decompression with batched processing

Extracted from trace.py to keep the main orchestrator focused on
coordination logic while this module handles compression-specific details.
"""

import structlog
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from rx.cli import prometheus as prom
from rx.compression import CompressionFormat, detect_compression, get_decompressor_command
from rx.file_utils import MAX_SUBPROCESSES
from rx.models import ContextLine, Submatch
from rx.rg_json import RgContextEvent, RgMatchEvent, parse_rg_json_event
from rx.seekable_index import get_or_build_index
from rx.seekable_zstd import decompress_frame, read_seek_table
from rx.utils import NEWLINE_SYMBOL


logger = structlog.get_logger()


def process_compressed_file(
    filepath: str,
    pattern_ids: dict[str, str],
    rg_extra_args: list | None = None,
    context_before: int = 0,
    context_after: int = 0,
    max_results: int | None = None,
) -> tuple[list[dict], list[ContextLine], float]:
    """
    Process a compressed file by decompressing to stdout and piping to rg.

    Compressed files cannot be chunked like regular files, so they are
    processed sequentially in a single pass.

    Args:
        filepath: Path to the compressed file
        pattern_ids: Dictionary mapping pattern_id -> pattern string
        rg_extra_args: Optional list of extra arguments to pass to ripgrep
        context_before: Number of context lines before each match
        context_after: Number of context lines after each match
        max_results: Optional maximum number of results to return

    Returns:
        Tuple of (list_of_match_dicts, list_of_context_lines, execution_time)
    """
    if rg_extra_args is None:
        rg_extra_args = []

    start_time = time.time()
    thread_id = threading.current_thread().name

    logger.info("processing_compressed_file", filepath=filepath, thread=thread_id)

    # Detect compression format and get decompressor command
    compression_format = detect_compression(filepath)
    if compression_format == CompressionFormat.NONE:
        raise ValueError(f'File is not compressed: {filepath}')

    decompress_cmd = get_decompressor_command(compression_format, filepath)

    logger.debug("using_decompressor", command=" ".join(decompress_cmd), thread=thread_id)

    try:
        # Start decompression process
        decompress_proc = subprocess.Popen(
            decompress_cmd,
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

        logger.debug("running_rg_command", command=" ".join(rg_cmd), thread=thread_id)

        rg_proc = subprocess.Popen(
            rg_cmd,
            stdin=decompress_proc.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if decompress_proc.stdout:
            decompress_proc.stdout.close()

        # Parse JSON events from ripgrep output
        matches = []
        context_lines = []
        match_count = 0

        for line in rg_proc.stdout or []:
            event = parse_rg_json_event(line)

            if isinstance(event, RgMatchEvent):
                # Check max_results limit
                if max_results is not None and match_count >= max_results:
                    break

                match_data = event.data

                # Extract submatches
                submatches = [Submatch(text=sm.text, start=sm.start, end=sm.end) for sm in match_data.submatches]

                # For compressed files, absolute_offset is in the decompressed stream
                # We don't know absolute line number for compressed files (no index)
                matches.append(
                    {
                        'offset': match_data.absolute_offset,  # Decompressed byte offset
                        'pattern_ids': list(pattern_ids.keys()),
                        'line_number': match_data.line_number,
                        'absolute_line_number': -1,  # Unknown for compressed files
                        'line_text': match_data.lines.text.rstrip(NEWLINE_SYMBOL),
                        'submatches': submatches,
                        'is_compressed': True,
                    }
                )
                match_count += 1

                logger.debug(
                    "compressed_match",
                    line=match_data.line_number,
                    offset=match_data.absolute_offset,
                    submatch_count=len(submatches),
                    thread=thread_id,
                )

            elif isinstance(event, RgContextEvent):
                context_data = event.data
                context_lines.append(
                    ContextLine(
                        relative_line_number=context_data.line_number,
                        absolute_line_number=-1,  # Unknown for compressed files
                        line_text=context_data.lines.text.rstrip(NEWLINE_SYMBOL),
                        absolute_offset=context_data.absolute_offset,
                    )
                )

        rg_proc.wait()
        decompress_proc.wait()

        # Check for decompression errors
        if decompress_proc.returncode != 0:
            stderr = decompress_proc.stderr.read().decode() if decompress_proc.stderr else ''
            logger.warning("decompression_warning", stderr=stderr, thread=thread_id)

        elapsed = time.time() - start_time

        logger.info(
            "compressed_file_completed",
            match_count=len(matches),
            context_line_count=len(context_lines),
            elapsed=round(elapsed, 3),
            thread=thread_id,
        )

        return (matches, context_lines, elapsed)

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error("compressed_file_failed", elapsed=round(elapsed, 3), error=str(e), thread=thread_id)
        raise


def process_seekable_zstd_frame_batch(
    filepath: str,
    frame_indices: list[int],
    frame_infos: list,
    pattern_ids: dict[str, str],
    rg_extra_args: list | None = None,
    context_before: int = 0,
    context_after: int = 0,
) -> tuple[list[int], list[dict], list[ContextLine], float]:
    """
    Process a batch of consecutive frames from a seekable zstd file.

    This batches multiple frames to reduce subprocess overhead. Consecutive
    compressed frames are concatenated and piped through a single zstd|rg pipeline.

    Args:
        filepath: Path to the seekable zstd file
        frame_indices: List of frame indices to process
        frame_infos: List of all FrameInfo objects (indexed by frame_index)
        pattern_ids: Dictionary mapping pattern_id -> pattern string
        rg_extra_args: Optional list of extra arguments to pass to ripgrep
        context_before: Number of context lines before each match
        context_after: Number of context lines after each match

    Returns:
        Tuple of (frame_indices, list_of_match_dicts, list_of_context_lines, execution_time)
    """
    if rg_extra_args is None:
        rg_extra_args = []

    start_time = time.time()
    thread_id = threading.current_thread().name

    logger.debug(
        "processing_seekable_frames",
        frame_count=len(frame_indices),
        frame_range_start=frame_indices[0],
        frame_range_end=frame_indices[-1],
        thread=thread_id,
    )

    # Track active workers
    prom.active_workers.inc()

    try:
        # Read all compressed frames and concatenate them
        # Consecutive zstd frames can be concatenated and decompressed as a stream
        compressed_chunks = []
        total_compressed_size = 0

        with open(filepath, 'rb') as f:
            for frame_idx in frame_indices:
                frame = frame_infos[frame_idx]
                f.seek(frame.compressed_offset)
                compressed_chunks.append(f.read(frame.compressed_size))
                total_compressed_size += frame.compressed_size

        compressed_data = b''.join(compressed_chunks)

        logger.debug(
            "read_compressed_frames",
            compressed_bytes=total_compressed_size,
            frame_count=len(frame_indices),
            thread=thread_id,
        )

        # Build ripgrep command
        rg_cmd = ['rg', '--json', '--no-heading', '--color=never']

        if context_before > 0:
            rg_cmd.extend(['-B', str(context_before)])
        if context_after > 0:
            rg_cmd.extend(['-A', str(context_after)])

        for pattern in pattern_ids.values():
            rg_cmd.extend(['-e', pattern])

        filtered_extra_args = [arg for arg in rg_extra_args if arg not in ['--byte-offset', '--only-matching']]
        rg_cmd.extend(filtered_extra_args)
        rg_cmd.append('-')

        # Create native pipeline: zstd -d | rg (no Python in the middle!)
        zstd_proc = subprocess.Popen(
            ['zstd', '-d', '-c'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        rg_proc = subprocess.Popen(
            rg_cmd,
            stdin=zstd_proc.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        zstd_proc.stdout.close()

        # Use line counts from the index (which are accurate since frames are line-aligned)
        # The index was built by decompressing each frame and counting lines precisely
        frame_line_counts = []
        for frame_idx in frame_indices:
            frame_info = frame_infos[frame_idx]
            # line_count in the index is accurate
            frame_line_counts.append(frame_info.line_count)

        # Now feed compressed data to pipeline and read results
        zstd_proc.stdin.write(compressed_data)
        zstd_proc.stdin.close()

        stdout, stderr = rg_proc.communicate()
        zstd_proc.wait()

        # Parse results and adjust offsets/line numbers
        matches = []
        context_lines = []

        for line in stdout.splitlines():
            event = parse_rg_json_event(line)

            if isinstance(event, RgMatchEvent):
                match_data = event.data

                # The offset from rg is relative to the concatenated decompressed stream
                # We need to find which frame it belongs to and adjust
                batch_offset = match_data.absolute_offset

                # Find which frame this match belongs to
                current_offset = 0
                for i, frame_idx in enumerate(frame_indices):
                    frame_info = frame_infos[frame_idx]
                    frame_size = frame_info.decompressed_size

                    if batch_offset < current_offset + frame_size:
                        # Match is in this frame
                        offset_in_frame = batch_offset - current_offset
                        absolute_offset = frame_info.decompressed_offset + offset_in_frame

                        # Adjust line number using ACTUAL line counts
                        # rg line numbers are 1-based within the batch
                        line_in_batch = match_data.line_number

                        # Count ACTUAL lines in previous frames in this batch
                        lines_before = sum(frame_line_counts[:i])

                        # Line within current frame
                        line_in_frame = line_in_batch - lines_before
                        adjusted_line_number = frame_info.first_line + line_in_frame - 1

                        # Debug logging
                        logger.debug(
                            "seekable_match",
                            frame_idx=frame_idx,
                            first_line=frame_info.first_line,
                            line_in_batch=line_in_batch,
                            lines_before=lines_before,
                            line_in_frame=line_in_frame,
                            adjusted_line_number=adjusted_line_number,
                            actual_line_count=frame_line_counts[i],
                            thread=thread_id,
                        )

                        submatches = [
                            Submatch(text=sm.text, start=sm.start, end=sm.end) for sm in match_data.submatches
                        ]

                        matches.append(
                            {
                                'offset': absolute_offset,
                                'frame_index': frame_idx,
                                'pattern_ids': list(pattern_ids.keys()),
                                'line_number': adjusted_line_number,
                                'absolute_line_number': adjusted_line_number,  # We know absolute line number
                                'line_text': match_data.lines.text.rstrip(NEWLINE_SYMBOL),
                                'submatches': submatches,
                                'is_compressed': True,
                                'is_seekable_zstd': True,
                            }
                        )
                        break

                    current_offset += frame_size

            elif isinstance(event, RgContextEvent):
                # Similar logic for context lines
                context_data = event.data
                batch_offset = context_data.absolute_offset

                current_offset = 0
                for i, frame_idx in enumerate(frame_indices):
                    frame_info = frame_infos[frame_idx]
                    frame_size = frame_info.decompressed_size

                    if batch_offset < current_offset + frame_size:
                        offset_in_frame = batch_offset - current_offset
                        absolute_offset = frame_info.decompressed_offset + offset_in_frame

                        line_in_batch = context_data.line_number
                        # Count ACTUAL lines in previous frames in this batch
                        lines_before = sum(frame_line_counts[:i])
                        line_in_frame = line_in_batch - lines_before
                        adjusted_line_number = frame_info.first_line + line_in_frame - 1

                        context_lines.append(
                            ContextLine(
                                relative_line_number=adjusted_line_number,
                                absolute_line_number=adjusted_line_number,  # We know absolute line number
                                line_text=context_data.lines.text.rstrip(NEWLINE_SYMBOL),
                                absolute_offset=absolute_offset,
                            )
                        )
                        break

                    current_offset += frame_size

        elapsed = time.time() - start_time

        logger.debug(
            "seekable_batch_completed",
            match_count=len(matches),
            context_line_count=len(context_lines),
            elapsed=round(elapsed, 3),
            thread=thread_id,
        )

        prom.worker_tasks_completed.inc()
        prom.active_workers.dec()

        return (frame_indices, matches, context_lines, elapsed)

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error("seekable_batch_failed", elapsed=round(elapsed, 3), error=str(e), thread=thread_id)
        prom.worker_tasks_failed.inc()
        prom.active_workers.dec()
        raise


def process_seekable_zstd_file(
    filepath: str,
    pattern_ids: dict[str, str],
    rg_extra_args: list | None = None,
    context_before: int = 0,
    context_after: int = 0,
    max_results: int | None = None,
) -> tuple[list[dict], list[ContextLine], float]:
    """
    Process a seekable zstd file using parallel frame decompression.

    Each frame is processed independently in parallel, enabling fast search
    on large compressed files.

    Args:
        filepath: Path to the seekable zstd file
        pattern_ids: Dictionary mapping pattern_id -> pattern string
        rg_extra_args: Optional list of extra arguments to pass to ripgrep
        context_before: Number of context lines before each match
        context_after: Number of context lines after each match
        max_results: Optional maximum number of results to return

    Returns:
        Tuple of (list_of_match_dicts, list_of_context_lines, execution_time)
    """
    if rg_extra_args is None:
        rg_extra_args = []

    start_time = time.time()

    logger.info("processing_seekable_zstd_file", filepath=filepath)

    # Get or build the index to get frame-to-line mapping
    index = get_or_build_index(filepath)

    logger.info(
        "seekable_file_indexed",
        frame_count=index.frame_count,
        line_count=index.line_count,
        decompressed_bytes=index.decompressed_size_bytes,
    )

    # Create enhanced frame info list with line mapping
    # We'll use FrameLineInfo objects directly (they have all needed fields)
    # Create a list indexed by frame_index for O(1) lookup
    zstd_frames = index.frames  # These are FrameLineInfo objects

    # Check if frames are line-aligned by testing first frame
    # Line-aligned frames end with newline; non-aligned frames split lines across boundaries
    first_frame_data = decompress_frame(filepath, 0, read_seek_table(filepath))
    frames_are_line_aligned = first_frame_data.endswith(b'\n')

    # Batch frames to reduce subprocess overhead
    # Process FRAMES_PER_BATCH consecutive frames in each worker
    # NOTE: Batching only works correctly for line-aligned frames
    if frames_are_line_aligned:
        FRAMES_PER_BATCH = 100  # Batching enabled for line-aligned frames
        logger.info("frames_line_aligned", batching_enabled=True, frames_per_batch=FRAMES_PER_BATCH)
    else:
        FRAMES_PER_BATCH = 1  # Disable batching for non-line-aligned frames
        logger.warning(
            "frames_not_line_aligned",
            batching_enabled=False,
            hint="Consider recreating this .zst file for better performance",
        )

    frame_batches = []
    for i in range(0, len(index.frames), FRAMES_PER_BATCH):
        batch_indices = list(range(i, min(i + FRAMES_PER_BATCH, len(index.frames))))
        frame_batches.append(batch_indices)

    logger.info("created_frame_batches", batch_count=len(frame_batches), frames_per_batch=FRAMES_PER_BATCH)

    # Track parallel tasks created
    prom.parallel_tasks_created.observe(len(frame_batches))

    # Process frame batches in parallel
    all_matches = []
    all_context_lines = []
    total_time = 0.0

    with ThreadPoolExecutor(max_workers=MAX_SUBPROCESSES, thread_name_prefix='SeekableWorker') as executor:
        # Submit batch tasks
        future_to_batch = {}
        for batch_indices in frame_batches:
            future = executor.submit(
                process_seekable_zstd_frame_batch,
                filepath,
                batch_indices,
                zstd_frames,
                pattern_ids,
                rg_extra_args,
                context_before,
                context_after,
            )
            future_to_batch[future] = batch_indices

        # Collect results
        for future in as_completed(future_to_batch):
            batch_indices = future_to_batch[future]

            try:
                _, matches, context_lines, elapsed = future.result()
                total_time += elapsed

                all_matches.extend(matches)
                all_context_lines.extend(context_lines)

                logger.debug(
                    "batch_results_collected",
                    batch_start=batch_indices[0],
                    batch_end=batch_indices[-1],
                    match_count=len(matches),
                    context_line_count=len(context_lines),
                )

                # Note: We don't break early on max_results because we need to sort all matches
                # by line number first to ensure we return the FIRST matches in the file,
                # not just the first matches found by parallel workers

            except Exception as e:
                logger.error(
                    "batch_failed",
                    batch_start=batch_indices[0],
                    batch_end=batch_indices[-1],
                    error=str(e),
                )

    # Sort matches by line number
    all_matches.sort(key=lambda m: m['line_number'])

    # Apply max_results limit
    if max_results is not None and len(all_matches) > max_results:
        all_matches = all_matches[:max_results]

    elapsed = time.time() - start_time

    logger.info(
        "seekable_file_completed",
        match_count=len(all_matches),
        context_line_count=len(all_context_lines),
        elapsed=round(elapsed, 3),
        worker_time=round(total_time, 3),
    )

    return (all_matches, all_context_lines, elapsed)
