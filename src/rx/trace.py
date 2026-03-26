"""Trace engine orchestrator.

Coordinates multi-file, multi-pattern search using ripgrep. Delegates to:
- trace_compressed: compressed file processing (gzip, xz, bz2, seekable zstd)
- trace_worker: per-chunk worker pipelines (dd|rg) and pattern matching

Public API: parse_paths(), HookCallbacks, identify_matching_patterns, DEBUG_MODE.
"""

import structlog
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from rx.cli import prometheus as prom
from rx.compression import is_compressed
from rx.file_utils import MAX_SUBPROCESSES, create_file_tasks, scan_directory_for_text_files, validate_file
from rx.models import ContextLine, FileScannedPayload, MatchFoundPayload, ParseResult
from rx.seekable_zstd import is_seekable_zstd
from rx.trace_compressed import process_compressed_file, process_seekable_zstd_file
from rx.trace_worker import HookCallbacks, identify_matching_patterns, process_task_worker  # noqa: F401
from rx.trace_cache import (
    build_cache_from_matches,
    get_cached_matches,
    get_compressed_cache_info,
    get_trace_cache_path,
    reconstruct_match_data,
    reconstruct_seekable_zstd_matches,
    save_trace_cache,
    should_cache_compressed_file,
    should_cache_file,
)
from rx.unified_index import (
    calculate_lines_for_offsets_batch,
    get_large_file_threshold_bytes,
    load_index,
)


logger = structlog.get_logger()


# NOTE: process_compressed_file, process_seekable_zstd_frame_batch, and
# process_seekable_zstd_file have been extracted to rx.trace_compressed.
# They are imported at the top of this module for use in the orchestrator.




# Debug mode - creates debug files with full rg commands and output
DEBUG_MODE = os.getenv('RX_DEBUG', '').lower() in ('1', 'true', 'yes')


# NOTE: HookCallbacks, identify_matching_patterns, and process_task_worker
# have been extracted to rx.trace_worker.
# They are re-exported from this module for backward compatibility.


def parse_multiple_files_multipattern(
    filepaths: list[str],
    pattern_ids: dict[str, str],
    file_ids: dict[str, str],
    max_results: int | None = None,
    rg_extra_args: list | None = None,
    context_before: int = 0,
    context_after: int = 0,
    hooks: HookCallbacks | None = None,
    use_cache: bool = True,
    use_index: bool = True,
) -> tuple[list[dict], dict[str, list[ContextLine]], dict[str, int]]:
    """
    Parse multiple files with multiple patterns and return rich match data.

    This function supports trace caching for large files. When a valid cache exists
    for a large file, matches are reconstructed from the cache instead of running
    the dd|rg pipeline. Cache is written for large files after successful complete scans.

    Args:
        filepaths: List of file paths
        pattern_ids: Dictionary mapping pattern_id -> pattern
        file_ids: Dictionary mapping file_id -> filepath
        max_results: Optional maximum number of results
        rg_extra_args: Optional list of extra arguments to pass to ripgrep
        context_before: Number of context lines before each match
        context_after: Number of context lines after each match
        hooks: Optional HookCallbacks for event notifications

    Returns:
        Tuple of (matches_list, context_dict, file_chunk_counts)

        matches_list: [{'pattern': 'p1', 'file': 'f1', 'offset': 100, 'line_number': 42, ...}, ...]
        context_dict: {'p1:f1:100': [ContextLine(...), ...], ...}
        file_chunk_counts: {'f1': 1, 'f2': 5, ...} - number of chunks per file
    """
    if rg_extra_args is None:
        rg_extra_args = []

    # Validate regex patterns
    for pattern in pattern_ids.values():
        test_proc = subprocess.run(['rg', '--', pattern], input=b'', capture_output=True, timeout=1)
        if test_proc.returncode == 2:
            error_msg = test_proc.stderr.decode('utf-8').strip()
            raise RuntimeError(f'Invalid regex pattern: {error_msg}')

    logger.info("parsing_files", file_count=len(filepaths), pattern_count=len(pattern_ids), mode="json")
    if context_before > 0 or context_after > 0:
        logger.info("context_requested", before=context_before, after=context_after)

    # Create reverse mapping: filepath -> file_id
    filepath_to_id = {v: k for k, v in file_ids.items()}

    # Get patterns list for cache operations
    patterns_list = list(pattern_ids.values())
    large_file_threshold = get_large_file_threshold_bytes()

    # Separate files into categories:
    # 1. Cached files (large files with valid cache)
    # 2. Seekable zstd files (parallel frame decompression)
    # 3. Compressed files (need sequential decompression)
    # 4. Regular files (can be chunked for parallel processing)
    files_to_process = []
    compressed_files = []  # List of compressed file paths
    seekable_zstd_files = []  # List of seekable zstd file paths to process
    seekable_zstd_cached = []  # (filepath, cache_info, file_size) for cache hits
    cached_files = []  # (filepath, cached_matches, file_size)

    for filepath in filepaths:
        try:
            file_size = os.path.getsize(filepath)
        except OSError:
            file_size = 0

        # Check if file is seekable zstd (prioritize over regular compressed)
        if is_seekable_zstd(filepath):
            # Check cache for seekable zstd files (if cache enabled)
            if use_cache:
                cache_info = get_compressed_cache_info(filepath, patterns_list, rg_extra_args)
                if cache_info is not None:
                    logger.info(
                        "seekable_zstd_cache_hit",
                        filepath=filepath,
                        match_count=cache_info["match_count"],
                        frame_count=len(cache_info["frames_with_matches"]),
                    )
                    seekable_zstd_cached.append((filepath, cache_info, file_size))
                    prom.trace_cache_hits_total.inc()
                    continue

            seekable_zstd_files.append((filepath, file_size))
            logger.info("detected_seekable_zstd_file", filepath=filepath)
            continue

        # Check if file is compressed
        if is_compressed(filepath):
            compressed_files.append((filepath, file_size))
            logger.info("detected_compressed_file", filepath=filepath)
            continue

        # Check cache for large files (if cache enabled)
        if use_cache and file_size >= large_file_threshold:
            cached_matches = get_cached_matches(filepath, patterns_list, rg_extra_args)
            if cached_matches is not None:
                logger.info("cache_hit", filepath=filepath, match_count=len(cached_matches))
                cached_files.append((filepath, cached_matches, file_size))
                continue

        files_to_process.append(filepath)

    # Create tasks from regular (non-compressed) files that need processing
    all_tasks = []
    file_chunk_counts = {}  # file_id -> number of chunks/workers

    for filepath in files_to_process:
        try:
            file_tasks = create_file_tasks(filepath)
            all_tasks.extend(file_tasks)

            # Track how many chunks this file was split into
            file_id = filepath_to_id.get(filepath)
            if file_id:
                file_chunk_counts[file_id] = len(file_tasks)
                if len(file_tasks) > 1:
                    logger.info("file_split_into_chunks", filepath=filepath, chunk_count=len(file_tasks))
        except Exception as e:
            logger.warning("skipping_file", filepath=filepath, error=str(e))

    # Mark cached files with chunk count of 0 (served from cache)
    for filepath, _, _ in cached_files:
        file_id = filepath_to_id.get(filepath)
        if file_id:
            file_chunk_counts[file_id] = 0  # 0 indicates cache hit

    # Mark seekable zstd files with their frame count (parallel processing)
    for filepath, _ in seekable_zstd_files:
        file_id = filepath_to_id.get(filepath)
        if file_id:
            # Get frame count for this file
            try:
                frames = read_seek_table(filepath)
                file_chunk_counts[file_id] = len(frames)
            except Exception:
                file_chunk_counts[file_id] = 1  # Fallback

    # Mark compressed files with chunk count of 1 (processed sequentially)
    for filepath, _ in compressed_files:
        file_id = filepath_to_id.get(filepath)
        if file_id:
            file_chunk_counts[file_id] = 1  # Compressed files are single-threaded

    logger.info(
        "tasks_created",
        task_count=len(all_tasks),
        file_count=len(files_to_process),
        cached_count=len(cached_files),
        seekable_zstd_cached_count=len(seekable_zstd_cached),
        seekable_zstd_count=len(seekable_zstd_files),
        compressed_count=len(compressed_files),
    )

    # Track parallel tasks created
    prom.parallel_tasks_created.observe(len(all_tasks))

    # Process cached files first - reconstruct matches
    matches = []
    all_context_lines = []
    total_time = 0.0

    for filepath, cached_matches, file_size in cached_files:
        file_id = filepath_to_id.get(filepath, 'f?')
        reconstruction_start = time.time()

        for cached_match in cached_matches:
            try:
                match_dict, ctx_lines = reconstruct_match_data(
                    filepath,
                    cached_match,
                    patterns_list,
                    pattern_ids,
                    file_id,
                    rg_extra_args,
                    context_before,
                    context_after,
                    use_index,
                )
                matches.append(match_dict)

                # Collect context lines with file_id association
                for ctx_line in ctx_lines:
                    all_context_lines.append((file_id, ctx_line))

                # Call on_match_found hook if configured
                if hooks and hooks.on_match_found:
                    try:
                        payload: dict = MatchFoundPayload(
                            request_id=hooks.request_id,
                            file_path=filepath,
                            pattern=patterns_list[cached_match.get('pattern_index', 0)],
                            offset=cached_match.get('offset', 0),
                            line_number=cached_match.get('line_number', 0),
                        ).model_dump()
                        hooks.on_match_found(payload)
                    except Exception as e:
                        logger.warning("on_match_found_hook_failed", error=str(e))

            except Exception as e:
                logger.warning("cache_match_reconstruction_failed", error=str(e))

        reconstruction_time = time.time() - reconstruction_start
        prom.trace_cache_reconstruction_seconds.observe(reconstruction_time)

        # Call on_file_scanned hook for cached file
        if hooks and hooks.on_file_scanned:
            try:
                payload = FileScannedPayload(
                    request_id=hooks.request_id,
                    file_path=filepath,
                    file_size_bytes=file_size,
                    scan_time_ms=int(reconstruction_time * 1000),
                    matches_count=len(cached_matches),
                ).model_dump()
                hooks.on_file_scanned(payload)
            except Exception as e:
                logger.warning("on_file_scanned_hook_failed", error=str(e))

        # Check max_results after cache reconstruction
        if max_results and len(matches) >= max_results:
            logger.info("max_results_reached_from_cache", max_results=max_results)

    # Process seekable zstd cached files - reconstruct matches from decompressed frames
    for filepath, cache_info, file_size in seekable_zstd_cached:
        if max_results and len(matches) >= max_results:
            logger.info("max_results_reached_skipping_seekable_zstd_cached", max_results=max_results)
            break

        file_id = filepath_to_id.get(filepath, 'f?')
        reconstruction_start = time.time()

        try:
            cached_matches = cache_info.get('matches', [])
            frames_with_matches = cache_info.get('frames_with_matches', [])

            # Reconstruct all matches from cached data (decompresses only needed frames)
            reconstructed_matches, reconstructed_context = reconstruct_seekable_zstd_matches(
                filepath,
                cached_matches,
                frames_with_matches,
                patterns_list,
                pattern_ids,
                file_id,
                rg_extra_args,
                context_before,
                context_after,
            )

            matches.extend(reconstructed_matches)

            # Collect context lines with file_id association
            for ctx_line in reconstructed_context:
                all_context_lines.append((file_id, ctx_line))

            # Call on_match_found hooks for each match
            if hooks and hooks.on_match_found:
                for match_dict in reconstructed_matches:
                    try:
                        payload: dict = MatchFoundPayload(
                            request_id=hooks.request_id,
                            file_path=filepath,
                            pattern=pattern_ids.get(match_dict.get('pattern', 'p1'), ''),
                            offset=match_dict.get('offset', 0),
                            line_number=match_dict.get('relative_line_number', 0),
                        ).model_dump()
                        hooks.on_match_found(payload)
                    except Exception as e:
                        logger.warning("on_match_found_hook_failed", error=str(e))

            reconstruction_time = time.time() - reconstruction_start

            # Call on_file_scanned hook
            if hooks and hooks.on_file_scanned:
                try:
                    payload = FileScannedPayload(
                        request_id=hooks.request_id,
                        file_path=filepath,
                        file_size_bytes=file_size,
                        scan_time_ms=int(reconstruction_time * 1000),
                        matches_count=len(reconstructed_matches),
                    ).model_dump()
                    hooks.on_file_scanned(payload)
                except Exception as e:
                    logger.warning("on_file_scanned_hook_failed", error=str(e))

            logger.info(
                "seekable_zstd_cache_reconstructed",
                filepath=filepath,
                match_count=len(reconstructed_matches),
                frame_count=len(frames_with_matches),
                elapsed_seconds=round(reconstruction_time, 3),
            )

        except Exception as e:
            logger.error("seekable_zstd_cache_reconstruction_failed", filepath=filepath, error=str(e))

    # Process compressed files (sequential decompression pipeline)
    for filepath, file_size in compressed_files:
        if max_results and len(matches) >= max_results:
            logger.info("max_results_reached_skipping_compressed", max_results=max_results)
            break

        file_id = filepath_to_id.get(filepath, 'f?')
        compression_start = time.time()

        try:
            # Calculate remaining results allowed
            remaining_results = None
            if max_results:
                remaining_results = max_results - len(matches)
                if remaining_results <= 0:
                    break

            compressed_matches, compressed_context, elapsed = process_compressed_file(
                filepath,
                pattern_ids,
                rg_extra_args,
                context_before,
                context_after,
                remaining_results,
            )
            total_time += elapsed

            # Convert compressed matches to API format
            for match_dict in compressed_matches:
                matching_pattern_ids = identify_matching_patterns(
                    match_dict['line_text'], match_dict['submatches'], pattern_ids, rg_extra_args
                )

                for matching_pattern_id in matching_pattern_ids:
                    match_entry = {
                        'pattern': matching_pattern_id,
                        'file': file_id,
                        'offset': match_dict['offset'],  # Decompressed byte offset
                        'relative_line_number': match_dict['line_number'],
                        'absolute_line_number': match_dict.get('absolute_line_number', -1),
                        'line_text': match_dict['line_text'],
                        'submatches': match_dict['submatches'],
                        'is_compressed': True,
                    }
                    matches.append(match_entry)

                    # Call on_match_found hook if configured
                    if hooks and hooks.on_match_found:
                        try:
                            payload: dict = MatchFoundPayload(
                                request_id=hooks.request_id,
                                file_path=filepath,
                                pattern=pattern_ids.get(matching_pattern_id, matching_pattern_id),
                                offset=match_dict['offset'],
                                line_number=match_dict['line_number'],
                            ).model_dump()
                            hooks.on_match_found(payload)
                        except Exception as e:
                            logger.warning("on_match_found_hook_failed", error=str(e))

            # Collect context lines with file_id association
            for ctx_line in compressed_context:
                all_context_lines.append((file_id, ctx_line))

            # Call on_file_scanned hook for compressed file
            if hooks and hooks.on_file_scanned:
                try:
                    payload = FileScannedPayload(
                        request_id=hooks.request_id,
                        file_path=filepath,
                        file_size_bytes=file_size,
                        scan_time_ms=int(elapsed * 1000),
                        matches_count=len(compressed_matches),
                    ).model_dump()
                    hooks.on_file_scanned(payload)
                except Exception as e:
                    logger.warning("on_file_scanned_hook_failed", error=str(e))

            logger.info("compressed_file_processed", filepath=filepath, match_count=len(compressed_matches), elapsed_seconds=round(elapsed, 3))

        except Exception as e:
            logger.error("compressed_file_processing_failed", filepath=filepath, error=str(e))

    # Process seekable zstd files (parallel frame decompression)
    for filepath, file_size in seekable_zstd_files:
        if max_results and len(matches) >= max_results:
            logger.info("max_results_reached_skipping_seekable_zstd", max_results=max_results)
            break

        file_id = filepath_to_id.get(filepath, 'f?')
        seekable_start = time.time()
        scan_completed = True  # Track if scan completed fully

        try:
            # Calculate remaining results allowed
            remaining_results = None
            if max_results:
                remaining_results = max_results - len(matches)
                if remaining_results <= 0:
                    break

            seekable_matches, seekable_context, elapsed = process_seekable_zstd_file(
                filepath,
                pattern_ids,
                rg_extra_args,
                context_before,
                context_after,
                remaining_results,
            )
            total_time += elapsed

            # Collect matches for caching (before converting to API format)
            matches_for_cache = []

            # Convert seekable zstd matches to API format
            for match_dict in seekable_matches:
                matching_pattern_ids = identify_matching_patterns(
                    match_dict['line_text'], match_dict['submatches'], pattern_ids, rg_extra_args
                )

                for matching_pattern_id in matching_pattern_ids:
                    match_entry = {
                        'pattern': matching_pattern_id,
                        'file': file_id,
                        'offset': match_dict['offset'],
                        'relative_line_number': match_dict['line_number'],
                        'absolute_line_number': match_dict.get('absolute_line_number', -1),
                        'line_text': match_dict['line_text'],
                        'submatches': match_dict['submatches'],
                        'frame_index': match_dict.get('frame_index'),  # Include for caching
                        'is_compressed': True,
                        'is_seekable_zstd': True,
                    }
                    matches.append(match_entry)
                    matches_for_cache.append(match_entry)

                    # Call on_match_found hook if configured
                    if hooks and hooks.on_match_found:
                        try:
                            payload: dict = MatchFoundPayload(
                                request_id=hooks.request_id,
                                file_path=filepath,
                                pattern=pattern_ids.get(matching_pattern_id, matching_pattern_id),
                                offset=match_dict['offset'],
                                line_number=match_dict['line_number'],
                            ).model_dump()
                            hooks.on_match_found(payload)
                        except Exception as e:
                            logger.warning("on_match_found_hook_failed", error=str(e))

            # Collect context lines with file_id association
            for ctx_line in seekable_context:
                all_context_lines.append((file_id, ctx_line))

            # Call on_file_scanned hook for seekable zstd file
            if hooks and hooks.on_file_scanned:
                try:
                    payload = FileScannedPayload(
                        request_id=hooks.request_id,
                        file_path=filepath,
                        file_size_bytes=file_size,
                        scan_time_ms=int(elapsed * 1000),
                        matches_count=len(seekable_matches),
                    ).model_dump()
                    hooks.on_file_scanned(payload)
                except Exception as e:
                    logger.warning("on_file_scanned_hook_failed", error=str(e))

            logger.info(
                "seekable_zstd_file_processed",
                filepath=filepath,
                match_count=len(seekable_matches),
                elapsed_seconds=round(elapsed, 3),
            )

            # Cache results for seekable zstd files (if cache enabled)
            if use_cache and should_cache_compressed_file(file_size, max_results, scan_completed):
                cache_data = build_cache_from_matches(
                    filepath,
                    patterns_list,
                    rg_extra_args,
                    matches_for_cache,
                    compression_format='zstd-seekable',
                )
                cache_path = get_trace_cache_path(filepath, patterns_list, rg_extra_args)
                save_trace_cache(cache_data, cache_path)
                logger.info(
                    "seekable_zstd_cache_written",
                    filepath=filepath,
                    match_count=len(matches_for_cache),
                )

        except Exception as e:
            logger.error("seekable_zstd_file_processing_failed", filepath=filepath, error=str(e))

    # Track per-file statistics for hooks and caching
    file_stats: dict[str, dict] = {}  # file_id -> {start_time, matches_count, file_size, ...}
    for file_id, filepath in file_ids.items():
        # Skip files already served from cache
        if any(fp == filepath for fp, _, _ in cached_files):
            continue
        # Skip seekable zstd cached files (already processed above)
        if any(fp == filepath for fp, _, _ in seekable_zstd_cached):
            continue
        # Skip compressed files (already processed above)
        if any(fp == filepath for fp, _ in compressed_files):
            continue
        # Skip seekable zstd files (already processed above)
        if any(fp == filepath for fp, _ in seekable_zstd_files):
            continue
        try:
            file_size = os.path.getsize(filepath)
        except OSError:
            file_size = 0
        file_stats[file_id] = {
            'start_time': time.time(),
            'matches_count': 0,
            'file_size': file_size,
            'filepath': filepath,
            'tasks_completed': 0,
            'tasks_total': file_chunk_counts.get(file_id, 1),
            'matches_for_cache': [],  # Collect matches for caching
        }

    # Track whether max_results was hit (affects caching)
    max_results_hit = max_results and len(matches) >= max_results

    if all_tasks and not max_results_hit:
        with ThreadPoolExecutor(max_workers=MAX_SUBPROCESSES, thread_name_prefix='Worker') as executor:
            future_to_task = {
                executor.submit(
                    process_task_worker, task, pattern_ids, rg_extra_args, context_before, context_after
                ): task
                for task in all_tasks
            }

            for future in as_completed(future_to_task):
                task = future_to_task[future]

                try:
                    task_result, match_dicts, context_lines, elapsed = future.result()
                    total_time += elapsed

                    # Get file_id for this task's filepath
                    file_id = filepath_to_id.get(task.filepath, 'f?')

                    # Convert match_dicts to API format
                    # Identify which patterns actually matched (a line may match multiple patterns)
                    for match_dict in match_dicts:
                        # Determine which patterns matched by analyzing the submatches
                        matching_pattern_ids = identify_matching_patterns(
                            match_dict['line_text'], match_dict['submatches'], pattern_ids, rg_extra_args
                        )

                        # Create one match per pattern that matched this line
                        for matching_pattern_id in matching_pattern_ids:
                            match_entry = {
                                'pattern': matching_pattern_id,
                                'file': file_id,
                                'offset': match_dict['offset'],
                                'relative_line_number': match_dict['line_number'],
                                'absolute_line_number': match_dict.get('absolute_line_number', -1),
                                'line_text': match_dict['line_text'],
                                'submatches': match_dict['submatches'],
                            }
                            matches.append(match_entry)

                            # Collect match for caching
                            if file_id in file_stats:
                                file_stats[file_id]['matches_for_cache'].append(match_entry)
                                file_stats[file_id]['matches_count'] += 1

                            # Call on_match_found hook if configured
                            if hooks and hooks.on_match_found:
                                try:
                                    payload: dict = MatchFoundPayload(
                                        request_id=hooks.request_id,
                                        file_path=file_stats[file_id]['filepath']
                                        if file_id in file_stats
                                        else task.filepath,
                                        pattern=pattern_ids.get(matching_pattern_id, matching_pattern_id),
                                        offset=match_dict['offset'],
                                        line_number=match_dict['line_number'],
                                    ).model_dump()
                                    hooks.on_match_found(payload)
                                except Exception as e:
                                    logger.warning("on_match_found_hook_failed", error=str(e))

                    # Collect context lines with file_id association
                    for ctx_line in context_lines:
                        # Add file_id as metadata for later grouping
                        ctx_line_with_file = (file_id, ctx_line)
                        all_context_lines.append(ctx_line_with_file)

                    # Track task completion for file
                    if file_id in file_stats:
                        file_stats[file_id]['tasks_completed'] += 1

                        # Check if all tasks for this file are complete
                        stats = file_stats[file_id]
                        if stats['tasks_completed'] >= stats['tasks_total']:
                            # Call on_file_scanned hook if configured
                            if hooks and hooks.on_file_scanned:
                                scan_time_ms = int((time.time() - stats['start_time']) * 1000)
                                try:
                                    payload = FileScannedPayload(
                                        request_id=hooks.request_id,
                                        file_path=stats['filepath'],
                                        file_size_bytes=stats['file_size'],
                                        scan_time_ms=scan_time_ms,
                                        matches_count=stats['matches_count'],
                                    ).model_dump()
                                    hooks.on_file_scanned(payload)
                                except Exception as e:
                                    logger.warning("on_file_scanned_hook_failed", error=str(e))

                    logger.debug(
                        "task_completed",
                        task_id=task.task_id,
                        filepath=task.filepath,
                        match_count=len(match_dicts),
                        context_line_count=len(context_lines),
                    )

                    # Check max_results
                    if max_results is not None and len(matches) >= max_results:
                        logger.info("max_results_reached_cancelling", max_results=max_results)
                        max_results_hit = True
                        for f in future_to_task:
                            f.cancel()
                        break

                except Exception as e:
                    logger.error("task_failed", error=str(e))

    # Write cache for large files that completed successfully
    # Only cache if no max_results limit was hit (partial scans shouldn't be cached)
    if use_cache and not max_results_hit:
        for file_id, stats in file_stats.items():
            filepath = stats['filepath']
            file_size = stats['file_size']

            # Check if this file should be cached
            if should_cache_file(file_size, max_results, stats['tasks_completed'] >= stats['tasks_total']):
                cache_data = build_cache_from_matches(
                    filepath,
                    patterns_list,
                    rg_extra_args,
                    stats['matches_for_cache'],
                )
                cache_path = get_trace_cache_path(filepath, patterns_list, rg_extra_args)
                save_trace_cache(cache_data, cache_path)
                logger.info(
                    "trace_cache_written",
                    filepath=filepath,
                    match_count=len(stats['matches_for_cache']),
                )

    # Sort matches by file, offset, then pattern
    matches.sort(key=lambda m: (m['file'], m['offset'], m['pattern']))

    # Apply max_results limit
    if max_results and len(matches) > max_results:
        matches = matches[:max_results]

    # Calculate absolute line numbers for matches that don't have them
    # This is needed for regular files processed in chunks where ripgrep only knows
    # the line number within the chunk, not the absolute line in the file

    # Group matches and context lines by file_id for batch processing
    matches_by_file: dict[str, list[dict]] = {}
    context_by_file: dict[str, list[tuple[int, ContextLine]]] = {}  # file_id -> [(index, ctx_line), ...]

    for match in matches:
        file_id = match['file']
        if file_id not in matches_by_file:
            matches_by_file[file_id] = []
        matches_by_file[file_id].append(match)

    for idx, (file_id, ctx_line) in enumerate(all_context_lines):
        if file_id not in context_by_file:
            context_by_file[file_id] = []
        context_by_file[file_id].append((idx, ctx_line))

    # For each file, calculate absolute line numbers for all matches and context lines in one pass
    for file_id in set(matches_by_file.keys()) | set(context_by_file.keys()):
        file_matches = matches_by_file.get(file_id, [])
        file_context = context_by_file.get(file_id, [])

        # Collect all offsets that need line numbers
        offsets_to_resolve: list[int] = []

        for match in file_matches:
            if match.get('absolute_line_number', -1) == -1 and not match.get('is_compressed'):
                offsets_to_resolve.append(match['offset'])

        for _, ctx_line in file_context:
            if ctx_line.absolute_line_number == -1:
                offsets_to_resolve.append(ctx_line.absolute_offset)

        if not offsets_to_resolve:
            continue

        # Get the filepath from file_ids mapping
        filepath = file_ids.get(file_id)
        if not filepath:
            continue

        # Skip compressed files (they have different offset semantics)
        if is_compressed(filepath):
            continue

        # Try to load unified index
        index_data = load_index(filepath)

        # Calculate all line numbers in a single file pass
        offset_to_line = calculate_lines_for_offsets_batch(filepath, offsets_to_resolve, index_data)

        # Update matches
        for match in file_matches:
            if match.get('absolute_line_number', -1) == -1 and not match.get('is_compressed'):
                abs_line = offset_to_line.get(match['offset'], -1)
                if abs_line != -1:
                    match['absolute_line_number'] = abs_line
                    match['relative_line_number'] = abs_line

        # Update context lines
        for idx, ctx_line in file_context:
            if ctx_line.absolute_line_number == -1:
                abs_line = offset_to_line.get(ctx_line.absolute_offset, -1)
                if abs_line != -1:
                    all_context_lines[idx] = (
                        file_id,
                        ContextLine(
                            relative_line_number=abs_line,
                            absolute_line_number=abs_line,
                            line_text=ctx_line.line_text,
                            absolute_offset=ctx_line.absolute_offset,
                        ),
                    )

    # Group context lines by match
    # Build composite key: "pattern:file:offset" -> [ContextLine, ...]
    context_dict = {}

    # Build a mapping of (file_id, line_number) -> match data for all matches
    # This helps us fill in matched lines that should appear in context but aren't in all_context_lines
    match_line_map = {(match['file'], match['relative_line_number']): match for match in matches}

    # For each match, build context lines
    for match in matches:
        match_line = match['relative_line_number']
        match_file = match['file']
        match_pattern = match['pattern']
        composite_key = f'{match_pattern}:{match_file}:{match["offset"]}'

        # Always create a ContextLine for the matched line itself
        matched_context_line = ContextLine(
            relative_line_number=match['relative_line_number'],
            absolute_line_number=match.get('absolute_line_number', -1),
            line_text=match['line_text'],
            absolute_offset=match['offset'],
        )

        if context_before > 0 or context_after > 0:
            # Find nearby context lines for this file near this match
            # Include ALL lines in the context range, even if they are matches
            # all_context_lines is now a list of (file_id, ContextLine) tuples
            nearby_context = [
                ctx_line
                for file_id, ctx_line in all_context_lines
                if (
                    file_id == match_file
                    and abs(ctx_line.relative_line_number - match_line) <= max(context_before, context_after)
                )
            ]

            # Also check if any matched lines fall in the context range but aren't in nearby_context
            # This can happen when a line matches multiple patterns
            context_line_numbers = {ctx.relative_line_number for ctx in nearby_context}
            for other_match in matches:
                if (
                    other_match['file'] == match_file
                    and other_match['relative_line_number'] != match_line
                    and abs(other_match['relative_line_number'] - match_line) <= max(context_before, context_after)
                    and other_match['relative_line_number'] not in context_line_numbers
                ):
                    # This matched line is in range but missing from context - add it
                    nearby_context.append(
                        ContextLine(
                            relative_line_number=other_match['relative_line_number'],
                            absolute_line_number=other_match.get('absolute_line_number', -1),
                            line_text=other_match['line_text'],
                            absolute_offset=other_match['offset'],
                        )
                    )

            # Combine context lines with the matched line and sort by line number
            all_lines = nearby_context + [matched_context_line]
            all_lines.sort(key=lambda ctx: ctx.relative_line_number)
            context_dict[composite_key] = all_lines
        else:
            # context=0: only show the matched line itself
            context_dict[composite_key] = [matched_context_line]

    logger.info(
        "parsing_completed",
        match_count=len(matches),
        context_group_count=len(context_dict),
        total_worker_time_seconds=round(total_time, 3),
    )

    return matches, context_dict, file_chunk_counts


def parse_paths(
    paths: list[str],
    regexps: list[str],
    max_results: int | None = None,
    rg_extra_args: list | None = None,
    context_before: int = 0,
    context_after: int = 0,
    hooks: HookCallbacks | None = None,
    use_cache: bool = True,
    use_index: bool = True,
) -> ParseResult:
    """
    Parse files or directories for multiple regex patterns.
    Returns ID-based structure with rich match data and optional context.

    Args:
        paths: List of file or directory paths to search
        regexps: List of regular expression patterns to search for
        max_results: Optional maximum number of results to find before stopping
        rg_extra_args: Optional list of extra arguments to pass to ripgrep
        context_before: Number of context lines before each match (0 = disabled)
        context_after: Number of context lines after each match (0 = disabled)
        hooks: Optional HookCallbacks for event notifications
        use_cache: Whether to use trace cache for reading/writing (default: True)
        use_index: Whether to use file indexes for faster processing (default: True)

    Returns:
        Dictionary with ID-based structure including rich match data and optional context.
    """
    if rg_extra_args is None:
        rg_extra_args = []

    pattern_ids = {f'p{i + 1}': pattern for i, pattern in enumerate(regexps)}

    # Update hooks with pattern info if provided
    if hooks:
        hooks.patterns = pattern_ids
    logger.info("processing_paths", pattern_count=len(pattern_ids), path_count=len(paths))

    # Collect all files to parse from all provided paths
    all_files_to_parse = []
    all_skipped_files = []
    all_scanned_dirs = []

    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f'Path not found: {path}')

        if os.path.isdir(path):
            logger.info("scanning_directory", path=path)
            text_files, skipped_files = scan_directory_for_text_files(path)

            if text_files:
                all_files_to_parse.extend(text_files)
                all_scanned_dirs.append(path)
            all_skipped_files.extend(skipped_files)
            logger.info("text_files_found", path=path, file_count=len(text_files))
        else:
            # Single file - validate and add to list
            try:
                validate_file(path)
                all_files_to_parse.append(path)
                logger.info("file_added", path=path)
            except ValueError as e:
                logger.warning("skipping_invalid_file", path=path, error=str(e))
                all_skipped_files.append(path)

    if not all_files_to_parse:
        logger.warning("no_valid_files_found")
        return ParseResult(
            patterns=pattern_ids,
            files={},
            matches=[],
            scanned_files=[],
            skipped_files=all_skipped_files,
            file_chunks={},
            context_lines={},
            before_context=context_before,
            after_context=context_after,
        )

    # Generate file IDs for all files
    file_ids = {f'f{i + 1}': filepath for i, filepath in enumerate(all_files_to_parse)}

    # Update hooks with file info if provided
    if hooks:
        hooks.files = file_ids

    # Parse all files
    logger.info("parsing_all_files", file_count=len(all_files_to_parse), pattern_count=len(pattern_ids))
    matches, context_dict, file_chunk_counts = parse_multiple_files_multipattern(
        all_files_to_parse,
        pattern_ids,
        file_ids,
        max_results,
        rg_extra_args,
        context_before,
        context_after,
        hooks,
        use_cache,
        use_index,
    )

    return ParseResult(
        patterns=pattern_ids,
        files=file_ids,
        matches=matches,
        scanned_files=all_files_to_parse if all_scanned_dirs else [],
        skipped_files=all_skipped_files,
        file_chunks=file_chunk_counts,
        context_lines=context_dict,
        before_context=context_before,
        after_context=context_after,
    )
