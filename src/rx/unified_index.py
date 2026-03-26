"""Unified index cache management.

This module provides cache management for the unified file index system.
All indexes are stored in $RX_CACHE_DIR/indexes/ (or ~/.cache/rx/indexes/).

Cache behavior:
- Without --analyze: Only cache large files (>=50MB) with line index only
- With --analyze: Cache ALL files with full analysis + anomaly detection
- Cache invalidation: Based on source file mtime and size
- Cache rebuild: When --analyze requested but analysis_performed=False
"""

import bisect
import hashlib
import json
import structlog
import os
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from rx.cli import prometheus as prom
from rx.models import UnifiedFileIndex
from rx.utils import get_rx_cache_dir


logger = structlog.get_logger()

# Cache format version - increment when format changes
UNIFIED_INDEX_VERSION = 2

# Default threshold for creating indexes without --analyze (50MB)
DEFAULT_LARGE_FILE_MB = 50


@dataclass
class LineInfo:
    """Information about a line containing a target offset."""

    line_number: int  # 1-based line number
    line_start_offset: int  # Byte offset where the line starts
    line_end_offset: int  # Byte offset where the line ends (after newline)


@dataclass
class IndexBuildResult:
    """Result of building an index."""

    line_index: list[list[int]]  # [[line_number, byte_offset], ...]
    line_count: int
    empty_line_count: int
    line_length_max: int
    line_length_avg: float
    line_length_median: float
    line_length_p95: float
    line_length_p99: float
    line_length_stddev: float
    line_length_max_line_number: int
    line_length_max_byte_offset: int
    line_ending: str


def _percentile(data: list[int], p: float) -> float:
    """Calculate the p-th percentile of data."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    n = len(sorted_data)
    k = (n - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < n else f
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def _detect_line_ending(sample_bytes: bytes) -> str:
    """Detect line ending style from a sample of bytes."""
    crlf_count = sample_bytes.count(b'\r\n')
    cr_count = sample_bytes.count(b'\r') - crlf_count
    lf_count = sample_bytes.count(b'\n') - crlf_count

    endings = []
    if crlf_count > 0:
        endings.append(('CRLF', crlf_count))
    if lf_count > 0:
        endings.append(('LF', lf_count))
    if cr_count > 0:
        endings.append(('CR', cr_count))

    if len(endings) == 0:
        return 'LF'  # Default
    elif len(endings) == 1:
        return endings[0][0]
    else:
        return 'mixed'


def build_index(source_path: str, step_bytes: int | None = None) -> IndexBuildResult:
    """Build a line-offset index for a file.

    Creates index entries at approximately every step_bytes interval,
    with all offsets aligned to line starts.

    Args:
        source_path: Path to the source file
        step_bytes: Bytes between index entries (default: from config)

    Returns:
        IndexBuildResult with line index and analysis data
    """
    start_time = time.time()

    if step_bytes is None:
        step_bytes = get_index_step_bytes()

    line_index: list[list[int]] = [[1, 0]]  # First line always at offset 0

    current_offset = 0
    current_line = 0
    next_checkpoint = step_bytes

    line_lengths: list[int] = []
    empty_line_count = 0
    max_line_length = 0
    max_line_number = 0
    max_line_offset = 0

    # Sample for line ending detection (first 64KB)
    line_ending_sample = b''
    sample_collected = False

    with open(source_path, 'rb') as f:
        for line in f:
            current_line += 1
            line_len_bytes = len(line)

            # Collect sample for line ending detection
            if not sample_collected:
                line_ending_sample += line
                if len(line_ending_sample) >= 65536:
                    sample_collected = True

            # Strip line ending for length calculation
            stripped = line.rstrip(b'\r\n')
            content_len = len(stripped)

            # Track line statistics (for non-empty lines)
            if stripped.strip():  # Has non-whitespace content
                line_lengths.append(content_len)
                if content_len > max_line_length:
                    max_line_length = content_len
                    max_line_number = current_line
                    max_line_offset = current_offset
            else:
                empty_line_count += 1

            current_offset += line_len_bytes

            # Check if we've passed the next checkpoint
            if current_offset >= next_checkpoint:
                # Record the start of the NEXT line
                # current_offset is now at the start of the next line
                line_index.append([current_line + 1, current_offset])
                next_checkpoint = current_offset + step_bytes

    # Calculate statistics
    if line_lengths:
        line_length_avg = statistics.mean(line_lengths)
        line_length_median = statistics.median(line_lengths)
        line_length_p95 = _percentile(line_lengths, 95)
        line_length_p99 = _percentile(line_lengths, 99)
        line_length_stddev = statistics.stdev(line_lengths) if len(line_lengths) > 1 else 0.0
    else:
        line_length_avg = 0.0
        line_length_median = 0.0
        line_length_p95 = 0.0
        line_length_p99 = 0.0
        line_length_stddev = 0.0

    line_ending = _detect_line_ending(line_ending_sample)

    # Record metrics
    prom.index_build_duration_seconds.observe(time.time() - start_time)

    return IndexBuildResult(
        line_index=line_index,
        line_count=current_line,
        empty_line_count=empty_line_count,
        line_length_max=max_line_length,
        line_length_avg=line_length_avg,
        line_length_median=line_length_median,
        line_length_p95=line_length_p95,
        line_length_p99=line_length_p99,
        line_length_stddev=line_length_stddev,
        line_length_max_line_number=max_line_number,
        line_length_max_byte_offset=max_line_offset,
        line_ending=line_ending,
    )


def get_index_cache_dir() -> Path:
    """Get the unified index cache directory path.

    Returns:
        Path to $RX_CACHE_DIR/indexes/ (or ~/.cache/rx/indexes/)
    """
    return get_rx_cache_dir('indexes')


def get_large_file_threshold_bytes() -> int:
    """Get the threshold in bytes for creating indexes without --analyze.

    Controlled by RX_LARGE_FILE_MB environment variable.
    Default: 50 MB (DEFAULT_LARGE_FILE_MB constant)

    Returns:
        Threshold in bytes for considering a file "large" enough to index
    """
    threshold_mb = os.environ.get('RX_LARGE_FILE_MB')
    if threshold_mb:
        try:
            mb = int(threshold_mb)
            if mb > 0:
                return mb * 1024 * 1024
        except ValueError:
            pass
    return DEFAULT_LARGE_FILE_MB * 1024 * 1024


def get_index_step_bytes() -> int:
    """Get the index step size in bytes.

    Step size is threshold / 50.
    Default: 1MB (when threshold is 50MB)
    """
    return get_large_file_threshold_bytes() // 50


def get_cache_key(file_path: str) -> str:
    """Generate cache key for a file.

    Uses SHA256 hash of absolute path (first 16 chars) plus filename
    to create a unique but identifiable cache filename.

    Args:
        file_path: Absolute path to file

    Returns:
        Cache key (filename-safe string)
    """
    abs_path = os.path.abspath(file_path)
    path_hash = hashlib.sha256(abs_path.encode('utf-8')).hexdigest()[:16]
    basename = os.path.basename(file_path)
    # Sanitize basename to be filename-safe
    safe_basename = ''.join(c if c.isalnum() or c in '._-' else '_' for c in basename)
    return f'{safe_basename}_{path_hash}'


def get_index_path(source_path: str) -> Path:
    """Get the cache file path for a given source file.

    Args:
        source_path: Absolute path to source file

    Returns:
        Path to cache file in index directory
    """
    cache_dir = get_index_cache_dir()
    cache_key = get_cache_key(source_path)
    return cache_dir / f'{cache_key}.json'


def is_index_valid(source_path: str, index: UnifiedFileIndex) -> bool:
    """Check if cached index is still valid for the source file.

    Index is valid if source file size and mtime haven't changed.

    Args:
        source_path: Absolute path to source file
        index: Cached index to validate

    Returns:
        True if index is valid, False otherwise
    """
    try:
        stat = os.stat(source_path)
        current_mtime = datetime.fromtimestamp(stat.st_mtime).isoformat()
        return index.source_modified_at == current_mtime and index.source_size_bytes == stat.st_size
    except OSError:
        return False


def needs_rebuild(source_path: str, index: UnifiedFileIndex | None, analyze: bool) -> bool:
    """Check if index needs to be rebuilt.

    Rebuild is needed if:
    - No index exists
    - Index is invalid (source file changed)
    - analyze=True but index.analysis_performed=False

    Args:
        source_path: Absolute path to source file
        index: Existing index (or None)
        analyze: Whether full analysis is requested

    Returns:
        True if rebuild needed, False otherwise
    """
    if index is None:
        return True

    if not is_index_valid(source_path, index):
        return True

    if analyze and not index.analysis_performed:
        return True

    return False


def load_index(source_path: str) -> UnifiedFileIndex | None:
    """Load cached index for a source file.

    Args:
        source_path: Absolute path to source file

    Returns:
        UnifiedFileIndex if valid cache exists, None otherwise
    """
    cache_path = get_index_path(source_path)

    if not cache_path.exists():
        logger.debug("cache_not_found", path=source_path)
        return None

    try:
        with open(cache_path, encoding='utf-8') as f:
            data = json.load(f)

        # Check version
        if data.get('version') != UNIFIED_INDEX_VERSION:
            logger.debug("cache_version_mismatch", path=source_path)
            return None

        index = UnifiedFileIndex(**data)

        # Validate against source file
        if not is_index_valid(source_path, index):
            logger.debug("cache_invalid", path=source_path, reason="file_changed")
            return None

        logger.debug("cache_hit", path=source_path)
        return index

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.warning("cache_load_failed", path=source_path, error=str(e))
        return None


def save_index(index: UnifiedFileIndex) -> bool:
    """Save index to cache.

    Args:
        index: UnifiedFileIndex to save

    Returns:
        True if saved successfully, False otherwise
    """
    try:
        cache_path = get_index_path(index.source_path)

        # Use compact JSON for large indexes (no indent)
        anomaly_count = len(index.anomalies) if index.anomalies else 0
        line_index_count = len(index.line_index)
        use_compact = anomaly_count > 1000 or line_index_count > 10000

        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(index.model_dump(mode='json'), f, indent=None if use_compact else 2)

        cache_size = cache_path.stat().st_size
        logger.info(
            "cache_saved",
            path=index.source_path,
            cache_size=cache_size,
            anomaly_count=anomaly_count,
            checkpoint_count=line_index_count,
        )
        return True

    except Exception as e:
        logger.warning("cache_save_failed", path=index.source_path, error=str(e), error_type=type(e).__name__)
        return False


def delete_index(source_path: str) -> bool:
    """Delete cached index for a source file.

    Args:
        source_path: Absolute path to source file

    Returns:
        True if deleted, False if didn't exist or error
    """
    cache_path = get_index_path(source_path)

    try:
        if cache_path.exists():
            cache_path.unlink()
            logger.info("cache_deleted", path=source_path)
            return True
        return False
    except OSError as e:
        logger.warning("cache_delete_failed", path=source_path, error=str(e))
        return False


def clear_all_indexes() -> int:
    """Clear all cached indexes.

    Returns:
        Number of cache files deleted
    """
    cache_dir = get_index_cache_dir()
    count = 0

    for cache_file in cache_dir.glob('*.json'):
        try:
            cache_file.unlink()
            count += 1
        except OSError as e:
            logger.warning("cache_file_delete_failed", path=str(cache_file), error=str(e))

    logger.info("cache_cleared", deleted_count=count)
    return count


def should_create_index(file_size_bytes: int, analyze: bool) -> bool:
    """Determine if an index should be created for a file.

    Args:
        file_size_bytes: Size of the file in bytes
        analyze: Whether full analysis is requested

    Returns:
        True if index should be created, False otherwise
    """
    # Always create index with --analyze
    if analyze:
        return True

    # Without --analyze, only index large files
    return file_size_bytes >= get_large_file_threshold_bytes()


def get_cached_line_count(source_path: str) -> int | None:
    """Get line count from cached index if available.

    This is a convenience function for callers that just need line count
    without loading the full index.

    Args:
        source_path: Absolute path to source file

    Returns:
        Line count if cached and valid, None otherwise
    """
    index = load_index(source_path)
    if index and index.line_count:
        return index.line_count
    return None


def find_line_offset(line_index: list[list[int]], target_line: int) -> tuple[int, int]:
    """Find the closest indexed line before or at target_line.

    Uses binary search for efficient lookup.

    Args:
        line_index: List of [line_number, byte_offset] pairs
        target_line: The line number to find

    Returns:
        Tuple of (line_number, byte_offset) for the closest previous indexed line
    """
    if not line_index:
        return (1, 0)

    # Extract line numbers for binary search
    lines = [entry[0] for entry in line_index]

    # Find rightmost entry with line <= target_line
    idx = bisect.bisect_right(lines, target_line) - 1
    if idx < 0:
        idx = 0

    return (line_index[idx][0], line_index[idx][1])


def calculate_exact_offset_for_line(filename: str, target_line: int, index: UnifiedFileIndex | None = None) -> int:
    """Calculate the exact byte offset for a given line number.

    Args:
        filename: Path to the file
        target_line: Line number (1-based) to find offset for
        index: Optional UnifiedFileIndex. If None, will try to load

    Returns:
        Byte offset of the line, or -1 if cannot determine (large file without index)
    """
    # If no index provided, try to load it
    if index is None:
        index = load_index(filename)

    # If we have an index, use it
    if index and index.line_index:
        line_index = index.line_index

        # Find closest indexed line
        indexed_line, indexed_offset = find_line_offset(line_index, target_line)

        # If exact match, return it
        if indexed_line == target_line:
            return indexed_offset

        # Read from indexed position and count to target
        try:
            with open(filename, 'rb') as f:
                f.seek(indexed_offset)
                current_line = indexed_line
                current_offset = indexed_offset

                for line_bytes in f:
                    if current_line == target_line:
                        return current_offset
                    current_offset += len(line_bytes)
                    current_line += 1

                # Reached EOF before finding target line
                return -1
        except OSError as e:
            logger.error("file_read_failed", path=filename, error=str(e))
            return -1

    # No index - check if file is small enough to read
    try:
        file_size = os.path.getsize(filename)
        threshold = get_large_file_threshold_bytes()

        if file_size > threshold:
            # Large file without index - cannot determine
            return -1

        # Small file - read from beginning
        with open(filename, 'rb') as f:
            current_line = 0
            current_offset = 0

            for line_bytes in f:
                current_line += 1
                if current_line == target_line:
                    return current_offset
                current_offset += len(line_bytes)

            # Target line beyond EOF
            return -1
    except OSError as e:
        logger.error("file_process_failed", path=filename, error=str(e))
        return -1


def calculate_exact_line_for_offset(filename: str, target_offset: int, index: UnifiedFileIndex | None = None) -> int:
    """Calculate the exact line number for a given byte offset.

    Args:
        filename: Path to the file
        target_offset: Byte offset to find line number for
        index: Optional UnifiedFileIndex. If None, will try to load

    Returns:
        Line number (1-based) at the offset, or -1 if cannot determine
    """
    # If no index provided, try to load it
    if index is None:
        index = load_index(filename)

    # If we have an index, use it
    if index and index.line_index:
        line_index = index.line_index

        # Find closest indexed line before target offset (binary search by offset)
        offsets = [entry[1] for entry in line_index]
        idx = bisect.bisect_right(offsets, target_offset) - 1
        if idx < 0:
            idx = 0

        indexed_line, indexed_offset = line_index[idx]

        # If exact match, return it
        if indexed_offset == target_offset:
            return indexed_line

        # Read from indexed position and count lines to target offset
        try:
            with open(filename, 'rb') as f:
                f.seek(indexed_offset)
                current_line = indexed_line
                current_offset = indexed_offset

                for line_bytes in f:
                    if current_offset == target_offset:
                        return current_line
                    if current_offset + len(line_bytes) > target_offset:
                        # Target offset is within this line
                        return current_line
                    current_offset += len(line_bytes)
                    current_line += 1

                # Reached EOF
                return -1
        except OSError as e:
            logger.error("file_read_failed", path=filename, error=str(e))
            return -1

    # No index - check if file is small enough to read
    try:
        file_size = os.path.getsize(filename)
        threshold = get_large_file_threshold_bytes()

        if file_size > threshold:
            # Large file without index - cannot determine
            return -1

        # Small file - read from beginning
        with open(filename, 'rb') as f:
            current_line = 0
            current_offset = 0

            for line_bytes in f:
                current_line += 1
                if current_offset == target_offset:
                    return current_line
                if current_offset + len(line_bytes) > target_offset:
                    # Target offset is within this line
                    return current_line
                current_offset += len(line_bytes)

            # EOF
            return -1
    except OSError as e:
        logger.error("file_process_failed", path=filename, error=str(e))
        return -1


def calculate_lines_for_offsets_batch(
    filename: str, target_offsets: list[int], index: UnifiedFileIndex | None = None
) -> dict[int, int]:
    """Calculate line numbers for multiple byte offsets in a single file pass.

    This is much more efficient than calling calculate_exact_line_for_offset
    multiple times, as it reads the file only once.

    Args:
        filename: Path to the file
        target_offsets: List of byte offsets to find line numbers for
        index: Optional UnifiedFileIndex. If None, will try to load

    Returns:
        Dictionary mapping offset -> line_number (or -1 if cannot determine)
    """
    if not target_offsets:
        return {}

    # If no index provided, try to load it
    if index is None:
        index = load_index(filename)

    # Sort offsets to process them in order (single pass through file)
    sorted_offsets = sorted(set(target_offsets))
    results: dict[int, int] = {offset: -1 for offset in target_offsets}

    if not index:
        # No index - check if file is small enough to read
        try:
            file_size = os.path.getsize(filename)
            threshold = get_large_file_threshold_bytes()
            if file_size > threshold:
                return results  # Large file without index - cannot determine
        except OSError:
            return results

    # Find the best starting point from index
    line_index = index.line_index if index and index.line_index else [[1, 0]]

    # Find the indexed position before the first offset we need
    first_offset = sorted_offsets[0]
    offsets_in_index = [entry[1] for entry in line_index]
    idx = bisect.bisect_right(offsets_in_index, first_offset) - 1
    if idx < 0:
        idx = 0

    start_line, start_offset = line_index[idx]

    # Read file once and calculate all line numbers
    try:
        with open(filename, 'rb') as f:
            f.seek(start_offset)
            current_line = start_line
            current_offset = start_offset
            offset_idx = 0  # Index into sorted_offsets

            # Skip offsets that are before our start position
            while offset_idx < len(sorted_offsets) and sorted_offsets[offset_idx] < start_offset:
                offset_idx += 1

            for line_bytes in f:
                line_end_offset = current_offset + len(line_bytes)

                # Check all offsets that fall within this line
                while offset_idx < len(sorted_offsets):
                    target = sorted_offsets[offset_idx]
                    if target < current_offset:
                        # This shouldn't happen if we started correctly
                        offset_idx += 1
                    elif current_offset <= target < line_end_offset:
                        # This offset is within the current line
                        results[target] = current_line
                        offset_idx += 1
                    else:
                        # Target is beyond this line, move to next line
                        break

                # If we've found all offsets, stop reading
                if offset_idx >= len(sorted_offsets):
                    break

                current_offset = line_end_offset
                current_line += 1

    except OSError as e:
        logger.error("file_read_failed", path=filename, error=str(e))

    return results


def calculate_line_info_for_offsets(
    filename: str, target_offsets: list[int], index: UnifiedFileIndex | None = None
) -> dict[int, LineInfo]:
    """Calculate line information for multiple byte offsets in a single file pass.

    This function returns comprehensive information about each line containing
    the target offsets, including line number and the byte offsets where the
    line starts and ends. This enables efficient seek-based reading.

    Args:
        filename: Path to the file
        target_offsets: List of byte offsets to find line info for
        index: Optional UnifiedFileIndex. If None, will try to load

    Returns:
        Dictionary mapping offset -> LineInfo (line_number, line_start_offset, line_end_offset)
        Returns empty LineInfo with -1 values if cannot determine.
    """
    if not target_offsets:
        return {}

    # Default result for failures
    default_info = LineInfo(line_number=-1, line_start_offset=-1, line_end_offset=-1)
    results: dict[int, LineInfo] = {offset: default_info for offset in target_offsets}

    # If no index provided, try to load it
    if index is None:
        index = load_index(filename)

    # Sort offsets to process them in order (single pass through file)
    sorted_offsets = sorted(set(target_offsets))

    if not index:
        # No index - check if file is small enough to read
        try:
            file_size = os.path.getsize(filename)
            threshold = get_large_file_threshold_bytes()
            if file_size > threshold:
                return results  # Large file without index - cannot determine
        except OSError:
            return results

    # Find the best starting point from index
    line_index = index.line_index if index and index.line_index else [[1, 0]]

    # Find the indexed position before the first offset we need
    first_offset = sorted_offsets[0]
    offsets_in_index = [entry[1] for entry in line_index]
    idx = bisect.bisect_right(offsets_in_index, first_offset) - 1
    if idx < 0:
        idx = 0

    start_line, start_offset = line_index[idx]

    # Read file once and calculate all line info
    try:
        with open(filename, 'rb') as f:
            f.seek(start_offset)
            current_line = start_line
            current_offset = start_offset
            offset_idx = 0  # Index into sorted_offsets

            # Skip offsets that are before our start position
            while offset_idx < len(sorted_offsets) and sorted_offsets[offset_idx] < start_offset:
                offset_idx += 1

            for line_bytes in f:
                line_end_offset = current_offset + len(line_bytes)

                # Check all offsets that fall within this line
                while offset_idx < len(sorted_offsets):
                    target = sorted_offsets[offset_idx]
                    if target < current_offset:
                        # This shouldn't happen if we started correctly
                        offset_idx += 1
                    elif current_offset <= target < line_end_offset:
                        # This offset is within the current line
                        results[target] = LineInfo(
                            line_number=current_line,
                            line_start_offset=current_offset,
                            line_end_offset=line_end_offset,
                        )
                        offset_idx += 1
                    else:
                        # Target is beyond this line, move to next line
                        break

                # If we've found all offsets, stop reading
                if offset_idx >= len(sorted_offsets):
                    break

                current_offset = line_end_offset
                current_line += 1

    except OSError as e:
        logger.error("file_read_failed", path=filename, error=str(e))

    return results
