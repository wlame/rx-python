"""Compressed file index management for random access.

This module manages decompression indexes that enable efficient random access
to compressed files. Without an index, getting content from a specific line
in a compressed file requires decompressing from the beginning.

Index Storage: $RX_CACHE_DIR/indexes/ (or ~/.cache/rx/indexes/)

Index Structure:
- Metadata about the source compressed file (path, mtime, size)
- Compression format
- Line-to-decompressed-offset mapping (sparse, sampled)
- Optional: external index files for tools like indexed_gzip
"""

import hashlib
import logging
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from rx.compression import (
    CompressionFormat,
    decompress_to_stdout,
    detect_compression,
)
from rx.utils import get_rx_cache_dir


logger = logging.getLogger(__name__)

# Line sampling interval for building line index
# We store line number -> decompressed byte offset for every Nth line
LINE_SAMPLE_INTERVAL = 1000

# Import unified index version and model for consistency
# Type hint import (avoid circular import at runtime)
from typing import TYPE_CHECKING

from rx.unified_index import UNIFIED_INDEX_VERSION


if TYPE_CHECKING:
    from rx.models import UnifiedFileIndex


def get_compressed_index_dir() -> Path:
    """Get the directory for storing compressed file indexes.

    Returns:
        Path to the unified index directory
    """
    return get_rx_cache_dir('indexes')


def get_compressed_index_path(source_path: str | Path) -> Path:
    """Get the path to a compressed file's index.

    The index filename uses the unified format: {filename}_{hash}.json
    This matches the format used by unified_index.py for consistency.

    Args:
        source_path: Path to the compressed file

    Returns:
        Path to the index file
    """
    source_path = Path(source_path).resolve()
    path_hash = hashlib.sha256(str(source_path).encode()).hexdigest()[:16]
    filename = source_path.name
    # Sanitize filename to be safe for filesystem
    safe_filename = ''.join(c if c.isalnum() or c in '._-' else '_' for c in filename)

    index_dir = get_compressed_index_dir()
    # Use unified format: {filename}_{hash}.json (not legacy {hash}_{filename}.json)
    return index_dir / f'{safe_filename}_{path_hash}.json'


def build_compressed_index(
    source_path: str | Path,
    progress_callback: Callable | None = None,
) -> dict:
    """Build a decompression + line index for a compressed file.

    This decompresses the entire file while tracking:
    - Total decompressed size
    - Line number to decompressed byte offset mapping (sampled)
    - Line statistics (count, empty, lengths, etc.)

    Args:
        source_path: Path to the compressed file
        progress_callback: Optional callback(bytes_processed, total_bytes) for progress

    Returns:
        Index data dictionary

    Raises:
        ValueError: If file is not compressed
        RuntimeError: If decompression fails
    """
    import statistics

    source_path = Path(source_path).resolve()

    compression_format = detect_compression(source_path)
    if compression_format == CompressionFormat.NONE:
        raise ValueError(f'File is not compressed: {source_path}')

    logger.info(f'Building compressed index for {source_path}')
    start_time = time.time()

    # Get source file metadata
    stat = source_path.stat()
    source_mtime = datetime.fromtimestamp(stat.st_mtime).isoformat()
    source_size = stat.st_size

    # Start decompression
    proc = decompress_to_stdout(source_path, compression_format)

    # Read and count lines, building sparse line index
    line_index = []  # List of [line_number, decompressed_offset]
    current_line = 0  # Will be incremented to 1 when first line is complete
    current_offset = 0
    line_start_offset = 0  # Track where current line started

    # Statistics tracking
    line_lengths: list[int] = []
    empty_line_count = 0
    max_line_length = 0
    max_line_number = 0
    max_line_offset = 0
    line_ending = None  # Will detect LF vs CRLF
    prev_byte = None

    # Buffer for partial lines across chunks
    partial_line_length = 0

    try:
        while True:
            chunk = proc.stdout.read(65536)  # 64KB chunks
            if not chunk:
                break

            # Process each byte position in chunk
            chunk_offset = 0
            while chunk_offset < len(chunk):
                newline_pos = chunk.find(b'\n', chunk_offset)
                if newline_pos == -1:
                    # No more newlines in this chunk
                    partial_line_length += len(chunk) - chunk_offset
                    break

                # Found a newline - complete this line
                current_line += 1
                line_length = partial_line_length + (newline_pos - chunk_offset)

                # Check for CRLF
                if newline_pos > 0 and chunk[newline_pos - 1 : newline_pos] == b'\r':
                    line_length -= 1  # Don't count \r in line length
                    if line_ending is None:
                        line_ending = 'CRLF'
                elif prev_byte == b'\r' and newline_pos == 0:
                    line_length -= 1
                    if line_ending is None:
                        line_ending = 'CRLF'
                elif line_ending is None:
                    line_ending = 'LF'

                # Track statistics
                line_lengths.append(line_length)
                if line_length == 0:
                    empty_line_count += 1
                if line_length > max_line_length:
                    max_line_length = line_length
                    max_line_number = current_line
                    max_line_offset = line_start_offset

                # Add first line checkpoint
                if current_line == 1:
                    line_index.append([1, 0])

                # Sample every Nth line
                elif current_line % LINE_SAMPLE_INTERVAL == 0:
                    line_index.append([current_line, current_offset + newline_pos + 1])

                # Move to next line
                line_start_offset = current_offset + newline_pos + 1
                partial_line_length = 0
                chunk_offset = newline_pos + 1

            # Track last byte for CRLF detection across chunks
            prev_byte = chunk[-1:] if chunk else None
            current_offset += len(chunk)

            if progress_callback:
                progress_callback(current_offset, None)

        proc.wait()

        if proc.returncode != 0:
            stderr = proc.stderr.read().decode() if proc.stderr else ''
            raise RuntimeError(f'Decompression failed: {stderr}')

        # Handle final line without newline
        if partial_line_length > 0:
            current_line += 1
            line_lengths.append(partial_line_length)
            if partial_line_length == 0:
                empty_line_count += 1
            if partial_line_length > max_line_length:
                max_line_length = partial_line_length
                max_line_number = current_line
                max_line_offset = line_start_offset

    except Exception as e:
        proc.kill()
        raise RuntimeError(f'Failed to build index: {e}')

    elapsed = time.time() - start_time
    total_lines = current_line
    decompressed_size = current_offset

    # Compute statistics
    if line_lengths:
        line_length_avg = statistics.mean(line_lengths)
        line_length_median = statistics.median(line_lengths)
        line_length_stddev = statistics.stdev(line_lengths) if len(line_lengths) > 1 else 0.0
        sorted_lengths = sorted(line_lengths)
        p95_idx = int(len(sorted_lengths) * 0.95)
        p99_idx = int(len(sorted_lengths) * 0.99)
        line_length_p95 = sorted_lengths[min(p95_idx, len(sorted_lengths) - 1)]
        line_length_p99 = sorted_lengths[min(p99_idx, len(sorted_lengths) - 1)]
    else:
        line_length_avg = 0.0
        line_length_median = 0.0
        line_length_stddev = 0.0
        line_length_p95 = 0
        line_length_p99 = 0

    logger.info(
        f'Built index for {source_path}: {total_lines} lines, '
        f'{decompressed_size} bytes decompressed, {len(line_index)} checkpoints, '
        f'{elapsed:.2f}s'
    )

    index_data = {
        'version': UNIFIED_INDEX_VERSION,
        'source_path': str(source_path),
        'source_modified_at': source_mtime,
        'source_size_bytes': source_size,
        'compression_format': compression_format.value,
        'decompressed_size_bytes': decompressed_size,
        'total_lines': total_lines,
        'line_sample_interval': LINE_SAMPLE_INTERVAL,
        'line_index': line_index,
        'created_at': datetime.now().isoformat(),
        'build_time_seconds': elapsed,
        # Statistics
        'empty_line_count': empty_line_count,
        'line_ending': line_ending or 'LF',
        'line_length_max': max_line_length,
        'line_length_avg': round(line_length_avg, 1),
        'line_length_median': round(line_length_median, 1),
        'line_length_p95': line_length_p95,
        'line_length_p99': line_length_p99,
        'line_length_stddev': round(line_length_stddev, 1),
        'line_length_max_line_number': max_line_number,
        'line_length_max_byte_offset': max_line_offset,
    }

    return index_data


def find_nearest_checkpoint(line_index: list[list[int]], target_line: int) -> tuple[int, int]:
    """Find the nearest checkpoint at or before the target line.

    Args:
        line_index: List of [line_number, byte_offset] checkpoints
        target_line: Target line number (1-indexed)

    Returns:
        Tuple of (checkpoint_line, checkpoint_offset)
    """
    if not line_index:
        line_index = [[1, 0]]

    # Binary search for nearest checkpoint <= target_line
    left, right = 0, len(line_index) - 1
    result = line_index[0]

    while left <= right:
        mid = (left + right) // 2
        checkpoint_line, checkpoint_offset = line_index[mid]

        if checkpoint_line <= target_line:
            result = line_index[mid]
            left = mid + 1
        else:
            right = mid - 1

    return result[0], result[1]


def get_decompressed_lines(
    source_path: str | Path,
    start_line: int,
    count: int = 1,
    index: 'UnifiedFileIndex | None' = None,
) -> list[str]:
    """Get specific lines from a compressed file.

    Uses the index to seek to a nearby checkpoint, then decompresses
    forward to the target lines.

    Args:
        source_path: Path to the compressed file
        start_line: Starting line number (1-indexed)
        count: Number of lines to retrieve
        index: Optional pre-loaded UnifiedFileIndex.
               If not provided, will load from cache or build new.

    Returns:
        List of line strings (without newlines)
    """

    source_path = Path(source_path)

    # Get index from cache or build
    if index is None:
        from rx.unified_index import load_index

        index = load_index(str(source_path))
        if index is None:
            # No cached index, build on the fly and wrap in UnifiedFileIndex
            from rx.indexer import FileIndexer

            # Use analyze=True to ensure index is always built (even for small files)
            indexer = FileIndexer(analyze=True)
            index = indexer.index_file(str(source_path))
            if index is None:
                raise RuntimeError(f'Failed to build index for {source_path}')

    # Find nearest checkpoint
    line_index = index.line_index or [[1, 0]]
    checkpoint_line, checkpoint_offset = find_nearest_checkpoint(line_index, start_line)

    logger.debug(
        f'Getting lines {start_line}-{start_line + count - 1} from {source_path}, '
        f'starting from checkpoint line {checkpoint_line} at offset {checkpoint_offset}'
    )

    # Decompress from beginning (we can't seek in compressed stream without special tools)
    comp_format_str = index.compression_format or 'gzip'
    compression_format = CompressionFormat.from_string(comp_format_str)
    proc = decompress_to_stdout(source_path, compression_format)

    try:
        # Skip to checkpoint offset
        bytes_to_skip = checkpoint_offset
        while bytes_to_skip > 0:
            chunk_size = min(bytes_to_skip, 65536)
            chunk = proc.stdout.read(chunk_size)
            if not chunk:
                break
            bytes_to_skip -= len(chunk)

        # Now read lines from checkpoint_line to start_line + count
        current_line = checkpoint_line
        lines_to_skip = start_line - checkpoint_line
        result_lines = []

        # Read line by line
        buffer = b''
        while len(result_lines) < count:
            chunk = proc.stdout.read(4096)
            if not chunk:
                # Handle last line without newline
                if buffer and lines_to_skip <= 0:
                    result_lines.append(buffer.decode('utf-8', errors='replace'))
                break

            buffer += chunk

            while b'\n' in buffer:
                line, buffer = buffer.split(b'\n', 1)

                if lines_to_skip > 0:
                    lines_to_skip -= 1
                else:
                    result_lines.append(line.decode('utf-8', errors='replace'))
                    if len(result_lines) >= count:
                        break

                current_line += 1

        proc.stdout.close()
        proc.wait()

        return result_lines

    except Exception as e:
        proc.kill()
        raise RuntimeError(f'Failed to read lines from compressed file: {e}')


def get_decompressed_content_at_line(
    source_path: str | Path,
    line_number: int,
    context_before: int = 0,
    context_after: int = 0,
    index: 'UnifiedFileIndex | None' = None,
) -> list[str]:
    """Get content around a specific line in a compressed file.

    Args:
        source_path: Path to the compressed file
        line_number: Target line number (1-indexed)
        context_before: Number of lines before target
        context_after: Number of lines after target
        index: Optional pre-loaded UnifiedFileIndex

    Returns:
        List of lines (context_before + 1 + context_after lines)
    """
    start_line = max(1, line_number - context_before)
    total_lines = context_before + 1 + context_after

    # Adjust if we hit the beginning
    if line_number - context_before < 1:
        total_lines = line_number + context_after

    return get_decompressed_lines(source_path, start_line, total_lines, index)
