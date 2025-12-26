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
import json
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

# Index file version for compatibility checking
INDEX_VERSION = 1


def get_compressed_index_dir() -> Path:
    """Get the directory for storing compressed file indexes.

    Returns:
        Path to the unified index directory
    """
    return get_rx_cache_dir('indexes')


def get_compressed_index_path(source_path: str | Path) -> Path:
    """Get the path to a compressed file's index.

    The index filename is based on a hash of the absolute source path.

    Args:
        source_path: Path to the compressed file

    Returns:
        Path to the index file
    """
    source_path = Path(source_path).resolve()
    path_hash = hashlib.sha256(str(source_path).encode()).hexdigest()[:16]
    filename = source_path.name

    index_dir = get_compressed_index_dir()
    return index_dir / f'{path_hash}_{filename}.json'


def load_compressed_index(source_path: str | Path) -> dict | None:
    """Load an existing compressed file index.

    Args:
        source_path: Path to the compressed file

    Returns:
        Index data dict, or None if not found or invalid
    """
    index_path = get_compressed_index_path(source_path)

    if not index_path.exists():
        return None

    try:
        with open(index_path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f'Failed to load compressed index {index_path}: {e}')
        return None


def is_compressed_index_valid(source_path: str | Path) -> bool:
    """Check if an index exists and is current for the source file.

    An index is valid if:
    1. It exists
    2. Version matches current INDEX_VERSION
    3. Source file mtime and size match

    Args:
        source_path: Path to the compressed file

    Returns:
        True if index is valid and current
    """
    source_path = Path(source_path)

    index_data = load_compressed_index(source_path)
    if index_data is None:
        return False

    # Check version
    if index_data.get('version') != INDEX_VERSION:
        logger.debug(f'Index version mismatch for {source_path}')
        return False

    # Check source file still exists and matches
    try:
        stat = source_path.stat()
        source_mtime = datetime.fromtimestamp(stat.st_mtime).isoformat()
        source_size = stat.st_size

        if index_data.get('source_size_bytes') != source_size:
            logger.debug(f'Source file size changed for {source_path}')
            return False

        if index_data.get('source_modified_at') != source_mtime:
            logger.debug(f'Source file mtime changed for {source_path}')
            return False

    except OSError:
        return False

    return True


def build_compressed_index(
    source_path: str | Path,
    progress_callback: Callable | None = None,
) -> dict:
    """Build a decompression + line index for a compressed file.

    This decompresses the entire file while tracking:
    - Total decompressed size
    - Line number to decompressed byte offset mapping (sampled)

    Args:
        source_path: Path to the compressed file
        progress_callback: Optional callback(bytes_processed, total_bytes) for progress

    Returns:
        Index data dictionary

    Raises:
        ValueError: If file is not compressed
        RuntimeError: If decompression fails
    """
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
    current_line = 1
    current_offset = 0

    # Add first line
    line_index.append([1, 0])

    try:
        while True:
            chunk = proc.stdout.read(65536)  # 64KB chunks
            if not chunk:
                break

            # Count newlines in chunk and track positions
            chunk_offset = 0
            while True:
                newline_pos = chunk.find(b'\n', chunk_offset)
                if newline_pos == -1:
                    break

                current_line += 1
                line_start_offset = current_offset + newline_pos + 1

                # Sample every Nth line
                if current_line % LINE_SAMPLE_INTERVAL == 0:
                    line_index.append([current_line, line_start_offset])

                chunk_offset = newline_pos + 1

            current_offset += len(chunk)

            if progress_callback:
                progress_callback(current_offset, None)

        proc.wait()

        if proc.returncode != 0:
            stderr = proc.stderr.read().decode() if proc.stderr else ''
            raise RuntimeError(f'Decompression failed: {stderr}')

    except Exception as e:
        proc.kill()
        raise RuntimeError(f'Failed to build index: {e}')

    elapsed = time.time() - start_time
    total_lines = current_line
    decompressed_size = current_offset

    logger.info(
        f'Built index for {source_path}: {total_lines} lines, '
        f'{decompressed_size} bytes decompressed, {len(line_index)} checkpoints, '
        f'{elapsed:.2f}s'
    )

    index_data = {
        'version': INDEX_VERSION,
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
    }

    return index_data


def save_compressed_index(index_data: dict, source_path: str | Path) -> Path:
    """Save a compressed file index to disk.

    Args:
        index_data: Index data dictionary
        source_path: Path to the compressed file (used to derive index path)

    Returns:
        Path where index was saved
    """
    index_path = get_compressed_index_path(source_path)

    # Ensure directory exists
    index_path.parent.mkdir(parents=True, exist_ok=True)

    with open(index_path, 'w') as f:
        json.dump(index_data, f, indent=2)

    logger.debug(f'Saved compressed index to {index_path}')
    return index_path


def get_or_build_compressed_index(
    source_path: str | Path,
    progress_callback: Callable | None = None,
) -> dict:
    """Get an existing valid index or build a new one.

    Args:
        source_path: Path to the compressed file
        progress_callback: Optional callback for build progress

    Returns:
        Index data dictionary
    """
    if is_compressed_index_valid(source_path):
        logger.debug(f'Using cached compressed index for {source_path}')
        return load_compressed_index(source_path)

    # Build new index
    index_data = build_compressed_index(source_path, progress_callback)
    save_compressed_index(index_data, source_path)
    return index_data


def find_nearest_checkpoint(index_data: dict, target_line: int) -> tuple[int, int]:
    """Find the nearest checkpoint at or before the target line.

    Args:
        index_data: Index data dictionary
        target_line: Target line number (1-indexed)

    Returns:
        Tuple of (checkpoint_line, checkpoint_offset)
    """
    line_index = index_data.get('line_index', [[1, 0]])

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
    index_data: dict | None = None,
) -> list[str]:
    """Get specific lines from a compressed file.

    Uses the index to seek to a nearby checkpoint, then decompresses
    forward to the target lines.

    Args:
        source_path: Path to the compressed file
        start_line: Starting line number (1-indexed)
        count: Number of lines to retrieve
        index_data: Optional pre-loaded index (will load/build if not provided)

    Returns:
        List of line strings (without newlines)
    """
    source_path = Path(source_path)

    # Get or build index
    if index_data is None:
        index_data = get_or_build_compressed_index(source_path)

    # Find nearest checkpoint
    checkpoint_line, checkpoint_offset = find_nearest_checkpoint(index_data, start_line)

    logger.debug(
        f'Getting lines {start_line}-{start_line + count - 1} from {source_path}, '
        f'starting from checkpoint line {checkpoint_line} at offset {checkpoint_offset}'
    )

    # Decompress from beginning (we can't seek in compressed stream without special tools)
    # For Tier 1, we decompress from start and skip to checkpoint
    compression_format = CompressionFormat.from_string(index_data.get('compression_format', 'gzip'))
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
    index_data: dict | None = None,
) -> list[str]:
    """Get content around a specific line in a compressed file.

    Args:
        source_path: Path to the compressed file
        line_number: Target line number (1-indexed)
        context_before: Number of lines before target
        context_after: Number of lines after target
        index_data: Optional pre-loaded index

    Returns:
        List of lines (context_before + 1 + context_after lines)
    """
    start_line = max(1, line_number - context_before)
    total_lines = context_before + 1 + context_after

    # Adjust if we hit the beginning
    if line_number - context_before < 1:
        total_lines = line_number + context_after

    return get_decompressed_lines(source_path, start_line, total_lines, index_data)


def delete_compressed_index(source_path: str | Path) -> bool:
    """Delete the index for a compressed file.

    Args:
        source_path: Path to the compressed file

    Returns:
        True if index was deleted, False if it didn't exist
    """
    index_path = get_compressed_index_path(source_path)

    if index_path.exists():
        index_path.unlink()
        logger.debug(f'Deleted compressed index {index_path}')
        return True

    return False


def list_compressed_indexes() -> list[dict]:
    """List all compressed file indexes.

    Returns:
        List of index metadata dictionaries
    """
    index_dir = get_compressed_index_dir()
    indexes = []

    for index_file in index_dir.glob('*.json'):
        try:
            with open(index_file) as f:
                data = json.load(f)
                indexes.append(
                    {
                        'index_path': str(index_file),
                        'source_path': data.get('source_path'),
                        'compression_format': data.get('compression_format'),
                        'decompressed_size_bytes': data.get('decompressed_size_bytes'),
                        'total_lines': data.get('total_lines'),
                        'created_at': data.get('created_at'),
                    }
                )
        except (OSError, json.JSONDecodeError):
            continue

    return indexes


def clear_compressed_indexes() -> int:
    """Delete all compressed file indexes.

    Returns:
        Number of indexes deleted
    """
    index_dir = get_compressed_index_dir()
    count = 0

    for index_file in index_dir.glob('*.json'):
        try:
            index_file.unlink()
            count += 1
        except OSError:
            pass

    logger.info(f'Cleared {count} compressed indexes')
    return count
