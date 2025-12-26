"""Unified index cache management.

This module provides cache management for the unified file index system.
All indexes are stored in $RX_CACHE_DIR/indexes/ (or ~/.cache/rx/indexes/).

Cache behavior:
- Without --analyze: Only cache large files (>=50MB) with line index only
- With --analyze: Cache ALL files with full analysis + anomaly detection
- Cache invalidation: Based on source file mtime and size
- Cache rebuild: When --analyze requested but analysis_performed=False
"""

import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from rx.models import UnifiedFileIndex
from rx.utils import get_rx_cache_dir


logger = logging.getLogger(__name__)

# Cache format version - increment when format changes
UNIFIED_INDEX_VERSION = 2

# Default threshold for creating indexes without --analyze (50MB)
DEFAULT_LARGE_FILE_MB = 50


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
        logger.debug(f'No cache found for {source_path}')
        return None

    try:
        with open(cache_path, encoding='utf-8') as f:
            data = json.load(f)

        # Check version
        if data.get('version') != UNIFIED_INDEX_VERSION:
            logger.debug(f'Cache version mismatch for {source_path}')
            return None

        index = UnifiedFileIndex(**data)

        # Validate against source file
        if not is_index_valid(source_path, index):
            logger.debug(f'Cache invalid (file changed) for {source_path}')
            return None

        logger.debug(f'Cache hit for {source_path}')
        return index

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.warning(f'Failed to load cache for {source_path}: {e}')
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
            f'Saved cache for {index.source_path} '
            f'({cache_size:,} bytes, {anomaly_count} anomalies, {line_index_count} checkpoints)'
        )
        return True

    except Exception as e:
        logger.warning(f'Failed to save cache for {index.source_path}: {type(e).__name__}: {e}')
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
            logger.info(f'Deleted cache for {source_path}')
            return True
        return False
    except OSError as e:
        logger.warning(f'Failed to delete cache for {source_path}: {e}')
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
            logger.warning(f'Failed to delete {cache_file}: {e}')

    logger.info(f'Cleared {count} index cache files')
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
