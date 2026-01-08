"""Unified file indexer module.

This module provides the FileIndexer class which creates UnifiedFileIndex
entries for files, combining line indexing and optional analysis.

Key behaviors:
- Without --analyze: Only index large files (>=50MB)
- With --analyze: Index ALL files with full analysis + anomaly detection
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from time import time

from rx import seekable_index, seekable_zstd
from rx.analyze import default_detectors
from rx.compression import detect_compression, is_compound_archive, is_compressed
from rx.file_utils import is_text_file
from rx.models import AnomalyRangeResult, FileType, UnifiedFileIndex


def is_indexable_file(filepath: str) -> bool:
    """Check if a file can be indexed (text or compressed text).

    We only index:
    - Plain text files (logs, source code, etc.)
    - Compressed text files (.gz, .zst, .xz, .bz2 containing text)

    We skip:
    - Binary files (images, videos, audio, executables, etc.)
    - Compound archives (.tar.gz, .tar.xz, etc.) that contain multiple files
    - Archives (zip, rar, 7z, etc.) that require extraction

    Args:
        filepath: Path to the file to check

    Returns:
        True if file can be indexed, False otherwise
    """
    # Skip compound archives like .tar.gz
    if is_compound_archive(filepath):
        return False

    # Seekable zstd files are indexable
    if seekable_zstd.is_seekable_zstd(filepath):
        return True

    # Other compressed files (gzip, xz, bz2) are indexable
    if is_compressed(filepath):
        return True

    # Plain text files are indexable
    if is_text_file(filepath):
        return True

    # Everything else (binary) is not indexable
    return False


from rx.unified_index import (
    build_index,
    get_index_step_bytes,
    load_index,
    needs_rebuild,
    save_index,
    should_create_index,
)


logger = logging.getLogger(__name__)


class IndexResult:
    """Result of indexing one or more files."""

    def __init__(self):
        self.indexed: list[UnifiedFileIndex] = []
        self.skipped: list[str] = []
        self.errors: list[tuple[str, str]] = []
        self.total_time: float = 0.0

    @property
    def count(self) -> int:
        return len(self.indexed)


class FileIndexer:
    """Creates unified file indexes with optional analysis.

    This class provides the main entry point for the `rx index` command
    and `/v1/index` API endpoint.

    Key behaviors:
    - analyze=False: Only create index for large files (>=50MB)
    - analyze=True: Create full index with analysis for ALL files
    """

    def __init__(self, analyze: bool = False, force: bool = False):
        """Initialize the indexer.

        Args:
            analyze: If True, run full analysis including anomaly detection
            force: If True, rebuild index even if valid cache exists
        """
        self.analyze = analyze
        self.force = force
        self.detectors = default_detectors() if analyze else []

    def index_file(self, filepath: str) -> UnifiedFileIndex | None:
        """Index a single file.

        Args:
            filepath: Path to file to index

        Returns:
            UnifiedFileIndex if successful, None if skipped or error
        """
        filepath = os.path.abspath(filepath)

        if not os.path.isfile(filepath):
            logger.warning(f'Not a file: {filepath}')
            return None

        # Check if file is indexable (text or compressed text)
        if not is_indexable_file(filepath):
            logger.debug(f'Skipping non-indexable file (binary/archive): {filepath}')
            return None

        try:
            stat = os.stat(filepath)
            file_size = stat.st_size

            # Check if we should create an index at all
            if not should_create_index(file_size, self.analyze):
                logger.debug(f'Skipping small file without --analyze: {filepath}')
                return None

            # Check for valid cached index
            if not self.force:
                cached = load_index(filepath)
                if cached and not needs_rebuild(filepath, cached, self.analyze):
                    logger.debug(f'Using cached index: {filepath}')
                    return cached

            # Build new index
            logger.debug(f'[INDEX] Starting index build for {filepath} ({file_size:,} bytes)')
            start_time = time()
            idx = self._build_index(filepath, stat)
            idx.build_time_seconds = time() - start_time

            # Log build summary
            logger.debug(
                f'[INDEX] Completed {filepath}: '
                f'{idx.line_count:,} lines in {idx.build_time_seconds:.2f}s'
            )
            if idx.anomalies:
                logger.debug(
                    f'[INDEX] Anomalies found: {len(idx.anomalies)} '
                    f'(summary: {idx.anomaly_summary})'
                )

            # Save to cache
            save_index(idx)

            return idx

        except Exception as e:
            logger.error(f'Failed to index {filepath}: {e}')
            return None

    def index_paths(
        self,
        paths: list[str],
        recursive: bool = True,
        max_workers: int = 10,
    ) -> IndexResult:
        """Index multiple files/directories in parallel.

        Args:
            paths: List of file or directory paths
            recursive: If True, recurse into directories
            max_workers: Maximum parallel workers

        Returns:
            IndexResult with all indexed files and statistics
        """
        result = IndexResult()
        start_time = time()

        # Collect all files to process
        files_to_process: list[str] = []
        for path in paths:
            path = os.path.abspath(path)
            if os.path.isfile(path):
                files_to_process.append(path)
            elif os.path.isdir(path):
                if recursive:
                    for root, _, files in os.walk(path):
                        for f in files:
                            files_to_process.append(os.path.join(root, f))
                else:
                    for f in os.listdir(path):
                        fp = os.path.join(path, f)
                        if os.path.isfile(fp):
                            files_to_process.append(fp)

        if not files_to_process:
            result.total_time = time() - start_time
            return result

        logger.debug(
            f'[INDEX] Processing {len(files_to_process)} files with {max_workers} workers '
            f'(analyze={self.analyze}, force={self.force})'
        )

        # Process files in parallel
        completed_count = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(self.index_file, f): f for f in files_to_process}

            for future in as_completed(future_to_file):
                filepath = future_to_file[future]
                completed_count += 1
                try:
                    idx = future.result()
                    if idx:
                        result.indexed.append(idx)
                        logger.debug(
                            f'[INDEX] [{completed_count}/{len(files_to_process)}] '
                            f'Indexed: {filepath}'
                        )
                    else:
                        result.skipped.append(filepath)
                        logger.debug(
                            f'[INDEX] [{completed_count}/{len(files_to_process)}] '
                            f'Skipped: {filepath}'
                        )
                except Exception as e:
                    result.errors.append((filepath, str(e)))
                    logger.debug(
                        f'[INDEX] [{completed_count}/{len(files_to_process)}] '
                        f'Error: {filepath}: {e}'
                    )

        result.total_time = time() - start_time
        logger.debug(
            f'[INDEX] Completed: {len(result.indexed)} indexed, '
            f'{len(result.skipped)} skipped, {len(result.errors)} errors '
            f'in {result.total_time:.2f}s'
        )
        return result

    def _build_index(self, filepath: str, stat: os.stat_result) -> UnifiedFileIndex:
        """Build a new index for a file.

        Args:
            filepath: Absolute path to file
            stat: os.stat result for the file

        Returns:
            UnifiedFileIndex with all computed data
        """
        # Determine file type
        file_type = self._detect_file_type(filepath)

        # Get file metadata
        mtime = datetime.fromtimestamp(stat.st_mtime).isoformat()
        permissions = oct(stat.st_mode)[-3:]

        try:
            import pwd

            owner = pwd.getpwuid(stat.st_uid).pw_name
        except (ImportError, KeyError):
            owner = str(stat.st_uid)

        # Create base index
        idx = UnifiedFileIndex(
            source_path=filepath,
            source_modified_at=mtime,
            source_size_bytes=stat.st_size,
            file_type=file_type,
            is_text=file_type == FileType.TEXT,
            permissions=permissions,
            owner=owner,
            analysis_performed=self.analyze,
        )

        # Handle different file types
        if file_type == FileType.BINARY:
            # Binary files - no line indexing
            return idx

        elif file_type == FileType.TEXT:
            # Regular text file
            self._build_text_index(filepath, idx)

        elif file_type == FileType.COMPRESSED:
            # Compressed file (gzip, xz, bz2)
            self._build_compressed_index(filepath, idx)

        elif file_type == FileType.SEEKABLE_ZSTD:
            # Seekable zstd file
            self._build_seekable_zstd_index(filepath, idx)

        return idx

    def _detect_file_type(self, filepath: str) -> FileType:
        """Detect the type of a file."""
        if seekable_zstd.is_seekable_zstd(filepath):
            return FileType.SEEKABLE_ZSTD
        elif is_compressed(filepath):
            comp = detect_compression(filepath)
            if comp:
                return FileType.COMPRESSED

        if is_text_file(filepath):
            return FileType.TEXT

        return FileType.BINARY

    def _build_text_index(self, filepath: str, idx: UnifiedFileIndex) -> None:
        """Build index for a regular text file.

        Args:
            filepath: Path to text file
            idx: UnifiedFileIndex to populate
        """
        # Use existing line_index module for line offset mapping
        try:
            build_result = build_index(filepath)
            if build_result:
                idx.line_index = build_result.line_index
                idx.index_step_bytes = get_index_step_bytes()

                # Copy analysis stats from build result (stats are directly on IndexBuildResult)
                idx.line_count = build_result.line_count
                idx.empty_line_count = build_result.empty_line_count
                idx.line_length_max = build_result.line_length_max
                idx.line_length_avg = build_result.line_length_avg
                idx.line_length_median = build_result.line_length_median
                idx.line_length_p95 = build_result.line_length_p95
                idx.line_length_p99 = build_result.line_length_p99
                idx.line_length_stddev = build_result.line_length_stddev
                idx.line_length_max_line_number = build_result.line_length_max_line_number
                idx.line_length_max_byte_offset = build_result.line_length_max_byte_offset
                idx.line_ending = build_result.line_ending

        except Exception as e:
            logger.warning(f'Failed to build line index for {filepath}: {e}')

        # Run anomaly detection if analyze mode
        if self.analyze and self.detectors:
            self._run_anomaly_detection(filepath, idx)

    def _build_compressed_index(self, filepath: str, idx: UnifiedFileIndex) -> None:
        """Build index for a compressed file.

        Args:
            filepath: Path to compressed file
            idx: UnifiedFileIndex to populate
        """
        comp = detect_compression(filepath)
        if comp:
            idx.compression_format = comp.value

        # Build compressed index (decompresses file to count lines and collect stats)
        # Note: We use build_compressed_index directly, NOT get_or_build_compressed_index,
        # because the latter saves a dict format that conflicts with UnifiedFileIndex format.
        # The FileIndexer.index_file() will save the final UnifiedFileIndex.
        try:
            from rx import compressed_index

            comp_idx = compressed_index.build_compressed_index(filepath)
            if comp_idx:
                idx.line_index = comp_idx.get('line_index', [])
                idx.decompressed_size_bytes = comp_idx.get('decompressed_size_bytes')
                idx.line_count = comp_idx.get('total_lines')

                if idx.decompressed_size_bytes and idx.source_size_bytes:
                    idx.compression_ratio = idx.decompressed_size_bytes / idx.source_size_bytes

                # Copy line statistics
                idx.empty_line_count = comp_idx.get('empty_line_count')
                idx.line_ending = comp_idx.get('line_ending')
                idx.line_length_max = comp_idx.get('line_length_max')
                idx.line_length_avg = comp_idx.get('line_length_avg')
                idx.line_length_median = comp_idx.get('line_length_median')
                idx.line_length_p95 = comp_idx.get('line_length_p95')
                idx.line_length_p99 = comp_idx.get('line_length_p99')
                idx.line_length_stddev = comp_idx.get('line_length_stddev')
                idx.line_length_max_line_number = comp_idx.get('line_length_max_line_number')
                idx.line_length_max_byte_offset = comp_idx.get('line_length_max_byte_offset')

        except Exception as e:
            logger.warning(f'Failed to build compressed index for {filepath}: {e}')

        # Run anomaly detection if analyze mode (FileAnalyzer handles decompression)
        if self.analyze and self.detectors:
            self._run_anomaly_detection(filepath, idx)

    def _build_seekable_zstd_index(self, filepath: str, idx: UnifiedFileIndex) -> None:
        """Build index for a seekable zstd file.

        Args:
            filepath: Path to seekable zstd file
            idx: UnifiedFileIndex to populate
        """
        idx.compression_format = 'zstd'

        try:
            # Use existing seekable_index module (now returns UnifiedFileIndex)
            szst_idx = seekable_index.get_or_build_index(filepath)
            if szst_idx:
                idx.line_index = szst_idx.line_index
                idx.frame_count = szst_idx.frame_count
                idx.frame_size_target = szst_idx.frame_size_target
                idx.decompressed_size_bytes = szst_idx.decompressed_size_bytes
                idx.line_count = szst_idx.line_count

                # Copy frames directly (already FrameLineInfo from UnifiedFileIndex)
                if szst_idx.frames:
                    idx.frames = szst_idx.frames

                if idx.decompressed_size_bytes and idx.source_size_bytes:
                    idx.compression_ratio = idx.decompressed_size_bytes / idx.source_size_bytes

        except Exception as e:
            logger.warning(f'Failed to build seekable zstd index: {e}')

    def _run_anomaly_detection(self, filepath: str, idx: UnifiedFileIndex) -> None:
        """Run anomaly detection on a file.

        This is a simplified version - the full implementation will use
        the existing FileAnalyzer anomaly detection logic.

        Args:
            filepath: Path to file
            idx: UnifiedFileIndex to populate with anomalies
        """
        try:
            from rx.analyzer import FileAnalyzer

            logger.debug(f'[ANALYZE] Starting anomaly detection for {filepath}')
            logger.debug(f'[ANALYZE] Using {len(self.detectors)} detectors: {[d.name for d in self.detectors]}')
            analyze_start = time()

            analyzer = FileAnalyzer(
                use_index_cache=False,
                detect_anomalies=True,
                anomaly_detectors=self.detectors,
            )

            result = analyzer.analyze_file(filepath, 'f1')
            analyze_time = time() - analyze_start

            logger.debug(
                f'[ANALYZE] Completed {filepath} in {analyze_time:.2f}s: '
                f'{result.line_count:,} lines, {len(result.anomalies)} anomalies'
            )

            # Copy anomalies to unified index, fixing line numbers from offsets
            if result.anomalies:
                idx.anomalies = []
                for a in result.anomalies:
                    start_line = a.start_line
                    end_line = a.end_line

                    # If line numbers are unknown (-1), calculate from offsets using line_index
                    if start_line == -1 and idx.line_index:
                        start_line = self._offset_to_line(a.start_offset, idx.line_index)
                    if end_line == -1 and idx.line_index:
                        end_line = self._offset_to_line(a.end_offset, idx.line_index)

                    idx.anomalies.append(
                        AnomalyRangeResult(
                            start_line=start_line,
                            end_line=end_line,
                            start_offset=a.start_offset,
                            end_offset=a.end_offset,
                            severity=a.severity,
                            category=a.category,
                            description=a.description,
                            detector=a.detector,
                        )
                    )
                idx.anomaly_summary = result.anomaly_summary

        except Exception as e:
            logger.warning(f'Anomaly detection failed for {filepath}: {e}')

    def _offset_to_line(self, offset: int, line_index: list[list[int]]) -> int:
        """Convert byte offset to line number using binary search on line_index.

        Note: This is an APPROXIMATION. The line_index only stores checkpoints
        at intervals (typically 1MB), so we estimate the line number based on
        average line length between checkpoints. This can be off by a few lines
        when line lengths vary significantly.

        Args:
            offset: Byte offset in the file
            line_index: List of [line_number, byte_offset] entries

        Returns:
            Approximate line number (1-based), or -1 if cannot be determined
        """
        if not line_index:
            return -1

        # Binary search for the checkpoint at or before this offset
        left, right = 0, len(line_index) - 1
        result_idx = 0

        while left <= right:
            mid = (left + right) // 2
            checkpoint_offset = line_index[mid][1]

            if checkpoint_offset <= offset:
                result_idx = mid
                left = mid + 1
            else:
                right = mid - 1

        # Get the checkpoint line number and offset
        checkpoint_line = line_index[result_idx][0]
        checkpoint_offset = line_index[result_idx][1]

        # Estimate additional lines based on average line length
        # Use next checkpoint to calculate average if available
        if result_idx + 1 < len(line_index):
            next_line = line_index[result_idx + 1][0]
            next_offset = line_index[result_idx + 1][1]
            lines_in_range = next_line - checkpoint_line
            bytes_in_range = next_offset - checkpoint_offset
            if bytes_in_range > 0 and lines_in_range > 0:
                avg_line_length = bytes_in_range / lines_in_range
                extra_bytes = offset - checkpoint_offset
                # Use round() instead of int() for better approximation
                extra_lines = round(extra_bytes / avg_line_length)
                # Clamp to not exceed the next checkpoint
                extra_lines = min(extra_lines, lines_in_range - 1)
                return checkpoint_line + extra_lines

        # Fallback: just return the checkpoint line
        return checkpoint_line
